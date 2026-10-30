//! Surya-2.5 compact block output → HTML.
//!
//! Surya-ocr checkpoints (Datalab's `surya-ocr-2.5`, and anything else trained
//! on that contract) do not emit `<div data-label data-bbox>` markup. A
//! top-level block is a *label token* plus four *coordinate tokens* on a
//! 0–1000 grid, followed by the block's HTML with no `</div>`; `</td>`,
//! `</th>` and `</tr>` are omitted, a `data-bbox` on any other tag is written
//! as `<|box|>` plus four coordinates before that tag, and merged table cells
//! are `data-kind="parent"` / `"ghost"` cells with `data-cs` / `data-rs`.
//!
//! This is a port of `surya.inference.compact.decode` plus the
//! `surya.inference.merged_cells.convert_merged_cells` pass the surya client
//! runs afterwards. It exists so `serve` can answer a request in the shape
//! every downstream consumer of this model expects, without the caller having
//! to install the Python package.
//!
//! Raw output stays the default: the model card requires a server to return
//! the special tokens (`llama-server --special`), because surya's own client
//! parses exactly that string. Clients opt into the decoded form per request
//! (`ocr_format`), and a client that wants raw output — the surya package
//! itself — simply omits the field.

use serde_json::json;

/// Every label token that can head a top-level block (`LAYOUT_LABELS`).
pub(crate) const BLOCK_LABELS: &[&str] = &[
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
    "Picture",
];

/// Attribute ordering the Python decoder canonicalizes to when it has to
/// rebuild a tag (the `<|box|>` path). Anything unlisted sorts last by name.
const ATTR_ORDER: &[&str] = &[
    "type",
    "checked",
    "value",
    "colspan",
    "rowspan",
    "data-cs",
    "data-rs",
    "data-kind",
    "display",
    "class",
    "align",
    "style",
    "border",
    "href",
    "alt",
    "text",
];

const BOOLEAN_ATTRS: &[&str] = &["checked"];
const VOID_TAGS: &[&str] = &["br", "hr", "img", "input"];
/// End tags the compact format leaves out.
const OMIT_END: &[&str] = &["td", "th", "tr"];

/// Requested rendering of a compact-block assistant turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OcrFormat {
    /// `<div data-bbox="…" data-label="…">…</div>` per block, newline-joined.
    Html,
    /// JSON array of `{"label", "bbox", "html"}` (stringified into `content`).
    Blocks,
}

impl OcrFormat {
    /// Parse the request field's value. `None` is a client error.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "html" => Some(Self::Html),
            "blocks" => Some(Self::Blocks),
            _ => None,
        }
    }

    /// The value as accepted on the wire (for error text).
    pub(crate) fn wire_values() -> &'static str {
        "\"html\" or \"blocks\""
    }

    /// Map the serve-level default (`serve.ocr_format` /
    /// `HIPFIRE_SERVE_OCR_FORMAT`) to a rendering. `raw` is the identity, so
    /// it maps to `None`; an unknown value is `None` at the outer level.
    pub(crate) fn from_config(value: &str) -> Option<Option<Self>> {
        match value {
            "raw" | "" => Some(None),
            "html" => Some(Some(Self::Html)),
            "blocks" => Some(Some(Self::Blocks)),
            _ => None,
        }
    }

    /// The config spelling, for `/health` and logs.
    pub(crate) fn config_value(self) -> &'static str {
        match self {
            Self::Html => "html",
            Self::Blocks => "blocks",
        }
    }
}

/// One decoded block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) label: String,
    pub(crate) bbox: [u16; 4],
    pub(crate) html: String,
}

impl Block {
    fn bbox_string(&self) -> String {
        self.bbox
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn div(&self) -> String {
        format!(
            "<div data-bbox=\"{}\" data-label=\"{}\">{}</div>",
            self.bbox_string(),
            self.label,
            self.html
        )
    }

    fn json(&self) -> serde_json::Value {
        json!({ "label": self.label, "bbox": self.bbox, "html": self.html })
    }
}

/// Decode compact block output into the `<div data-label data-bbox>` HTML that
/// surya's parsers (`clean_block_html` / `parse_full_page_html`) read.
///
/// Output with no block headers is returned unchanged: a text-only answer is
/// not an OCR page, and rewriting it would hide that.
pub(crate) fn decode_html(text: &str) -> String {
    let decoded = Decoder::run(text);
    if decoded.blocks.is_empty() {
        return text.to_owned();
    }
    let mut html = decoded.leading;
    for block in &decoded.blocks {
        if !html.is_empty() && !html.ends_with('\n') {
            html.push('\n');
        }
        html.push_str(&block.div());
    }
    html
}

/// Same decode, rendered as a JSON array of blocks (stringified, because
/// OpenAI's `message.content` is a string). Output with no block headers is
/// passed through unchanged, matching [`decode_html`].
pub(crate) fn blocks_json(text: &str) -> String {
    let blocks = decode_blocks(text);
    if blocks.is_empty() {
        return text.to_owned();
    }
    let values: Vec<serde_json::Value> = blocks.iter().map(Block::json).collect();
    serde_json::Value::Array(values).to_string()
}

/// The decoded blocks, for callers that want them structured.
pub(crate) fn decode_blocks(text: &str) -> Vec<Block> {
    Decoder::run(text).blocks
}

// ---------------------------------------------------------------------------
// Token scanning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct TagRef<'a> {
    start: usize,
    end: usize,
    /// The tag text exactly as written.
    raw: &'a str,
    close: bool,
    /// Name exactly as written, e.g. `TD`.
    name_raw: &'a str,
    /// Attribute text between the name and the closing `>` (may lead with
    /// whitespace; excludes a trailing `/`).
    attrs: &'a str,
    selfclose: bool,
}

impl<'a> TagRef<'a> {
    fn name_lower(&self) -> String {
        self.name_raw.to_ascii_lowercase()
    }

}

/// One unit of model output.
#[derive(Debug)]
enum Token<'a> {
    Header { label: &'a str, bbox: [u16; 4] },
    Box([u16; 4]),
    Tag(TagRef<'a>),
    Text(&'a str),
}

/// A header token: `<|Label|>` + four coordinate tokens, all adjacent.
fn header_at(s: &str) -> Option<(usize, &str, [u16; 4])> {
    let rest = s.strip_prefix("<|")?;
    let close = rest.find("|>")?;
    let label = &rest[..close];
    if !BLOCK_LABELS.contains(&label) {
        return None;
    }
    let mut consumed = 2 + close + 2;
    let mut bbox = [0u16; 4];
    for slot in bbox.iter_mut() {
        let (len, value) = coord_at(&s[consumed..])?;
        *slot = value;
        consumed += len;
    }
    Some((consumed, label, bbox))
}

/// One coordinate token: `<|123|>` (1–4 digits).
fn coord_at(s: &str) -> Option<(usize, u16)> {
    let rest = s.strip_prefix("<|")?;
    let close = rest.find("|>")?;
    let digits = &rest[..close];
    if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((2 + close + 2, digits.parse().ok()?))
}

/// `<|box|>` + four coordinates: a `data-bbox` for the tag that follows.
fn box_at(s: &str) -> Option<(usize, [u16; 4])> {
    if !s.starts_with("<|box|>") {
        return None;
    }
    let mut consumed = "<|box|>".len();
    let mut bbox = [0u16; 4];
    for slot in bbox.iter_mut() {
        let (len, value) = coord_at(&s[consumed..])?;
        *slot = value;
        consumed += len;
    }
    Some((consumed, bbox))
}

/// A plain HTML tag: `<` [`/`] name attrs [`/`] `>`.
///
/// Mirrors `compact._TAG_RE`: the attribute run cannot contain `<` or `>`, and
/// a non-empty attribute run must start with whitespace. Malformed tags stay
/// literal text, which is what the Python decoder does too.
fn tag_at(s: &str) -> Option<TagRef<'_>> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut i = 1;
    let close = bytes.get(i) == Some(&b'/');
    if close {
        i += 1;
    }
    let name_start = i;
    if !matches!(bytes.get(i), Some(b) if b.is_ascii_alphabetic()) {
        return None;
    }
    i += 1;
    while matches!(bytes.get(i), Some(b) if b.is_ascii_alphanumeric() || *b == b'-') {
        i += 1;
    }
    let name_raw = &s[name_start..i];
    let gt = i + s[i..].find('>')?;
    let raw_attrs = &s[i..gt];
    if raw_attrs.contains('<') {
        return None;
    }
    let trimmed = raw_attrs.trim_end_matches([' ', '\t', '\n', '\r', '\x0c']);
    let (attrs, selfclose) = match trimmed.strip_suffix('/') {
        Some(prefix) => (prefix, true),
        None => (trimmed, false),
    };
    if !attrs.is_empty() && !attrs.starts_with([' ', '\t', '\n', '\r', '\x0c']) {
        return None;
    }
    Some(TagRef {
        start: 0,
        end: gt + 1,
        raw: &s[..gt + 1],
        close,
        name_raw,
        attrs,
        selfclose,
    })
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Decoded {
    leading: String,
    blocks: Vec<Block>,
}

struct Decoder {
    leading: String,
    blocks: Vec<Block>,
    /// Content of the block being decoded.
    out: Vec<String>,
    stack: Vec<String>,
    in_block: bool,
    label: String,
    bbox: [u16; 4],
    pending_box: Option<String>,
}

impl Decoder {
    fn run(text: &str) -> Decoded {
        let mut decoder = Self {
            leading: String::new(),
            blocks: Vec::new(),
            out: Vec::new(),
            stack: Vec::new(),
            in_block: false,
            label: String::new(),
            bbox: [0; 4],
            pending_box: None,
        };
        let mut pos = 0usize;
        while pos < text.len() {
            let rest = &text[pos..];
            if let Some((consumed, label, bbox)) = header_at(rest) {
                decoder.close_block();
                decoder.stack.clear();
                decoder.stack.push("div".to_owned());
                decoder.in_block = true;
                decoder.label = label.to_owned();
                decoder.bbox = bbox;
                decoder.pending_box = None;
                pos += consumed;
                continue;
            }
            if let Some((consumed, bbox)) = box_at(rest) {
                decoder.pending_box = Some(
                    bbox.iter()
                        .map(u16::to_string)
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                pos += consumed;
                continue;
            }
            if let Some(tag) = tag_at(rest) {
                let len = tag.end;
                let mut tag = tag;
                tag.start = pos;
                tag.end = pos + len;
                decoder.on_tag(tag);
                pos += len;
                continue;
            }
            // Unmatched `<` is text; consume up to the next candidate.
            let end = rest[1..].find('<').map_or(rest.len(), |i| i + 1);
            decoder.emit(&rest[..end]);
            pos += end;
        }
        decoder.close_block();
        Decoded {
            leading: decoder.leading,
            blocks: decoder.blocks,
        }
    }

    fn emit(&mut self, s: &str) {
        if self.in_block {
            self.out.push(s.to_owned());
        } else {
            self.leading.push_str(s);
        }
    }

    fn close_cells(&mut self, upto: &[&str]) {
        loop {
            let Some(top) = self.stack.last() else {
                break;
            };
            if !OMIT_END.contains(&top.as_str()) || upto.contains(&top.as_str()) {
                break;
            }
            let top = self.stack.pop().expect("top checked above");
            self.emit(&format!("</{top}>"));
        }
    }

    fn close_block(&mut self) {
        if !self.in_block {
            return;
        }
        self.close_cells(&[]);
        while let Some(top) = self.stack.last() {
            if !OMIT_END.contains(&top.as_str()) {
                break;
            }
            let top = self.stack.pop().expect("top checked above");
            self.emit(&format!("</{top}>"));
        }
        if self.stack.last().map(String::as_str) == Some("div") {
            self.stack.pop();
        }
        let inner = self.out.concat();
        self.blocks.push(Block {
            label: std::mem::take(&mut self.label),
            bbox: self.bbox,
            html: convert_merged_cells(&inner),
        });
        self.out.clear();
        self.in_block = false;
    }

    fn on_tag(&mut self, tag: TagRef<'_>) {
        let name = tag.name_lower();
        if tag.close {
            // A box token never applies across a closing tag.
            self.pending_box = None;
            if name == "table" || name == "thead" || name == "tbody" {
                let upto: &[&str] = if name == "table" {
                    &["table"]
                } else {
                    &["table", "thead", "tbody"]
                };
                self.close_cells(upto);
            }
            if name == "div" && self.in_block && self.stack.len() == 1 && self.stack[0] == "div" {
                self.close_block();
                return;
            }
            if let Some(match_at) = self.stack.iter().rposition(|open| *open == name) {
                while self.stack.len() > match_at + 1 {
                    let top = self.stack.pop().expect("above match");
                    if OMIT_END.contains(&top.as_str()) {
                        self.emit(&format!("</{top}>"));
                    }
                }
                self.stack.pop();
            }
            self.emit(tag.raw);
            return;
        }
        if name == "td" || name == "th" {
            if matches!(self.stack.last().map(String::as_str), Some("td" | "th")) {
                let top = self.stack.pop().expect("checked above");
                self.emit(&format!("</{top}>"));
            }
        } else if name == "tr" {
            self.close_cells(&["table", "thead", "tbody"]);
        } else if name == "thead" || name == "tbody" {
            self.close_cells(&["table"]);
        }
        let rendered = match self.pending_box.take() {
            Some(bbox) => {
                let attrs = if tag.attrs.trim().is_empty() {
                    format!(" data-bbox=\"{bbox}\"")
                } else {
                    format!("{} data-bbox=\"{bbox}\"", tag.attrs)
                };
                canonical_tag(false, tag.name_raw, &attrs, tag.selfclose)
            }
            None => tag.raw.to_owned(),
        };
        self.emit(&rendered);
        if !VOID_TAGS.contains(&name.as_str()) && !tag.selfclose {
            self.stack.push(name);
        }
    }
}

// ---------------------------------------------------------------------------
// Tag canonicalization (the `<|box|>` path)
// ---------------------------------------------------------------------------

fn attr_key(name: &str) -> (usize, &str) {
    match ATTR_ORDER.iter().position(|key| *key == name) {
        Some(index) => (index, ""),
        None => (ATTR_ORDER.len(), name),
    }
}

/// One spelling per tag, as `compact.canonical_tag` writes it: lowercase name,
/// fixed attribute order, double quotes, booleans valueless, void tags
/// self-closed.
fn canonical_tag(close: bool, name: &str, attrs: &str, selfclose: bool) -> String {
    let name = name.to_ascii_lowercase();
    if close {
        return format!("</{name}>");
    }
    let mut pairs = parse_attrs(attrs);
    pairs.sort_by(|left, right| attr_key(&left.0).cmp(&attr_key(&right.0)));
    let mut parts = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        if BOOLEAN_ATTRS.contains(&key.as_str()) {
            parts.push(key);
        } else if let Some(value) = value {
            parts.push(format!("{key}=\"{}\"", value.replace('"', "&quot;")));
        } else {
            parts.push(format!("{key}=\"\""));
        }
    }
    let inner = if parts.is_empty() {
        name.clone()
    } else {
        format!("{name} {}", parts.join(" "))
    };
    if VOID_TAGS.contains(&name.as_str()) || selfclose {
        format!("<{inner}/>")
    } else {
        format!("<{inner}>")
    }
}

/// Attribute names/values in an attribute run, mirroring `compact._ATTR_RE`.
fn parse_attrs(raw: &str) -> Vec<(String, Option<String>)> {
    let bytes = raw.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let is_name_start = |b: u8| b.is_ascii_alphabetic() || b == b'_' || b == b':';
        if !is_name_start(bytes[i]) {
            i += 1;
            continue;
        }
        let name_start = i;
        i += 1;
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric()
                || matches!(bytes[i], b'-' | b'_' | b':' | b'.'))
        {
            i += 1;
        }
        let name = raw[name_start..i].to_ascii_lowercase();
        let mut cursor = i;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let mut value = None;
        if cursor < bytes.len() && bytes[cursor] == b'=' {
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            if cursor < bytes.len() && matches!(bytes[cursor], b'"' | b'\'') {
                let quote = bytes[cursor];
                cursor += 1;
                let value_start = cursor;
                while cursor < bytes.len() && bytes[cursor] != quote {
                    cursor += 1;
                }
                value = Some(raw[value_start..cursor].to_owned());
                if cursor < bytes.len() {
                    cursor += 1;
                }
            } else {
                let value_start = cursor;
                while cursor < bytes.len()
                    && !bytes[cursor].is_ascii_whitespace()
                    && !matches!(bytes[cursor], b'"' | b'\'' | b'=' | b'<' | b'>' | b'`')
                {
                    cursor += 1;
                }
                value = Some(raw[value_start..cursor].to_owned());
            }
            i = cursor;
        }
        out.push((name, value));
    }
    out
}

// ---------------------------------------------------------------------------
// Merged cells (data-kind parent/ghost → colspan/rowspan)
// ---------------------------------------------------------------------------

/// Turn `data-kind="parent"` / `"ghost"` cells into real `colspan` / `rowspan`
/// and drop the ghost cells, as `merged_cells.convert_merged_cells` does.
///
/// Only top-level tables are converted (a table nested inside a cell of
/// another table keeps its own parent/ghost markup); the model does not emit
/// those, and surya's BeautifulSoup pass is the reference for anything else.
fn convert_merged_cells(html: &str) -> String {
    if !html.contains("data-kind") {
        return html.to_owned();
    }
    let tags = scan_tags(html);
    let mut edits: Vec<(usize, usize, Option<String>)> = Vec::new();
    let mut index = 0usize;
    while index < tags.len() {
        let tag = &tags[index];
        if tag.close || tag.selfclose || !tag.name_lower().eq("table") {
            index += 1;
            continue;
        }
        let Some(close) = matching_close(&tags, index, "table") else {
            index += 1;
            continue;
        };
        convert_table(&tags, index + 1, close, &mut edits);
        index = close + 1;
    }
    if edits.is_empty() {
        return html.to_owned();
    }
    edits.sort_by_key(|(start, _, _)| *start);
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0usize;
    for (start, end, replacement) in edits {
        if start < cursor {
            continue;
        }
        out.push_str(&html[cursor..start]);
        if let Some(text) = replacement {
            out.push_str(&text);
        }
        cursor = end;
    }
    out.push_str(&html[cursor..]);
    out
}

struct Cell {
    open: usize,
    close: usize,
    attrs: Vec<(String, Option<String>)>,
}

impl Cell {
    fn data_kind(&self) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(name, _)| name == "data-kind")
            .and_then(|(_, value)| value.as_deref())
    }
}

fn convert_table(
    tags: &[TagRef<'_>],
    body_start: usize,
    body_end: usize,
    edits: &mut Vec<(usize, usize, Option<String>)>,
) {
    let mut rows: Vec<(usize, Vec<Cell>)> = Vec::new();
    let mut index = body_start;
    while index < body_end {
        let tag = &tags[index];
        if tag.close || !tag.name_lower().eq("tr") {
            index += 1;
            continue;
        }
        let Some(row_close) = matching_close(tags, index, "tr") else {
            index += 1;
            continue;
        };
        let mut cells = Vec::new();
        let mut cursor = index + 1;
        while cursor < row_close {
            let cell = &tags[cursor];
            let name = cell.name_lower();
            if cell.close || !(name == "td" || name == "th") {
                cursor += 1;
                continue;
            }
            match matching_close(tags, cursor, &name) {
                Some(cell_close) => {
                    cells.push(Cell {
                        open: cursor,
                        close: cell_close,
                        attrs: parse_attrs(cell.attrs),
                    });
                    cursor = cell_close + 1;
                }
                None => cursor += 1,
            }
        }
        rows.push((index, cells));
        index = row_close + 1;
    }
    if rows.is_empty() {
        return;
    }
    let column_count = rows.iter().map(|(_, cells)| cells.len()).max().unwrap_or(0);
    if column_count == 0 {
        return;
    }
    let row_count = rows.len();
    let mut processed = vec![vec![false; column_count]; row_count];
    for row in 0..row_count {
        for column in 0..column_count {
            let Some(cell) = rows[row].1.get(column) else {
                processed[row][column] = true;
                continue;
            };
            if processed[row][column] {
                continue;
            }
            if cell.data_kind() != Some("parent") {
                processed[row][column] = true;
                continue;
            }
            let mut colspan = 1usize;
            while column + colspan < column_count {
                let neighbour = &rows[row].1.get(column + colspan);
                match neighbour {
                    Some(next) if next.data_kind() == Some("ghost") => colspan += 1,
                    _ => break,
                }
            }
            let mut rowspan = 1usize;
            while row + rowspan < row_count {
                let mut all_ghosts = true;
                for offset in 0..colspan {
                    match rows[row + rowspan].1.get(column + offset) {
                        Some(next) if next.data_kind() == Some("ghost") => {}
                        _ => {
                            all_ghosts = false;
                            break;
                        }
                    }
                }
                if !all_ghosts {
                    break;
                }
                rowspan += 1;
            }
            for span in 0..rowspan {
                for offset in 0..colspan {
                    if row + span < row_count && column + offset < column_count {
                        processed[row + span][column + offset] = true;
                    }
                }
            }
            let open = &tags[cell.open];
            let rebuilt = rebuild_cell_tag(open.name_raw, &cell.attrs, colspan, rowspan);
            edits.push((open.start, open.end, Some(rebuilt)));
        }
    }
    for (_, cells) in &rows {
        for cell in cells {
            if cell.data_kind() == Some("ghost") {
                let open = &tags[cell.open];
                let close = &tags[cell.close];
                edits.push((open.start, close.end, None));
            }
        }
    }
}

/// A parent cell's opening tag: `data-cs` / `data-rs` / `data-kind` dropped,
/// `colspan` / `rowspan` appended when they are real.
fn rebuild_cell_tag(
    name: &str,
    attrs: &[(String, Option<String>)],
    colspan: usize,
    rowspan: usize,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(attrs.len() + 2);
    for (key, value) in attrs {
        if matches!(key.as_str(), "data-cs" | "data-rs" | "data-kind") {
            continue;
        }
        match value {
            Some(value) => parts.push(format!("{key}=\"{}\"", value.replace('"', "&quot;"))),
            None => parts.push(key.clone()),
        }
    }
    if colspan > 1 {
        parts.push(format!("colspan=\"{colspan}\""));
    }
    if rowspan > 1 {
        parts.push(format!("rowspan=\"{rowspan}\""));
    }
    if parts.is_empty() {
        format!("<{name}>")
    } else {
        format!("<{name} {}>", parts.join(" "))
    }
}

/// Index of the `</name>` that closes the opening tag at `open`, counting
/// nested same-name elements.
fn matching_close(tags: &[TagRef<'_>], open: usize, name: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, tag) in tags.iter().enumerate().skip(open) {
        if !tag.name_lower().eq(name) || tag.selfclose {
            continue;
        }
        if tag.close {
            if depth == 0 {
                return Some(offset);
            }
            depth -= 1;
        } else if offset != open {
            depth += 1;
        }
    }
    None
}

fn scan_tags(html: &str) -> Vec<TagRef<'_>> {
    let mut tags = Vec::new();
    let mut pos = 0usize;
    while let Some(offset) = html[pos..].find('<') {
        let start = pos + offset;
        match tag_at(&html[start..]) {
            Some(mut tag) => {
                tag.start = start;
                tag.end = start + tag.end;
                pos = tag.end;
                tags.push(tag);
            }
            None => pos = start + 1,
        }
    }
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expectations generated with `surya.inference.compact.decode` (and
    /// `merged_cells.convert_merged_cells` where a table is involved).
    #[test]
    fn decodes_blocks_and_bboxes() {
        let input = "<|Text|><|41|><|25|><|128|><|54|><p>hello</p>\
                     <|Text|><|0|><|0|><|10|><|10|><p>world</p>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"41 25 128 54\" data-label=\"Text\"><p>hello</p></div>\n\
             <div data-bbox=\"0 0 10 10\" data-label=\"Text\"><p>world</p></div>"
        );
    }

    #[test]
    fn restores_box_attribute_on_the_following_tag() {
        let input = "<|Text|><|0|><|0|><|10|><|10|>\
                     <div><|box|><|1|><|2|><|3|><|4|><img src=\"x\"></div>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 10 10\" data-label=\"Text\">\
             <div><img data-bbox=\"1 2 3 4\" src=\"x\"/></div></div>"
        );
    }

    #[test]
    fn restores_omitted_cell_and_row_closers() {
        let input = "<|Table|><|1|><|2|><|3|><|4|>\
                     <table><tr><th>H1<th>H2<tr><td>a<td>b</table>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"1 2 3 4\" data-label=\"Table\"><table><tr><th>H1</th><th>H2</th></tr>\
             <tr><td>a</td><td>b</td></tr></table></div>"
        );
    }

    #[test]
    fn converts_merged_cells_to_colspan_and_drops_ghosts() {
        let input = "<|Table|><|10|><|10|><|90|><|90|>\
                     <table><tr><td data-cs=\"2\" data-kind=\"parent\">A<td data-kind=\"ghost\">B\
                     <tr><td data-kind=\"ghost\">C<td>D</table>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"10 10 90 90\" data-label=\"Table\"><table><tr>\
             <td colspan=\"2\">A</td></tr><tr><td>D</td></tr></table></div>"
        );
    }

    #[test]
    fn canonicalizes_a_boxed_cell_tag() {
        let input = "<|Table|><|1|><|1|><|9|><|9|><table><tr>\
                     <|box|><|2|><|2|><|8|><|3|><td data-kind=\"parent\" data-cs=\"2\">A\
                     <td data-kind=\"ghost\">B</table>";
        // The merged-cell pass runs after the box pass, so the cell carries the
        // restored `data-bbox` plus the derived span. Attribute order is not
        // part of the contract (surya reads this with BeautifulSoup).
        let html = decode_html(input);
        assert_eq!(
            html,
            "<div data-bbox=\"1 1 9 9\" data-label=\"Table\"><table><tr>\
             <td data-bbox=\"2 2 8 3\" colspan=\"2\">A</td></tr></table></div>"
        );
    }

    #[test]
    fn keeps_unrecognized_token_shaped_text() {
        let input = "<|Foo|><|1|><|1|><|2|><|2|><p>lit</p>";
        assert_eq!(decode_html(input), input);
        assert!(decode_blocks(input).is_empty());
    }

    #[test]
    fn blocks_render_as_json() {
        let input = "<|Text|><|0|><|0|><|10|><|10|><p>x</p>";
        assert_eq!(
            blocks_json(input),
            "[{\"label\":\"Text\",\"bbox\":[0,0,10,10],\"html\":\"<p>x</p>\"}]"
        );
    }

    #[test]
    fn nested_divs_inside_a_block_stay_balanced() {
        let input = "<|Text|><|0|><|0|><|5|><|5|><div>outer<div>inner</div>tail</div>";
        assert_eq!(
            decode_html(input),
            "<div data-bbox=\"0 0 5 5\" data-label=\"Text\">\
             <div>outer<div>inner</div>tail</div></div>"
        );
    }

    #[test]
    fn formats_parse_and_reject() {
        assert_eq!(OcrFormat::parse("html"), Some(OcrFormat::Html));
        assert_eq!(OcrFormat::parse("blocks"), Some(OcrFormat::Blocks));
        assert_eq!(OcrFormat::parse("HTML"), None);
        assert_eq!(OcrFormat::parse("json"), None);
    }

    #[test]
    fn config_values_map_and_reject() {
        assert_eq!(OcrFormat::from_config("raw"), Some(None));
        assert_eq!(OcrFormat::from_config(""), Some(None));
        assert_eq!(OcrFormat::from_config("html"), Some(Some(OcrFormat::Html)));
        assert_eq!(OcrFormat::from_config("blocks"), Some(Some(OcrFormat::Blocks)));
        assert_eq!(OcrFormat::from_config("markdown"), None);
        assert_eq!(OcrFormat::Html.config_value(), "html");
        assert_eq!(OcrFormat::Blocks.config_value(), "blocks");
    }
}
