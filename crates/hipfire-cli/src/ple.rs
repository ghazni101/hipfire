// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next PLE row sidecars: `hipfire ple derive|split`, the registry pull
//! plan (`hipfire pull --ple`), and the on-demand download of a missing
//! sidecar when `run`/`serve`/`bench` load a registry split model.
//!
//! A split base (`<name>-base.<ext>`) carries everything but the 128 PLE n-gram
//! shards; the rows live in `<base>.ple-q8` (exact Q8F16) and `<base>.ple-q4`
//! (4-bit Lloyd-Max). The decisions here are pure functions over the registry
//! entry and which files exist, so they are testable without a GPU or network.

use crate::Paths;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use hipfire_config::memory::{ple_sidecar_path, PleFormat, PleFormatRequest};
use hipfire_registry::{
    ModelEntry, RegistryV1, Sidecar, FLASH_NEXT_XTS_CANONICAL_FILE, FLASH_NEXT_XTS_LEGACY_FILE,
};
use hipfire_runtime::ple_convert::{self, DeriveReport, PleProgress, SplitReport};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Minimum spacing of two progress lines inside one conversion phase.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

/// Suffixes of files that belong to a model but are never models themselves.
const PLE_SIDECAR_SUFFIXES: [&str; 4] = [".ple-q4", ".ple-q8", ".plehot", ".plehotrows"];

/// `hipfire pull --ple` selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PleSelection {
    /// The format `memory.ple_format` and the local GPU arch prefer (q4 on
    /// gfx1151, q8 elsewhere); unknown arch means q8.
    Auto,
    /// The 4-bit Lloyd-Max rows.
    Q4,
    /// The exact Q8F16 rows.
    Q8,
    /// Both sidecars.
    All,
    /// The base only; the sidecar is fetched on first load.
    #[value(name = "none")]
    Skip,
}

#[derive(Args, Debug)]
pub(crate) struct PleArgs {
    #[command(subcommand)]
    action: PleAction,
}

#[derive(Subcommand, Debug)]
enum PleAction {
    /// Derive the 4-bit PLE rows (`<model>.ple-q4`) from the model's Q8 rows.
    Derive(DeriveArgs),
    /// Split a monolithic model into a PLE-less base plus `<base>.ple-q8`.
    Split(SplitArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DeriveFormat {
    Q4,
    Q8,
}

#[derive(Args, Debug)]
struct DeriveArgs {
    /// Row format to derive. Only q4 is derivable: q8 rows come from the
    /// model itself or from `hipfire ple split`.
    #[arg(long, value_enum, default_value_t = DeriveFormat::Q4)]
    format: DeriveFormat,
    /// Frequency rank (`.plehot`) that orders the derived rows. Default:
    /// `<model>.plehot` when present, else row order.
    #[arg(long, value_name = "PATH")]
    rank: Option<PathBuf>,
    /// Model path or registry tag.
    model: String,
}

#[derive(Args, Debug)]
struct SplitArgs {
    /// Base output path. Default: `-base` inserted before the model's extension
    /// (`qwen3.8-flash-next.mq4-xts` -> `qwen3.8-flash-next-base.mq4-xts`).
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,
    /// Keep the monolithic input after a verified split (default: remove it).
    #[arg(long)]
    keep_original: bool,
    /// Model path or registry tag.
    model: String,
}

pub(crate) fn ple_command(paths: &Paths, args: PleArgs) -> Result<()> {
    match args.action {
        PleAction::Derive(args) => derive_command(paths, args),
        PleAction::Split(args) => split_command(paths, args),
    }
}

// ---------------------------------------------------------------------------
// Pure decisions
// ---------------------------------------------------------------------------

/// The registry slot of `format`.
pub(crate) fn ple_slot(entry: &ModelEntry, format: PleFormat) -> Option<&Sidecar> {
    match format {
        PleFormat::Q4 => entry.ple_q4.as_ref(),
        PleFormat::Q8 => entry.ple_q8.as_ref(),
    }
}

fn other_format(format: PleFormat) -> PleFormat {
    match format {
        PleFormat::Q4 => PleFormat::Q8,
        PleFormat::Q8 => PleFormat::Q4,
    }
}

/// Sidecars a pull downloads next to the base, in download order.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct PlePullPlan<'a> {
    pub sidecars: Vec<(PleFormat, &'a Sidecar)>,
    /// Why the plan differs from the request (missing registry slot).
    pub notes: Vec<String>,
}

/// Choose the sidecars `hipfire pull --ple <selection>` downloads.
///
/// `Auto` follows `request` (the `memory.ple_format` config) resolved with the
/// local GPU arch. A missing preferred slot falls back to the other published
/// format for `Auto` and `Q4` (a q8 file lets the loader derive q4); an
/// explicit `Q8` never falls back, since q4 cannot produce exact rows.
pub(crate) fn plan_pull<'a>(
    entry: &'a ModelEntry,
    selection: PleSelection,
    request: PleFormatRequest,
    gpu_arch: &str,
) -> PlePullPlan<'a> {
    let (wanted, fallback): (Vec<PleFormat>, bool) = match selection {
        PleSelection::Skip => (Vec::new(), false),
        PleSelection::Q4 => (vec![PleFormat::Q4], true),
        PleSelection::Q8 => (vec![PleFormat::Q8], false),
        PleSelection::All => (vec![PleFormat::Q4, PleFormat::Q8], false),
        PleSelection::Auto => (vec![request.resolve(gpu_arch)], true),
    };
    let mut plan = PlePullPlan::default();
    for format in wanted {
        let chosen = match ple_slot(entry, format) {
            Some(sidecar) => Some((format, sidecar)),
            None => {
                let other = other_format(format);
                match ple_slot(entry, other).filter(|_| fallback) {
                    Some(sidecar) => {
                        let derive = if other == PleFormat::Q8 {
                            " (the loader derives q4 from it)"
                        } else {
                            ""
                        };
                        plan.notes.push(format!(
                            "registry publishes no ple-{} sidecar; fetching ple-{} instead{derive}",
                            format.name(),
                            other.name()
                        ));
                        Some((other, sidecar))
                    }
                    None => {
                        plan.notes.push(format!(
                            "registry publishes no ple-{} sidecar for this model",
                            format.name()
                        ));
                        None
                    }
                }
            }
        };
        if let Some((format, sidecar)) = chosen {
            if !plan.sidecars.iter().any(|(planned, _)| *planned == format) {
                plan.sidecars.push((format, sidecar));
            }
        }
    }
    plan
}

/// Which PLE sidecar files sit next to a split base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlePresence {
    pub q4: bool,
    pub q8: bool,
}

impl PlePresence {
    fn has(self, format: PleFormat) -> bool {
        match format {
            PleFormat::Q4 => self.q4,
            PleFormat::Q8 => self.q8,
        }
    }
}

/// The sidecar a load must download first, if any.
///
/// * `q8`: download ple-q8 when absent.
/// * `q4`: nothing when ple-q4 *or* ple-q8 is present (the loader derives q4
///   locally); otherwise download ple-q4, or ple-q8 when the registry
///   publishes no q4.
/// * `auto`: the arch's preferred format when present; else the other one when
///   present (auto never downloads over an available file); with neither
///   present, download the preferred format (or the other when unpublished).
///
/// The result always names a registry slot; `None` also covers "the registry
/// has nothing to offer", where the loader's own refusal explains it.
pub(crate) fn plan_on_demand(
    entry: &ModelEntry,
    request: PleFormatRequest,
    gpu_arch: &str,
    present: PlePresence,
) -> Option<PleFormat> {
    let (target, fallback) = match request {
        PleFormatRequest::Q8 => {
            if present.q8 {
                return None;
            }
            (PleFormat::Q8, false)
        }
        PleFormatRequest::Q4 => {
            if present.q4 || present.q8 {
                return None;
            }
            (PleFormat::Q4, true)
        }
        PleFormatRequest::Auto => {
            let preferred = request.resolve(gpu_arch);
            if present.has(preferred) || present.has(other_format(preferred)) {
                return None;
            }
            (preferred, true)
        }
    };
    if ple_slot(entry, target).is_some() {
        return Some(target);
    }
    let other = other_format(target);
    (fallback && ple_slot(entry, other).is_some()).then_some(other)
}

/// How a resolved model file relates to a registry split entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelLayout {
    /// The split base itself (`entry.file`).
    SplitBase,
    /// The grandfathered monolithic sibling (either legacy spelling).
    Monolithic,
    /// Anything else (a user path, a different artifact).
    Other,
}

pub(crate) fn classify_layout(
    registry: &RegistryV1,
    entry: &ModelEntry,
    file_name: &str,
) -> ModelLayout {
    if file_name == entry.file {
        return ModelLayout::SplitBase;
    }
    match registry.monolithic_sibling(entry) {
        Some((_, sibling))
            if file_name == sibling.file
                || (sibling.is_flash_next_xts() && file_name == FLASH_NEXT_XTS_LEGACY_FILE) =>
        {
            ModelLayout::Monolithic
        }
        _ => ModelLayout::Other,
    }
}

/// Local file names that serve a split `entry`, best first: the split base,
/// then its grandfathered monolithic sibling (and that pin's legacy spelling).
/// This is what lets a user who never migrated keep using the local
/// monolithic under the split tag instead of re-downloading.
pub(crate) fn split_model_files(registry: &RegistryV1, entry: &ModelEntry) -> Vec<String> {
    let mut files = vec![entry.file.clone()];
    if let Some((_, sibling)) = registry.monolithic_sibling(entry) {
        files.push(sibling.file.clone());
        if sibling.is_flash_next_xts() {
            files.push(FLASH_NEXT_XTS_LEGACY_FILE.to_owned());
        }
    }
    files
}

/// True for files that belong to a model without being one: the PLE row
/// sidecars, the hot-rank file and the hot-row packs.
pub(crate) fn is_ple_sidecar_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    PLE_SIDECAR_SUFFIXES
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}

/// The PLE row sidecars of `model` and their hot-row packs; `rm` removes the
/// ones that exist.
pub(crate) fn ple_sidecar_paths(model: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for format in [PleFormat::Q4, PleFormat::Q8] {
        let sidecar = ple_sidecar_path(model, format);
        let mut hot = sidecar.as_os_str().to_owned();
        hot.push(".plehotrows");
        paths.push(sidecar);
        paths.push(PathBuf::from(hot));
    }
    // The frequency rank (`<model>.plehot`) belongs to the model it ranks.
    let mut rank = model.as_os_str().to_owned();
    rank.push(".plehot");
    paths.push(PathBuf::from(rank));
    paths
}

fn format_gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn format_size(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "size unknown".to_owned(), format_gb)
}

/// One line of the pull plan.
pub(crate) fn plan_line(label: &str, file: &str, size: Option<u64>, state: &str) -> String {
    format!("  {label:<8} {file}  ({})  {state}", format_size(size))
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h{:02}m{:02}s", secs / 3600, secs % 3600 / 60, secs % 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

/// Throttles conversion progress to one line per phase every
/// [`PROGRESS_INTERVAL`], plus a line at each phase start and completion.
pub(crate) struct ProgressPrinter {
    phase: Option<&'static str>,
    phase_start: Instant,
    last_print: Instant,
    completed: bool,
}

impl ProgressPrinter {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            phase: None,
            phase_start: now,
            last_print: now,
            completed: false,
        }
    }

    /// The line to print for `progress` at `now`, if one is due.
    pub(crate) fn update(&mut self, progress: &PleProgress, now: Instant) -> Option<String> {
        let new_phase = self.phase != Some(progress.phase);
        if new_phase {
            self.phase = Some(progress.phase);
            self.phase_start = now;
            self.completed = false;
        }
        let finished = progress.total_bytes > 0 && progress.done_bytes >= progress.total_bytes;
        let due = new_phase
            || (finished && !self.completed)
            || now.saturating_duration_since(self.last_print) >= PROGRESS_INTERVAL;
        if !due {
            return None;
        }
        self.completed |= finished;
        self.last_print = now;
        let elapsed = now
            .saturating_duration_since(self.phase_start)
            .as_secs_f64();
        let rate = progress.done_bytes as f64 / elapsed.max(0.001);
        let percent = if progress.total_bytes > 0 {
            format!(
                "{:>5.1}%  ",
                progress.done_bytes as f64 / progress.total_bytes as f64 * 100.0
            )
        } else {
            String::new()
        };
        Some(format!(
            "ple: {}: {percent}{:.2}/{:.2} GB  {:.0} MB/s",
            progress.phase,
            progress.done_bytes as f64 / 1e9,
            progress.total_bytes as f64 / 1e9,
            rate / 1e6
        ))
    }
}

fn derive_summary(report: &DeriveReport) -> Vec<String> {
    vec![
        format!("derived q4 rows: {}", report.output.display()),
        format!(
            "  size            {} B ({})",
            report.bytes,
            format_gb(report.bytes)
        ),
        format!("  rows source     {}", report.source.display()),
        format!("  payload_sha256  {}", report.payload_sha256),
        format!("  q8_rows_sha256  {}", report.source_rows_sha256),
        format!(
            "  elapsed         {} ({} threads)",
            format_duration(report.elapsed),
            report.threads
        ),
        format!("  order           {}", report.order),
    ]
}

fn split_summary(report: &SplitReport) -> Vec<String> {
    vec![
        format!("split complete: base {}", report.base.display()),
        format!(
            "  base size        {} B ({})",
            report.base_bytes,
            format_gb(report.base_bytes)
        ),
        format!("  base_index_sha256 {}", report.base_index_sha256),
        format!("  ple-q8           {}", report.ple_q8.display()),
        format!(
            "  ple-q8 size      {} B ({})",
            report.ple_q8_bytes,
            format_gb(report.ple_q8_bytes)
        ),
        format!("  q8_rows_sha256   {}", report.q8_rows_sha256),
        format!("  elapsed          {}", format_duration(report.elapsed)),
    ]
}

/// Other hard links of the pinned monolithic (the canonical and legacy
/// spellings share one inode after a pull): removing only the named one would
/// not free the disk.
fn monolithic_twins(model: &Path) -> Vec<PathBuf> {
    let Some(dir) = model.parent() else {
        return Vec::new();
    };
    [FLASH_NEXT_XTS_CANONICAL_FILE, FLASH_NEXT_XTS_LEGACY_FILE]
        .into_iter()
        .map(|name| dir.join(name))
        .filter(|twin| twin != model && twin.is_file() && crate::same_installed_file(twin, model))
        .collect()
}

/// `<model>.plehot` / `.plehotrows` files next to `model` that exist.
fn plehot_siblings(model: &Path) -> Vec<PathBuf> {
    [".plehot", ".plehotrows"]
        .into_iter()
        .map(|suffix| {
            let mut path = model.as_os_str().to_owned();
            path.push(suffix);
            PathBuf::from(path)
        })
        .filter(|path| path.is_file())
        .collect()
}

// ---------------------------------------------------------------------------
// Environment-dependent helpers
// ---------------------------------------------------------------------------

/// The GPU arch the format preference is resolved with; empty when none is
/// detected (unknown arch resolves to q8).
pub(crate) fn local_gpu_arch() -> String {
    crate::detect_gpu_arches()
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// `memory.ple_format` from a resolved config; `auto` when absent or invalid.
pub(crate) fn request_from_config(resolved: &hipfire_config::ResolvedConfig) -> PleFormatRequest {
    crate::config_string(resolved, "memory.ple_format")
        .ok()
        .and_then(|value| PleFormatRequest::parse(&value))
        .unwrap_or(PleFormatRequest::Auto)
}

/// `memory.ple_format` from the global config + environment (pull has no
/// model context).
pub(crate) fn request_from_global(paths: &Paths) -> PleFormatRequest {
    crate::resolved_global(paths, true)
        .map(|(_, resolved)| request_from_config(&resolved))
        .unwrap_or(PleFormatRequest::Auto)
}

/// Print the pull plan for a split entry and return the sidecars to fetch.
pub(crate) fn announce_pull_plan<'a>(
    paths: &Paths,
    registry: &RegistryV1,
    entry: &'a ModelEntry,
    selection: PleSelection,
    base_present: bool,
) -> PlePullPlan<'a> {
    let request = if selection == PleSelection::Auto {
        request_from_global(paths)
    } else {
        PleFormatRequest::Auto
    };
    let arch = local_gpu_arch();
    let plan = plan_pull(entry, selection, request, &arch);
    eprintln!("Pull plan (--ple {}):", selection_name(selection));
    eprintln!(
        "{}",
        plan_line(
            "base",
            &entry.file,
            entry.size_bytes,
            if base_present { "present" } else { "download" }
        )
    );
    for (format, sidecar) in &plan.sidecars {
        let state = if paths.models.join(&sidecar.file).is_file() {
            "present (verified)"
        } else {
            "download"
        };
        eprintln!(
            "{}",
            plan_line(
                &format!("ple-{}", format.name()),
                &sidecar.file,
                sidecar.size_bytes,
                state
            )
        );
    }
    for note in &plan.notes {
        eprintln!("  note: {note}");
    }
    if plan.sidecars.is_empty() && selection != PleSelection::Skip {
        eprintln!("  note: no PLE sidecar will be downloaded");
    }
    if selection == PleSelection::Skip {
        eprintln!("  (the selected sidecar is fetched on first load)");
    }
    if !base_present {
        let local_monolithic = registry
            .monolithic_sibling(entry)
            .into_iter()
            .flat_map(|(_, sibling)| {
                std::iter::once(sibling.file.as_str()).chain(
                    sibling
                        .is_flash_next_xts()
                        .then_some(FLASH_NEXT_XTS_LEGACY_FILE),
                )
            })
            .find(|file| paths.models.join(file).is_file());
        if let Some(file) = local_monolithic {
            eprintln!(
                "  note: the local monolithic {file} already loads under this tag; \
                 `hipfire ple split` converts it in place without downloading the base"
            );
        }
    }
    plan
}

fn selection_name(selection: PleSelection) -> &'static str {
    match selection {
        PleSelection::Auto => "auto",
        PleSelection::Q4 => "q4",
        PleSelection::Q8 => "q8",
        PleSelection::All => "all",
        PleSelection::Skip => "none",
    }
}

/// Download one PLE sidecar to `destination` unless a verified copy exists.
pub(crate) fn fetch_ple_sidecar(
    entry: &ModelEntry,
    format: PleFormat,
    sidecar: &Sidecar,
    destination: &Path,
    force: bool,
) -> Result<()> {
    if destination.exists() && !force {
        if crate::existing_artifact_valid(
            destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
        ) {
            eprintln!(
                "  ple-{} sidecar already present: {}",
                format.name(),
                sidecar.file
            );
            return Ok(());
        }
        eprintln!(
            "  ple-{} sidecar stale, refreshing: {}",
            format.name(),
            sidecar.file
        );
    }
    eprintln!(
        "  Fetching ple-{} sidecar: {} ({})",
        format.name(),
        sidecar.file,
        format_size(sidecar.size_bytes)
    );
    crate::download_verified(
        &crate::artifact_url(entry, &sidecar.file),
        destination,
        sidecar.sha256.as_deref(),
        sidecar.size_bytes,
        false,
    )
    .with_context(|| format!("downloading {}", sidecar.file))
}

/// Make the PLE rows of a resolved registry model available before a load:
/// downloads the selected sidecar when it (and every acceptable substitute) is
/// missing, per [`plan_on_demand`]. A no-op for models without PLE slots, for
/// a split entry served from its grandfathered monolithic (one note), and for
/// paths that are not the registry artifact.
///
/// `allow_download = false` (a serve request with downloads disabled) turns a
/// needed download into an error naming the pull command.
pub(crate) fn ensure_ple_sidecar(
    registry: &RegistryV1,
    entry: Option<&ModelEntry>,
    tag: Option<&str>,
    model_path: &Path,
    request: PleFormatRequest,
    allow_download: bool,
) -> Result<()> {
    let Some(entry) = entry.filter(|entry| entry.has_ple_sidecars()) else {
        return Ok(());
    };
    let file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let label = tag.unwrap_or(&entry.file);
    match classify_layout(registry, entry, file_name) {
        ModelLayout::SplitBase => {}
        ModelLayout::Monolithic => {
            note_monolithic(label, file_name);
            return Ok(());
        }
        ModelLayout::Other => return Ok(()),
    }
    let present = PlePresence {
        q4: ple_sidecar_path(model_path, PleFormat::Q4).is_file(),
        q8: ple_sidecar_path(model_path, PleFormat::Q8).is_file(),
    };
    let Some(format) = plan_on_demand(entry, request, &local_gpu_arch(), present) else {
        return Ok(());
    };
    let sidecar = ple_slot(entry, format)
        .ok_or_else(|| anyhow!("registry slot ple-{} vanished", format.name()))?;
    if !allow_download {
        bail!(
            "ple: {} is missing; run `hipfire pull {label} --ple {}` on the server first \
             ({}; downloads named by a request are off: serve.allow_request_pull)",
            sidecar.file,
            format.name(),
            format_size(sidecar.size_bytes)
        );
    }
    eprintln!(
        "ple: downloading {} ({}) for memory.ple_format={}",
        sidecar.file,
        format_size(sidecar.size_bytes),
        request.name()
    );
    let destination = ple_sidecar_path(model_path, format);
    crate::download_verified(
        &crate::artifact_url(entry, &sidecar.file),
        &destination,
        sidecar.sha256.as_deref(),
        sidecar.size_bytes,
        false,
    )
    .with_context(|| format!("downloading {}", sidecar.file))
}

fn note_monolithic(label: &str, file: &str) {
    static NOTED: std::sync::Once = std::sync::Once::new();
    NOTED.call_once(|| {
        eprintln!(
            "note: serving the local monolithic {file} for {label}; `hipfire ple split {label}` \
             or `hipfire pull {label}` gets the split PLE layout (base + swappable PLE rows)"
        );
    });
}

// ---------------------------------------------------------------------------
// hipfire ple derive / split
// ---------------------------------------------------------------------------

fn resolve_local_model(paths: &Paths, model: &str) -> Result<PathBuf> {
    let registry = hipfire_registry::load(&paths.registry).registry;
    crate::resolve_model_path(paths, &registry, model)?.ok_or_else(|| {
        anyhow!("model not found locally: {model}; pass a model path or `hipfire pull` it first")
    })
}

fn derive_command(paths: &Paths, args: DeriveArgs) -> Result<()> {
    if args.format == DeriveFormat::Q8 {
        bail!(
            "q8 rows cannot be derived: they are the model's own exact rows (embedded in a \
             monolithic model) or the `<base>.ple-q8` written by `hipfire ple split`; only \
             `--format q4` derives (from the q8 rows)"
        );
    }
    let model = resolve_local_model(paths, &args.model)?;
    eprintln!("ple: deriving q4 rows for {}", model.display());
    let mut printer = ProgressPrinter::new(Instant::now());
    let mut report_progress = |progress: &PleProgress| {
        if let Some(line) = printer.update(progress, Instant::now()) {
            eprintln!("{line}");
        }
    };
    let report = ple_convert::derive_q4(&model, args.rank.as_deref(), &mut report_progress)
        .map_err(|error| anyhow!(error))?;
    for line in derive_summary(&report) {
        println!("{line}");
    }
    Ok(())
}

fn split_command(paths: &Paths, args: SplitArgs) -> Result<()> {
    let model = resolve_local_model(paths, &args.model)?;
    let base = args
        .out
        .clone()
        .unwrap_or_else(|| ple_convert::default_split_base_path(&model));
    if base == model {
        bail!(
            "split output {} is the input model; pass --out to name a different base",
            base.display()
        );
    }
    let q8 = ple_sidecar_path(&base, PleFormat::Q8);
    for output in [&base, &q8] {
        if output.exists() {
            bail!(
                "split output {} already exists; remove it or pass --out",
                output.display()
            );
        }
    }
    eprintln!(
        "ple: splitting {} into {} + {}",
        model.display(),
        base.display(),
        q8.display()
    );
    let mut printer = ProgressPrinter::new(Instant::now());
    let mut report_progress = |progress: &PleProgress| {
        if let Some(line) = printer.update(progress, Instant::now()) {
            eprintln!("{line}");
        }
    };
    let report = ple_convert::split(&model, args.out.as_deref(), &mut report_progress)
        .map_err(|error| anyhow!(error))?;
    for line in split_summary(&report) {
        println!("{line}");
    }
    let plehot = plehot_siblings(&model);
    let stale_q4 = ple_sidecar_path(&model, PleFormat::Q4);
    if args.keep_original {
        println!("kept original: {}", model.display());
    } else {
        // The twins share the monolithic's inode; list them before removal.
        let mut removed = monolithic_twins(&model);
        removed.insert(0, model.clone());
        for path in removed {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            println!("removed original: {}", path.display());
        }
    }
    if !plehot.is_empty() {
        let names: Vec<String> = plehot
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        println!(
            "note: {} belong to the monolithic layout and were left in place; they do not \
             apply to the split base",
            names.join(", ")
        );
    }
    if stale_q4.is_file() {
        println!(
            "note: {} is bound to the monolithic and does not load with the split base; \
             delete it or run `hipfire ple derive {}`",
            stale_q4.display(),
            report.base.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Commands};
    use clap::Parser;

    fn sidecar(file: &str) -> Sidecar {
        Sidecar {
            file: file.into(),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(1_000_000_000),
        }
    }

    fn split_entry(q4: bool, q8: bool) -> ModelEntry {
        ModelEntry {
            repo: "hipfire-models/qwen3.8-flash-next".into(),
            file: "qwen3.8-flash-next-base.mq4-xts".into(),
            ple_q4: q4.then(|| sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q4")),
            ple_q8: q8.then(|| sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q8")),
            ..Default::default()
        }
    }

    fn formats(plan: &PlePullPlan<'_>) -> Vec<PleFormat> {
        plan.sidecars.iter().map(|(format, _)| *format).collect()
    }

    const NONE: PlePresence = PlePresence {
        q4: false,
        q8: false,
    };
    const Q4: PlePresence = PlePresence {
        q4: true,
        q8: false,
    };
    const Q8: PlePresence = PlePresence {
        q4: false,
        q8: true,
    };
    const BOTH: PlePresence = PlePresence { q4: true, q8: true };

    #[test]
    fn pull_selection_picks_files_per_flag_and_arch() {
        let entry = split_entry(true, true);
        let pick =
            |selection, request, arch: &str| formats(&plan_pull(&entry, selection, request, arch));
        let auto = PleFormatRequest::Auto;
        assert_eq!(pick(PleSelection::Auto, auto, "gfx1151"), [PleFormat::Q4]);
        assert_eq!(pick(PleSelection::Auto, auto, "gfx1100"), [PleFormat::Q8]);
        assert_eq!(
            pick(PleSelection::Auto, auto, ""),
            [PleFormat::Q8],
            "unknown arch is q8"
        );
        // The configured request wins over the arch default.
        assert_eq!(
            pick(PleSelection::Auto, PleFormatRequest::Q8, "gfx1151"),
            [PleFormat::Q8]
        );
        assert_eq!(
            pick(PleSelection::Auto, PleFormatRequest::Q4, "gfx1100"),
            [PleFormat::Q4]
        );
        // Explicit selections ignore config and arch.
        assert_eq!(pick(PleSelection::Q4, auto, "gfx1100"), [PleFormat::Q4]);
        assert_eq!(pick(PleSelection::Q8, auto, "gfx1151"), [PleFormat::Q8]);
        assert_eq!(
            pick(PleSelection::All, auto, "gfx1100"),
            [PleFormat::Q4, PleFormat::Q8]
        );
        assert!(pick(PleSelection::Skip, auto, "gfx1151").is_empty());
        let plan = plan_pull(&entry, PleSelection::All, auto, "gfx1100");
        assert_eq!(
            plan.sidecars[1].1.file,
            "qwen3.8-flash-next-base.mq4-xts.ple-q8"
        );
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn pull_selection_handles_unpublished_slots() {
        let q8_only = split_entry(false, true);
        let auto = PleFormatRequest::Auto;
        // q4 falls back to q8 (the loader derives q4 from it), with a note.
        for selection in [PleSelection::Q4, PleSelection::Auto] {
            let plan = plan_pull(&q8_only, selection, auto, "gfx1151");
            assert_eq!(formats(&plan), [PleFormat::Q8], "{selection:?}");
            assert!(plan.notes[0].contains("derives q4"), "{:?}", plan.notes);
        }
        // `all` takes what exists and explains what does not.
        let plan = plan_pull(&q8_only, PleSelection::All, auto, "gfx1151");
        assert_eq!(formats(&plan), [PleFormat::Q8]);
        assert_eq!(plan.notes.len(), 1);

        let q4_only = split_entry(true, false);
        // An explicit q8 never degrades to q4.
        let plan = plan_pull(&q4_only, PleSelection::Q8, auto, "gfx1151");
        assert!(plan.sidecars.is_empty());
        assert!(plan.notes[0].contains("no ple-q8"));
        // Auto prefers q8 off gfx1151 but takes the published q4.
        let plan = plan_pull(&q4_only, PleSelection::Auto, auto, "gfx1100");
        assert_eq!(formats(&plan), [PleFormat::Q4]);
        assert_eq!(plan.notes.len(), 1);

        let bare = split_entry(false, false);
        assert!(plan_pull(&bare, PleSelection::Auto, auto, "gfx1151")
            .sidecars
            .is_empty());
    }

    #[test]
    fn on_demand_follows_the_request_and_never_downloads_over_a_usable_file() {
        let entry = split_entry(true, true);
        let go = |request, arch: &str, present| plan_on_demand(&entry, request, arch, present);
        // q8: present -> nothing; absent -> download, even when q4 is on disk.
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", Q8), None);
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", BOTH), None);
        assert_eq!(
            go(PleFormatRequest::Q8, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", Q4), Some(PleFormat::Q8));
        // q4: ple-q4 or ple-q8 on disk is enough (the loader derives); else download q4.
        assert_eq!(go(PleFormatRequest::Q4, "gfx1100", Q4), None);
        assert_eq!(go(PleFormatRequest::Q4, "gfx1100", Q8), None);
        assert_eq!(
            go(PleFormatRequest::Q4, "gfx1100", NONE),
            Some(PleFormat::Q4)
        );
        // auto: preferred present, or the other present -> nothing; neither -> preferred.
        assert_eq!(go(PleFormatRequest::Auto, "gfx1151", Q4), None);
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1151", Q8),
            None,
            "falls back, no download"
        );
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1100", Q4),
            None,
            "falls back, no download"
        );
        assert_eq!(go(PleFormatRequest::Auto, "gfx1100", Q8), None);
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1151", NONE),
            Some(PleFormat::Q4)
        );
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1100", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(go(PleFormatRequest::Auto, "", NONE), Some(PleFormat::Q8));
    }

    #[test]
    fn on_demand_substitutes_the_published_slot_and_never_names_a_missing_one() {
        let q8_only = split_entry(false, true);
        // q4 requested, nothing local, registry has only q8: fetch q8 (derive q4 locally).
        assert_eq!(
            plan_on_demand(&q8_only, PleFormatRequest::Q4, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(
            plan_on_demand(&q8_only, PleFormatRequest::Auto, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        let q4_only = split_entry(true, false);
        assert_eq!(
            plan_on_demand(&q4_only, PleFormatRequest::Auto, "gfx1100", NONE),
            Some(PleFormat::Q4)
        );
        // Exact q8 rows cannot come from a q4-only registry: let the loader refuse.
        assert_eq!(
            plan_on_demand(&q4_only, PleFormatRequest::Q8, "gfx1100", NONE),
            None
        );
        let bare = split_entry(false, false);
        assert_eq!(
            plan_on_demand(&bare, PleFormatRequest::Auto, "gfx1151", NONE),
            None
        );
    }

    fn registry_with_split() -> RegistryV1 {
        let pinned = |file: &str| {
            serde_json::json!({
                "repo": "hipfire-models/qwen3.8-flash-next", "file": file, "size_gb": 125.29,
                "min_vram_gb": 8.0, "desc": "pin",
                "sha256": "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972",
                "size_bytes": 125288544792_u64
            })
        };
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts",
                    "size_gb": 70.9, "min_vram_gb": 8.0, "desc": "split",
                    "ple_q8": {"file": "qwen3.8-flash-next-base.mq4-xts.ple-q8"}
                },
                "qwen3.8:flash-next-mq4-xts": pinned("qwen3.8-flash-next.mq4-xts"),
                "unrelated": {"repo": "x/y", "file": "plain.mq4", "size_gb": 1, "min_vram_gb": 1, "desc": "d"}
            },
            "aliases": {}
        });
        RegistryV1::parse(&value.to_string(), "ple test").unwrap()
    }

    #[test]
    fn layout_classification_and_tag_fallback_files() {
        let registry = registry_with_split();
        let split = &registry.models["qwen3.8:flash-next"];
        assert_eq!(
            classify_layout(&registry, split, "qwen3.8-flash-next-base.mq4-xts"),
            ModelLayout::SplitBase
        );
        for name in ["qwen3.8-flash-next.mq4-xts", FLASH_NEXT_XTS_LEGACY_FILE] {
            assert_eq!(
                classify_layout(&registry, split, name),
                ModelLayout::Monolithic,
                "{name}"
            );
        }
        assert_eq!(
            classify_layout(&registry, split, "plain.mq4"),
            ModelLayout::Other
        );
        assert_eq!(
            split_model_files(&registry, split),
            [
                "qwen3.8-flash-next-base.mq4-xts",
                "qwen3.8-flash-next.mq4-xts",
                FLASH_NEXT_XTS_LEGACY_FILE
            ]
        );
        // An entry without a catalogued sibling serves only its own file.
        let plain = &registry.models["unrelated"];
        assert_eq!(split_model_files(&registry, plain), ["plain.mq4"]);
    }

    #[test]
    fn ensure_is_a_noop_without_a_split_entry_or_for_other_layouts() {
        let registry = registry_with_split();
        let request = PleFormatRequest::Q8;
        // No entry, a plain entry, and the monolithic fallback never download.
        ensure_ple_sidecar(
            &registry,
            None,
            None,
            Path::new("/nonexistent/x.mq4"),
            request,
            true,
        )
        .unwrap();
        let plain = &registry.models["unrelated"];
        ensure_ple_sidecar(
            &registry,
            Some(plain),
            None,
            Path::new("/nonexistent/plain.mq4"),
            request,
            true,
        )
        .unwrap();
        let split = &registry.models["qwen3.8:flash-next"];
        ensure_ple_sidecar(
            &registry,
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next.mq4-xts"),
            request,
            false,
        )
        .unwrap();
        ensure_ple_sidecar(
            &registry,
            Some(split),
            None,
            Path::new("/nonexistent/other.mq4"),
            request,
            false,
        )
        .unwrap();
    }

    #[test]
    fn ensure_refuses_a_needed_download_when_downloads_are_off() {
        let registry = registry_with_split();
        let split = &registry.models["qwen3.8:flash-next"];
        let error = ensure_ple_sidecar(
            &registry,
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next-base.mq4-xts"),
            PleFormatRequest::Q8,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("hipfire pull qwen3.8:flash-next --ple q8"),
            "{error}"
        );
        assert!(error.contains("serve.allow_request_pull"), "{error}");
        // q4 with nothing local and a q8-only registry points at the q8 file.
        let error = ensure_ple_sidecar(
            &registry,
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next-base.mq4-xts"),
            PleFormatRequest::Q4,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--ple q8"), "{error}");
    }

    #[test]
    fn sidecar_classification_covers_rows_hot_rank_and_hot_packs() {
        for name in [
            "m.mq4-xts.ple-q4",
            "m.mq4-xts.ple-q8",
            "M.MQ4-XTS.PLE-Q8",
            "m.mq4-xts.plehot",
            "m.mq4-xts.plehotrows",
            "m.mq4-xts.ple-q4.plehotrows",
            "m.mq4-xts.ple-q8.plehotrows",
        ] {
            assert!(is_ple_sidecar_name(name), "{name}");
        }
        for name in [
            "m.mq4-xts",
            "qwen3.8-flash-next-base.mq4-xts",
            "qwen3.8-flash-next.mq4",
            "m.q8",
            "ple-q8.mq4",
        ] {
            assert!(!is_ple_sidecar_name(name), "{name}");
        }
    }

    #[test]
    fn sidecar_paths_name_rows_and_their_hot_packs() {
        let paths = ple_sidecar_paths(Path::new("/m/x-base.mq4-xts"));
        let names: Vec<_> = paths
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "/m/x-base.mq4-xts.ple-q4",
                "/m/x-base.mq4-xts.ple-q4.plehotrows",
                "/m/x-base.mq4-xts.ple-q8",
                "/m/x-base.mq4-xts.ple-q8.plehotrows",
                "/m/x-base.mq4-xts.plehot",
            ]
        );
    }

    #[test]
    fn progress_is_throttled_per_phase_and_reports_completion() {
        let t0 = Instant::now();
        let mut printer = ProgressPrinter::new(t0);
        let at = |secs: f64| t0 + Duration::from_secs_f64(secs);
        let progress = |phase, done, total| PleProgress {
            phase,
            done_bytes: done,
            total_bytes: total,
        };
        let first = printer
            .update(&progress("read", 0, 10_000_000_000), at(0.0))
            .unwrap();
        assert!(first.starts_with("ple: read:"), "{first}");
        assert!(printer
            .update(&progress("read", 100, 10_000_000_000), at(0.5))
            .is_none());
        assert!(printer
            .update(&progress("read", 200, 10_000_000_000), at(1.9))
            .is_none());
        let line = printer
            .update(&progress("read", 4_000_000_000, 10_000_000_000), at(2.0))
            .unwrap();
        assert!(line.contains("40.0%"), "{line}");
        assert!(line.contains("4.00/10.00 GB"), "{line}");
        assert!(line.contains("2000 MB/s"), "{line}");
        // Completion prints once even inside the interval.
        assert!(printer
            .update(&progress("read", 10_000_000_000, 10_000_000_000), at(2.1))
            .is_some());
        assert!(printer
            .update(&progress("read", 10_000_000_000, 10_000_000_000), at(2.2))
            .is_none());
        // A new phase prints immediately and restarts the rate.
        let verify = printer
            .update(&progress("verify", 0, 5_000_000_000), at(2.3))
            .unwrap();
        assert!(verify.starts_with("ple: verify:"), "{verify}");
        // Unknown total: no percentage.
        let mut printer = ProgressPrinter::new(t0);
        let line = printer
            .update(&progress("scan", 1_000_000_000, 0), at(1.0))
            .unwrap();
        assert!(!line.contains('%'), "{line}");
    }

    #[test]
    fn durations_and_plan_lines_are_human_readable() {
        assert_eq!(format_duration(Duration::from_secs_f64(12.34)), "12.3s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m05s");
        assert_eq!(format_duration(Duration::from_secs(3723)), "1h02m03s");
        let line = plan_line("ple-q8", "m.ple-q8", Some(54_400_000_000), "download");
        assert!(
            line.contains("54.40 GB") && line.contains("m.ple-q8") && line.ends_with("download")
        );
        assert!(plan_line("base", "m", None, "present").contains("size unknown"));
    }

    #[test]
    fn summaries_print_every_digest_the_registry_needs() {
        let derive = DeriveReport {
            output: PathBuf::from("/m/x.ple-q4"),
            source: PathBuf::from("/m/x.ple-q8"),
            bytes: 28_800_000_000,
            payload_sha256: "p".repeat(64),
            source_rows_sha256: "s".repeat(64),
            elapsed: Duration::from_secs(90),
            threads: 16,
            order: "rank",
        };
        let text = derive_summary(&derive).join("\n");
        for needle in [
            "/m/x.ple-q4",
            "/m/x.ple-q8",
            "28800000000 B",
            "28.80 GB",
            "payload_sha256",
            "q8_rows_sha256",
            "1m30s",
            "16 threads",
            "order           rank",
        ] {
            assert!(text.contains(needle), "{needle}: {text}");
        }
        let split = SplitReport {
            base: PathBuf::from("/m/x-base.mq4"),
            base_bytes: 70_900_000_000,
            base_index_sha256: "b".repeat(64),
            ple_q8: PathBuf::from("/m/x-base.mq4.ple-q8"),
            ple_q8_bytes: 54_400_000_000,
            q8_rows_sha256: "q".repeat(64),
            elapsed: Duration::from_secs(3700),
        };
        let text = split_summary(&split).join("\n");
        for needle in [
            "/m/x-base.mq4",
            "70.90 GB",
            "base_index_sha256",
            "x-base.mq4.ple-q8",
            "54.40 GB",
            "q8_rows_sha256",
            "1h01m40s",
        ] {
            assert!(text.contains(needle), "{needle}: {text}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn split_removal_covers_hardlinked_spellings_and_reports_hot_files() {
        let dir = std::env::temp_dir().join(format!("hipfire-ple-twins-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let canonical = dir.join(FLASH_NEXT_XTS_CANONICAL_FILE);
        let legacy = dir.join(FLASH_NEXT_XTS_LEGACY_FILE);
        fs::write(&canonical, b"monolithic").unwrap();
        fs::hard_link(&canonical, &legacy).unwrap();
        assert_eq!(monolithic_twins(&canonical), [legacy.clone()]);
        assert_eq!(monolithic_twins(&legacy), [canonical.clone()]);
        // A different file under the other spelling is not a twin.
        fs::remove_file(&legacy).unwrap();
        fs::write(&legacy, b"different").unwrap();
        assert!(monolithic_twins(&canonical).is_empty());
        // Any other model name has no twins.
        let other = dir.join("other.mq4");
        fs::write(&other, b"x").unwrap();
        assert!(monolithic_twins(&other).is_empty());
        assert!(plehot_siblings(&canonical).is_empty());
        let rank = dir.join(format!("{FLASH_NEXT_XTS_CANONICAL_FILE}.plehot"));
        let pack = dir.join(format!("{FLASH_NEXT_XTS_CANONICAL_FILE}.plehotrows"));
        fs::write(&rank, b"r").unwrap();
        fs::write(&pack, b"p").unwrap();
        assert_eq!(plehot_siblings(&canonical), [rank, pack]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn clap_parses_ple_subcommands() {
        let parse = |args: &[&str]| {
            let mut argv = vec!["hipfire"];
            argv.extend_from_slice(args);
            Cli::try_parse_from(argv)
        };
        let Commands::Ple(args) = parse(&["ple", "derive", "qwen3.8:flash-next"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(derive.format, DeriveFormat::Q4);
        assert_eq!(derive.model, "qwen3.8:flash-next");
        assert!(derive.rank.is_none());

        let Commands::Ple(args) = parse(&["ple", "derive", "--rank", "/m/x.plehot", "m"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(derive.rank.as_deref(), Some(Path::new("/m/x.plehot")));

        let Commands::Ple(args) = parse(&["ple", "derive", "--format", "q8", "/m/x.mq4"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(
            derive.format,
            DeriveFormat::Q8,
            "rejected at run time with an explanation"
        );

        let Commands::Ple(args) = parse(&[
            "ple",
            "split",
            "--out",
            "/o/x-base.mq4",
            "--keep-original",
            "m",
        ])
        .unwrap()
        .command
        .unwrap() else {
            panic!("expected ple");
        };
        let PleAction::Split(split) = args.action else {
            panic!("expected split");
        };
        assert_eq!(split.out.as_deref(), Some(Path::new("/o/x-base.mq4")));
        assert!(split.keep_original);
        assert_eq!(split.model, "m");

        let Commands::Ple(args) = parse(&["ple", "split", "m"]).unwrap().command.unwrap() else {
            panic!("expected ple");
        };
        let PleAction::Split(split) = args.action else {
            panic!("expected split");
        };
        assert!(split.out.is_none() && !split.keep_original);

        assert!(parse(&["ple"]).is_err());
        assert!(parse(&["ple", "derive"]).is_err());
        assert!(parse(&["ple", "derive", "--format", "q5", "m"]).is_err());
    }

    #[test]
    fn clap_parses_pull_ple_selection() {
        let selection = |args: &[&str]| {
            let mut argv = vec!["hipfire", "pull", "qwen3.8:flash-next"];
            argv.extend_from_slice(args);
            match Cli::try_parse_from(argv).unwrap().command.unwrap() {
                Commands::Pull(pull) => pull.ple,
                other => panic!("expected pull, got {other:?}"),
            }
        };
        assert_eq!(selection(&[]), PleSelection::Auto);
        assert_eq!(selection(&["--ple", "auto"]), PleSelection::Auto);
        assert_eq!(selection(&["--ple", "q4"]), PleSelection::Q4);
        assert_eq!(selection(&["--ple", "q8"]), PleSelection::Q8);
        assert_eq!(selection(&["--ple", "all"]), PleSelection::All);
        assert_eq!(selection(&["--ple", "none"]), PleSelection::Skip);
        assert!(Cli::try_parse_from(["hipfire", "pull", "m", "--ple", "q5"]).is_err());
    }
}
