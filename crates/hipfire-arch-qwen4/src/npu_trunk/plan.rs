// SPDX-License-Identifier: Apache-2.0
//! Pure deterministic crossover model. No clock or hardware input.
use super::{Calibration, CalKey, Family, FamilyCalibration, FamilyMask};
use super::calibration::{OutFamily, FAMILY_NAMES, OUT_FAMILY_NAMES};
use sha2::{Digest, Sha256};
use std::ops::Range;

pub const CHUNK: usize = 8192;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Gpu { chunk: Range<usize> },
    Pipe { a: Range<usize>, b: Range<usize> },
}
#[derive(Clone, Debug, PartialEq)]
pub struct Plan { pub steps: Vec<Step>, pub t_gpu_ms: f64, pub t_pipe_ms: f64 }
#[derive(Clone, Copy, Debug)]
pub struct GpuChunk { pub total: f64, pub gdn_rest: f64, pub qsa_rest: f64, pub trunk: [f64; 4] }
#[derive(Clone, Copy, Debug)]
pub struct BankSegments { pub gpu: [f64; 2], pub npu: [f64; 2], pub npu_out: [f64; 2], pub bytes: [f64; 2], pub chunk: GpuChunk }
#[derive(Clone, Copy, Debug)]
pub struct PairTime { pub t: f64, pub stall: f64, pub extra: f64, pub fill: f64, pub gpu_only: f64 }

pub fn gpu_chunk(cal: &Calibration, rows: usize, prefix: usize) -> GpuChunk {
    let g = &cal.gpu;
    let f = rows as f64 / CHUNK as f64;
    let pre = prefix as f64 / CHUNK as f64;
    let trunk = FAMILY_NAMES.map(|k| g.trunk_in_ms_per_8192[k] * f);
    let gdn_rest = (g.uniform_per_layer_ms + g.gdn_core_per_layer_ms) * f;
    let qsa_rest = (g.uniform_per_layer_ms + g.qsa_per_layer_ms + g.qsa_growth_per_layer_ms_per_8192_prefix * pre) * f;
    GpuChunk { total: 36.0 * gdn_rest + 12.0 * qsa_rest + trunk.iter().sum::<f64>() + g.fixed_per_chunk_ms, gdn_rest, qsa_rest, trunk }
}
fn npu_ms_with(rows: usize, cols: usize, k: usize, panel: usize, tops: f64) -> f64 {
    let padded = cols.div_ceil(panel) * panel;
    2.0 * rows as f64 * padded as f64 * k as f64 / (tops * 1e12) * 1e3
}
pub fn npu_ms(cal: &Calibration, rows: usize, cols: usize) -> f64 {
    npu_ms_with(rows, cols, 2560, cal.npu.panel_cols, cal.npu.tops_corun)
}
/// Out family of layer kind `k` (0 = `gdn.out`, 1 = `qsa.o`): its calibration and out-projection block.
fn out_family(cal: &Calibration, k: usize) -> (&FamilyCalibration, &OutFamily) {
    let family = &cal.families[OUT_FAMILY_NAMES[k]];
    (family, family.out.as_ref().expect("validated: out families carry an out block"))
}
/// Co-run NPU compute time of an out family: its simulator alone rate scaled by the in-proj co-run factor.
fn out_ms(cal: &Calibration, k: usize, rows: usize) -> f64 {
    let (family, o) = out_family(cal, k);
    let tops = o.tops_alone * cal.npu.tops_corun / cal.npu.tops_alone.expect("validated: out families need npu.tops_alone");
    npu_ms_with(rows, family.cols, o.k, o.panel_cols, tops)
}
fn selected(families: FamilyMask) -> [bool; 4] {
    [families.contains(Family::GdnQkv),
     [Family::GdnZ, Family::GdnA, Family::GdnB].iter().all(|f| families.contains(*f)),
     families.contains(Family::QsaQ),
     [Family::QsaK, Family::QsaV, Family::QsaIdx].iter().all(|f| families.contains(*f))]
}
fn supported(families: FamilyMask) -> bool {
    for group in [[Family::GdnZ, Family::GdnA, Family::GdnB], [Family::QsaK, Family::QsaV, Family::QsaIdx]] {
        let count = group.iter().filter(|f| families.contains(**f)).count();
        if count != 0 && count != 3 { return false; }
    }
    !families.is_empty()
}
/// What the planner moves to the NPU: the grouped in-proj families (qkv, zab, q, kvidx) and the out families
/// (`gdn.out`, `qsa.o`). `supported` is false for an empty or partial-group selection, which stays GPU-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved { pub inproj: [bool; 4], pub out: [bool; 2], pub supported: bool }
impl Moved {
    pub fn new(inproj: [bool; 4], out: [bool; 2]) -> Self {
        Self { inproj, out, supported: inproj.iter().chain(&out).any(|moved| *moved) }
    }
}
impl From<FamilyMask> for Moved {
    fn from(families: FamilyMask) -> Self {
        Self { supported: supported(families), ..Self::new(selected(families), [false; 2]) }
    }
}
pub fn bank_segments(cal: &Calibration, families: FamilyMask, rows: usize, prefix: usize) -> BankSegments {
    bank_segments_moved(cal, Moved::from(families), rows, prefix)
}
/// Per-layer-kind GPU segment, NPU in-proj / out chains and NPU DRAM bytes of one bank visit. A moved out family
/// removes its GPU o-proj GEMM from the GPU segment and adds the K=6144 pack.
pub fn bank_segments_moved(cal: &Calibration, moved: Moved, rows: usize, prefix: usize) -> BankSegments {
    let chunk = gpu_chunk(cal, rows, prefix);
    let f = rows as f64 / CHUNK as f64;
    let mut cols = [0; 2];
    let mut kept = [0.0; 2];
    let mut consume = [0.0; 2];
    for (i, name) in FAMILY_NAMES.iter().enumerate() {
        let kind = usize::from(cal.families[*name].layer == "qsa");
        if moved.inproj[i] {
            let family = &cal.families[*name];
            cols[kind] += family.cols;
            if cal.glue.consume_mode == "retile" {
                consume[kind] += cal.glue.consume_ms_per_col_8192[&family.out_dtype] * family.cols as f64;
            }
        } else { kept[kind] += chunk.trunk[i]; }
    }
    let mut gpu = [chunk.gdn_rest + kept[0] / 36.0, chunk.qsa_rest + kept[1] / 12.0];
    let mut npu = [0.0; 2];
    let mut npu_out = [0.0; 2];
    let mut bytes = [0.0; 2];
    for k in 0..2 {
        if cols[k] != 0 {
            gpu[k] += cal.glue.pack_ms_8192 * f;
            npu[k] = npu_ms(cal, rows, cols[k]) + cal.glue.sync_ms;
        }
        gpu[k] += consume[k] * f;
        if cols[k] != 0 || moved.out[k] { gpu[k] += cal.glue.visit_ms_8192 * f; }
        let padded = cols[k].div_ceil(cal.npu.panel_cols) * cal.npu.panel_cols;
        bytes[k] = 2.0 * rows as f64 * padded as f64 * 2560.0 / 1e3 * cal.npu.bytes_per_kop;
        if moved.out[k] {
            let (family, o) = out_family(cal, k);
            gpu[k] += (cal.glue.pack_ms_8192 * o.k as f64 / 2560.0 - o.gpu_ms_per_8192) * f;
            npu_out[k] = out_ms(cal, k, rows) + cal.glue.sync_ms;
            bytes[k] += 2.0 * rows as f64 * family.cols as f64 * o.k as f64 / 1e3 * o.bytes_per_kop;
        }
    }
    BankSegments { gpu, npu, npu_out, bytes, chunk }
}
/// Extra GPU ms while NPU chains `npu_this` (moving `nbytes` of DRAM traffic) overlap the other bank's GPU segment:
/// the legacy burst term plus the per-byte contention (`ms_per_gb`).
pub fn contention_extra(cal: &Calibration, gpu_other: f64, npu_this: f64, nbytes: f64) -> f64 {
    (cal.contention.alpha_overlap - 1.0) * gpu_other.min(npu_this) + cal.contention.ms_per_gb * nbytes / 1e9
}
/// Per QSA layer stall of the order `C_A C_B P_A P_B` (a moved `qsa.o`): I(l+1,A) overlaps P_B(l), I(l+1,B)
/// overlaps C_A(l+1), O_A overlaps C_B, O_B overlaps P_A; each chain stalls by `max(0, chain - window)`. Windows are
/// the measured 16K-pair Core/post phases scaled by rows (Core is linear in the prefix); the post window loses the
/// GPU o-proj GEMM and gains the K=6144 pack; the in-proj chain is the calibrated chain scaled by its columns.
fn qsa_order_stall(cal: &Calibration, moved: Moved, (r1, p1): (usize, usize), (r2, p2): (usize, usize), o1: f64, o2: f64) -> f64 {
    let q = cal.qsa_order.as_ref().expect("validated: qsa.o requires qsa_order");
    let o = &out_family(cal, 1).1;
    let (f1, f2) = (r1 as f64 / CHUNK as f64, r2 as f64 / CHUNK as f64);
    let core = |p: usize| q.core_a_ms + (q.core_b_ms - q.core_a_ms) * p as f64 / CHUNK as f64;
    let post = q.post_ms - o.gpu_ms_per_8192 + cal.glue.pack_ms_8192 * o.k as f64 / 2560.0;
    let qsa_cols = |all: bool| -> usize { [2usize, 3].iter().filter(|i| all || moved.inproj[**i]).map(|i| cal.families[FAMILY_NAMES[*i]].cols).sum() };
    let reference = npu_ms(cal, CHUNK, qsa_cols(true));
    let cols = qsa_cols(false);
    let chain = |r: usize| if cols == 0 { 0.0 } else { q.inproj_chain_ms * npu_ms(cal, r, cols) / reference };
    let (c_a, c_b) = (core(p1) * f1, core(p2) * f2);
    let (p_a, p_b) = (post * f1, post * f2);
    (chain(r1) - p_b).max(0.0) + (chain(r2) - c_a).max(0.0) + (o1 - c_b).max(0.0) + (o2 - p_a).max(0.0)
}
/// Steady-state stall per layer. A moved `qsa.o` forces the order `C_A C_B P_A P_B` (`qsa_order_stall`). Everywhere
/// else the NPU chains of a bank (in-proj, plus the out chain when moved) stall the GPU by what exceeds the other
/// bank's GPU window: zero at the measured balanced pairs (stall ~1 ms/pair, fill only), real for an unbalanced pair.
fn layer_stall(cal: &Calibration, moved: Moved, k: usize, a: (usize, usize), b: (usize, usize), sa: &BankSegments, sb: &BankSegments) -> f64 {
    if k == 1 && moved.out[1] {
        return qsa_order_stall(cal, moved, a, b, sa.npu_out[1], sb.npu_out[1]);
    }
    (sb.npu[k] + sb.npu_out[k] - sa.gpu[k]).max(0.0) + (sa.npu[k] + sa.npu_out[k] - sb.gpu[k]).max(0.0)
}
pub fn pair_time(cal: &Calibration, families: FamilyMask, r1: usize, p1: usize, r2: usize, p2: usize) -> PairTime {
    pair_time_moved(cal, Moved::from(families), r1, p1, r2, p2)
}
pub fn pair_time_moved(cal: &Calibration, moved: Moved, r1: usize, p1: usize, r2: usize, p2: usize) -> PairTime {
    let a = bank_segments_moved(cal, moved, r1, p1);
    let b = bank_segments_moved(cal, moved, r2, p2);
    let mut pr = PairTime { t: 0.0, stall: 0.0, extra: 0.0, fill: if a.npu[0] != 0.0 { a.npu[0] } else { a.npu[1] }, gpu_only: a.chunk.total + b.chunk.total };
    for (k, layers) in [36.0, 12.0].into_iter().enumerate() {
        let s = layer_stall(cal, moved, k, (r1, p1), (r2, p2), &a, &b);
        let e = contention_extra(cal, a.gpu[k], b.npu[k] + b.npu_out[k], b.bytes[k])
            + contention_extra(cal, b.gpu[k], a.npu[k] + a.npu_out[k], a.bytes[k]);
        pr.t += layers * (a.gpu[k] + b.gpu[k] + s + e);
        pr.stall += layers * s;
        pr.extra += layers * e;
    }
    pr.t += pr.fill + cal.gpu.fixed_per_chunk_ms * 2.0;
    pr
}
fn beats(cal: &Calibration, pipe: f64, gpu: f64) -> bool { pipe < (1.0 - cal.margin) * gpu }
fn half(rows: usize) -> usize { (rows / 2).div_ceil(64) * 64 }
#[derive(Clone, Copy)]
struct Choice { consumed: usize, split: Option<usize> }

/// GPU wins exact ties; natural earlier pairs win ties against rebalance/re-bank.
/// The grouped calibration supports whole zab/kvidx groups; partial masks stay GPU-only.
pub fn plan(cal: &Calibration, families: FamilyMask, rows: usize) -> Plan {
    plan_moved(cal, Moved::from(families), rows)
}
pub fn plan_moved(cal: &Calibration, moved: Moved, rows: usize) -> Plan {
    let len = rows.div_ceil(CHUNK);
    let g: Vec<_> = (0..len).map(|i| gpu_chunk(cal, (rows - i * CHUNK).min(CHUNK), i * CHUNK).total).collect();
    let t_gpu_ms = g.iter().sum();
    if !moved.supported {
        return Plan { steps: (0..len).map(|i| Step::Gpu { chunk: i * CHUNK..((i + 1) * CHUNK).min(rows) }).collect(), t_gpu_ms, t_pipe_ms: t_gpu_ms };
    }
    let mut best = vec![0.0; len + 1];
    let mut choice = vec![Choice { consumed: 1, split: None }; len + 1];
    for i in 1..=len {
        best[i] = best[i - 1] + g[i - 1];
        if i >= 2 {
            let start = (i - 2) * CHUNK;
            let last = (rows - (i - 1) * CHUNK).min(CHUNK);
            let pr = pair_time_moved(cal, moved, CHUNK, start, last, start + CHUNK);
            if beats(cal, pr.t, pr.gpu_only) && best[i - 2] + pr.t < best[i] {
                best[i] = best[i - 2] + pr.t;
                choice[i] = Choice { consumed: 2, split: Some(CHUNK) };
            }
        }
    }
    if len >= 2 && rows % CHUNK != 0 {
        let start = (len - 2) * CHUNK;
        let total = rows - start;
        let h1 = half(total);
        let h2 = total - h1;
        let pr = pair_time_moved(cal, moved, h1, start, h2, start + h1);
        if h1.min(h2) >= cal.gpu.min_bank_rows && beats(cal, pr.t, g[len - 2] + g[len - 1]) && best[len - 2] + pr.t < best[len] {
            best[len] = best[len - 2] + pr.t;
            choice[len] = Choice { consumed: 2, split: Some(h1) };
        }
    }
    if len == 1 && rows >= 2 * cal.gpu.min_bank_rows {
        let h1 = half(rows);
        let t = pair_time_moved(cal, moved, h1, 0, rows - h1, h1).t + cal.gpu.boundary_tax_ms;
        if beats(cal, t, g[0]) && t < best[1] {
            best[1] = t;
            choice[1] = Choice { consumed: 1, split: Some(h1) };
        }
    }
    let nb = 2 * rows.div_ceil(2 * CHUNK);
    if len >= 2 && nb > 0 && rows / nb >= cal.gpu.min_bank_rows {
        let size = (rows / nb).div_ceil(64) * 64;
        let last = rows as isize - (size * (nb - 1)) as isize;
        // Python can construct a nonpositive final bank for very small banks;
        // no executable row range exists in that case.
        if last > 0 {
            let mut t = 0.0;
            let mut eligible = true;
            for k in (0..nb).step_by(2) {
                let r2 = if k + 2 == nb { last as usize } else { size };
                let pr = pair_time_moved(cal, moved, size, k * size, r2, (k + 1) * size);
                t += pr.t;
                if cal.margin != 0.0 && !beats(cal, pr.t, pr.gpu_only) { eligible = false; }
            }
            t += cal.gpu.boundary_tax_ms * nb.saturating_sub(len) as f64;
            if eligible && beats(cal, t, t_gpu_ms) && t < best[len] {
                return Plan { steps: (0..nb).step_by(2).map(|k| Step::Pipe {
                    a: k * size..(k + 1) * size,
                    b: (k + 1) * size..if k + 2 == nb { rows } else { (k + 2) * size },
                }).collect(), t_gpu_ms, t_pipe_ms: t };
            }
        }
    }
    let mut steps = Vec::with_capacity(len);
    let mut i = len;
    while i > 0 {
        let c = choice[i];
        let start = (i - c.consumed) * CHUNK;
        let end = (i * CHUNK).min(rows);
        steps.push(match c.split {
            Some(split) => Step::Pipe { a: start..start + split, b: start + split..end },
            None => Step::Gpu { chunk: start..end },
        });
        i -= c.consumed;
    }
    steps.reverse();
    Plan { steps, t_gpu_ms, t_pipe_ms: best[len] }
}
/// First 64-row grid point in 1024..=49152 whose suffix always beats GPU-only.
pub fn crossover(cal: &Calibration, families: FamilyMask) -> Option<usize> {
    let mut first = None;
    for rows in (1024..=6 * CHUNK).step_by(64) {
        let p = plan(cal, families, rows);
        if p.t_pipe_ms < p.t_gpu_ms { if first.is_none() { first = Some(rows); } }
        else { first = None; }
    }
    first
}

pub const SELECTION_ROWS: [usize; 3] = [16384, 32768, 65536];
/// One row of the load-time eligibility table, in ascending subset-bit order.
#[derive(Clone, Debug)]
pub struct SelectionRow {
    pub families: FamilyMask,
    pub t_pipe_ms: [f64; 3],
    pub t_gpu_ms: [f64; 3],
    pub relative: [f64; 3],
    pub mean_relative: f64,
    pub beats_margin_16384: bool,
    /// Every family of the subset is `selectable`; only these can win.
    pub selectable: bool,
}
#[derive(Clone, Debug)]
pub struct SelectionReport {
    pub subsets: Vec<SelectionRow>,
    /// Minimum-mean candidate before the 16384-row margin gate.
    pub best_families: FamilyMask,
}
impl SelectionRow {
    pub fn family_names(&self) -> impl Iterator<Item = &'static str> {
        FAMILY_NAMES.into_iter().zip(selected(self.families)).filter_map(|(name, yes)| yes.then_some(name))
    }
}
fn subset_mask(bits: u8) -> FamilyMask {
    let groups = [FamilyMask::from(Family::GdnQkv), FamilyMask::from_bits(0x0e),
                  FamilyMask::from(Family::QsaQ), FamilyMask::from_bits(0xe0)];
    groups.into_iter().enumerate().fold(FamilyMask::NONE, |mask, (i, group)| {
        if bits & (1 << i) != 0 { mask | group } else { mask }
    })
}
fn compare_selection(a: &SelectionRow, b: &SelectionRow) -> std::cmp::Ordering {
    a.mean_relative.total_cmp(&b.mean_relative)
        .then_with(|| a.family_names().count().cmp(&b.family_names().count()))
        .then_with(|| {
            let names = ["kvidx", "q", "qkv", "zab"];
            names.into_iter().filter(|name| a.family_names().any(|n| n == *name))
                .cmp(names.into_iter().filter(|name| b.family_names().any(|n| n == *name)))
        })
}
/// Select the minimum mean T_pipe/T_gpu across the three fixed prompt sizes among the subsets whose every family is
/// `selectable` (the out families `gdn.out` / `qsa.o` are not in the in-proj mask and stay at zero selection).
/// Gate that winner (not a runner-up) by the strict 16384-row margin.
pub fn select_families(cal: &Calibration) -> (FamilyMask, SelectionReport) {
    let mut subsets = Vec::with_capacity(15);
    for bits in 1..16 {
        let families = subset_mask(bits);
        let plans = SELECTION_ROWS.map(|rows| plan(cal, families, rows));
        let t_pipe_ms = std::array::from_fn(|i| plans[i].t_pipe_ms);
        let t_gpu_ms = std::array::from_fn(|i| plans[i].t_gpu_ms);
        // A zero-cost GPU baseline has no possible strict improvement.
        let relative = std::array::from_fn(|i| if t_gpu_ms[i] == 0.0 { 1.0 } else { t_pipe_ms[i] / t_gpu_ms[i] });
        let selectable = FAMILY_NAMES.iter().enumerate().all(|(i, name)| bits & (1 << i) == 0 || cal.families[*name].selectable);
        subsets.push(SelectionRow {
            families, t_pipe_ms, t_gpu_ms, relative,
            mean_relative: relative.iter().sum::<f64>() / 3.0,
            beats_margin_16384: beats(cal, t_pipe_ms[0], t_gpu_ms[0]),
            selectable,
        });
    }
    let Some(best) = subsets.iter().filter(|row| row.selectable).min_by(|a, b| compare_selection(a, b)) else {
        return (FamilyMask::NONE, SelectionReport { subsets, best_families: FamilyMask::NONE });
    };
    let families = if best.beats_margin_16384 { best.families } else { FamilyMask::NONE };
    let best_families = best.families;
    (families, SelectionReport { subsets, best_families })
}

/// Admit any nonempty calibrated family subset.
pub fn auto_admission(cal: &Calibration, current: &CalKey) -> Result<FamilyMask, String> {
    cal.key_matches(current)?;
    let (families, _) = select_families(cal);
    if families.is_empty() { return Err("calibration selects no families".into()); }
    Ok(families)
}

/// Whether `a` + `b` is a pair the executor can run: the natural 8192 + B pair
/// (B in `[minimum, 8192]`) or the planner's last-two-chunk rebalanced pair, two
/// banks of about equal rows (64-row aligned split, each in `[minimum, 8192]`)
/// that end the request. Split and re-bank geometries match neither.
fn executable_pair(a: &Range<usize>, b: &Range<usize>, rows: usize, minimum: usize) -> bool {
    if a.start % CHUNK != 0 || b.start != a.end { return false; }
    let natural = a.len() == CHUNK && (minimum..=CHUNK).contains(&b.len()) && b.end == (a.start + 2 * CHUNK).min(rows);
    let rebalanced = a.len() != CHUNK && a.len() % 64 == 0
        && a.len().min(b.len()) >= minimum && a.len().max(b.len()) <= CHUNK
        && b.end == rows && b.end - a.start > CHUNK;
    natural || rebalanced
}

/// Project S5 onto the executor: natural pairs at the natural chunk boundaries
/// and the rebalanced last pair (`pipe-rebalanced`). Split/rebanked steps stay
/// GPU-only; never substitute force-pairs.
pub fn executable_plan(cal: &Calibration, planned: &Plan, rows: usize, n_star: Option<usize>) -> Plan {
    let minimum = 512.max(cal.gpu.min_bank_rows);
    let armed = n_star.is_some_and(|n| rows >= n);
    let mut steps = Vec::with_capacity(rows.div_ceil(CHUNK));
    let mut at = 0;
    let mut t_pipe_ms = 0.0;
    while at < rows {
        let pair = armed.then(|| planned.steps.iter().find_map(|step| match step {
            Step::Pipe { a, b } if a.start == at && executable_pair(a, b, rows, minimum) => Some((a.clone(), b.clone())),
            _ => None,
        })).flatten();
        if let Some((a, b)) = pair {
            t_pipe_ms += pair_time(cal, FamilyMask::ALL, a.len(), a.start, b.len(), b.start).t;
            at = b.end;
            steps.push(Step::Pipe { a, b });
        } else {
            let end = (at + CHUNK).min(rows);
            t_pipe_ms += gpu_chunk(cal, end - at, at).total;
            steps.push(Step::Gpu { chunk: at..end });
            at = end;
        }
    }
    Plan { steps, t_gpu_ms: planned.t_gpu_ms, t_pipe_ms }
}

/// Rows of the planned pair starting at `offset`: bank A, then bank B.
pub fn planned_pair(planned: &Plan, offset: usize) -> Option<(usize, usize)> {
    planned.steps.iter().find_map(|step| match step {
        Step::Pipe { a, b } if a.start == offset => Some((a.len(), b.len())),
        _ => None,
    })
}

pub fn planned_pair_rows(planned: &Plan, offset: usize) -> Option<usize> {
    planned_pair(planned, offset).map(|(a, b)| a + b)
}
/// Load-time log context, all caller-supplied deterministic state.
pub struct LoadLogKey<'a> { pub current: &'a CalKey, pub mode: &'a str, pub families: FamilyMask }
pub fn load_log_line(cal: &Calibration, key: &LoadLogKey<'_>, n_star: Option<usize>, t16384: &Plan, t65536: &Plan) -> String {
    if let Err(reason) = cal.key_matches(key.current) { return format!("npu-trunk: off ({reason})"); }
    let bytes = serde_json::to_vec(cal).expect("validated calibration JSON");
    let hash = Sha256::digest(&bytes);
    let names: Vec<_> = FAMILY_NAMES.into_iter().zip(selected(key.families)).filter_map(|(name, yes)| yes.then_some(name)).collect();
    format!("npu-trunk: mode={} contract={} families={} cal_key={} cal_sha256={:x} n*={} T(16384)={:.6}/{:.6} T(65536)={:.6}/{:.6}",
        key.mode, cal.npu.epilogue, names.join(","), serde_json::to_string(&cal.key).expect("key JSON"), hash,
        n_star.map_or_else(|| "none".into(), |n| n.to_string()), t16384.t_pipe_ms, t16384.t_gpu_ms, t65536.t_pipe_ms, t65536.t_gpu_ms)
}
pub fn request_log_line(rows: usize, plan: &Plan) -> String {
    let steps: Vec<_> = plan.steps.iter().map(|s| match s {
        Step::Gpu { chunk } => format!("gpu {}..{}", chunk.start, chunk.end),
        Step::Pipe { a, b } => format!("pipe {}..{}+{}..{}", a.start, a.end, b.start, b.end),
    }).collect();
    let percent = if plan.t_gpu_ms == 0.0 { 0.0 } else { 100.0 * (plan.t_pipe_ms - plan.t_gpu_ms) / plan.t_gpu_ms };
    format!("npu-trunk: rows={rows} plan=[{}] predicted={percent:+.1}%", steps.join(" | "))
}
