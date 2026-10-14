// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only raw-ABI oracle, screen and timing of the live F32 QSA
//! selector pair (`indexed_attention_select_scores_rows16_f32` +
//! `indexed_attention_select_from_scores`) on real `qsa-source-v1` snapshots.
//!
//! `qsa_select_pm_check SNAPSHOT_DIR... --out OUT.jsonl [--time N]
//!     [--score hipcc|pm] [--select hipcc|pm] [--pm-image ABS_FILE]
//!     [--cpu-rows STRIDE] [--no-screen] [--tied] [--shape-blocks N]
//!     [--tied-bf16-hipcc] [--g3-bf16]`
//!
//! Every snapshot (pre-prologue dumps skipped) is replayed with the
//! production grouping (64 MiB score scratch, rows16, one select WG per row):
//! - baseline hipcc/hipcc: ordered `selected.i32` and the final-row mirror
//!   must equal the snapshot bytes (else the capture/ABI/route is wrong);
//! - CPU: every `--cpu-rows`-th row (default 64) plus the last row is
//!   rescored with the per-head serial F32 FMA chain and bit-compared with the
//!   hipcc scores; every row's stable (score desc, block asc) CPU selection
//!   over the hipcc scores is compared with the hipcc selection;
//! - screen: the adaptive radix prefix replay (stop when above + boundary
//!   bucket <= 1024) over every row's hipcc scores;
//! - WMMA diagnostic (CPU emulation of F16/BF16 operand rounding with the
//!   serial F32 chain kept: a lower bound on a WMMA port's differences);
//! - with `--score pm` / `--select pm`, the experimental image's symbols
//!   replace the corresponding hipcc launch: valid score bits (and untouched
//!   poison outside them) and the ordered selection/mirror must be identical;
//! - `--time N`: N repetitions of the whole call, per-kernel event time.
//! `--tied` adds the synthetic all-tied rows (16384, 32768 and 65536 blocks)
//! for the PM selector only, against the CPU stable-order oracle; the
//! pre-fix hipcc selector never runs on them.
//!
//! `--g3 --beta-src ABS_FILE` runs the G3 exact-oracle + HIP-graph gate of the
//! PM pair as the default route (see `mod g3`; build with `--features lab`).
//!
//! `--g3-bf16 [SNAPSHOT_DIR...] --out OUT.jsonl` runs the BF16 production-route
//! gate (see `mod g3_bf16`; build with `--features lab`; no `--beta-src`, no
//! `--pm-image`, no timing/F32-route flags): BF16 `qsa-source-v1` snapshots
//! (optional; synthetic-only when no directory is given) plus synthetic
//! all-tied fixtures run the explicit lab arms (`Hipcc`, `ScoreOnly`, `Both`),
//! the default production route (mirrored and no-mirror), and the default
//! mirrored route captured into a HIP graph, at `shape_blocks =
//! pooled_capacity` and again at 65536; one JSONL record per case and shape,
//! then a summary line; nonzero exit on any failing record.
//!
//! BF16 route (auto: header `identity.selector.pooled_dtype == "BF16"`; the
//! projection stays F32, heads 4, dim 128; `pooled.source` is raw BF16,
//! `pooled_capacity * 128 * 2` bytes). F32 snapshots keep the exact F32 path
//! above; F32 and BF16 snapshots may be mixed in one run. The incumbent is the
//! production fused `indexed_attention_select_bf16_batched` launch (raw ABI,
//! `--shape-blocks N`, default 65536, which must cover the active blocks):
//! global score scratch iff `N*4 + 4096 > 65536` (groups of 256 rows,
//! `256 * stride * 4` bytes, `stride = ceil(N/4)*4`, dynamic LDS 0, per-group
//! block count / position start, mirror on the last group only), else one
//! launch with dynamic LDS `N*4`. Records carry `route:"bf16"`.
//! - baseline hipcc/hipcc (fused): selected/mirror must equal the snapshot
//!   bytes. In global mode the scratch is poisoned before every group, each
//!   group's causal prefix is downloaded into canonical `rows*block_count`
//!   poison-padded score keys and the whole fused scratch outside the causal
//!   extents (incl. unused rows) must still be 0xa5; the existing CPU
//!   rescoring / stable selection / screen / WMMA diagnostics then run on the
//!   BF16 pooled values widened exactly (`bits << 16`) with the F32 query. In
//!   LDS mode no score capture exists: the CPU comparisons and screen are
//!   reported as skipped (never substituted by another kernel);
//! - `--score pm --select pm|hipcc` (`indexed_attention_select_scores_rows16_bf16_pm_gfx1151`,
//!   existing PM select or hipcc from_scores select, 64 MiB rows16 grouping):
//!   selected/mirror must equal the fused bytes, every causal score bit must
//!   equal the fused global capture (`skipped` in LDS mode) and the WHOLE
//!   candidate scratch outside causal extents, including unused allocated
//!   rows after each score group, must remain 0xa5 (poisoned before every
//!   score launch). `--score hipcc --select pm` and any BF16 hipcc score arm
//!   are rejected;
//! - `--time N` (baseline): `fused_ms`, `score_phase_ms` (appended lab kernel
//!   `qsa_bf16_batched_score_only_lab` = the fused pre-selection code through
//!   its first `__syncthreads`, extracted from the live source) and
//!   `select_phase_ms = fused - score` per rep; PM arms report `score_ms` /
//!   `select_ms` as the F32 route does;
//! - `--tied-bf16-hipcc`: the six all-tied fixtures (16384/32768/65536 blocks
//!   x BF16 value 0x0000/0x3f80, query all 0/1; rows 3, compress 4, budget
//!   512, capacity 2051, stride 512, `position_start = blocks*4`) run the
//!   fused launch (and the PM pair iff `--pm-image` supplies the BF16 score
//!   and select symbols) with selected/mirror each inside a 64 KiB 0xa5
//!   guard on both sides; the ordered rows + mirror must equal the CPU stable
//!   oracle and every guard of both buffers must be unchanged.

use hip_bridge::{Function, KernargBlob, Module};
use rayon::prelude::*;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String { format!("{e:?}") }
fn sha256(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }

const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const HIPCC_SCORE: &str = "indexed_attention_select_scores_rows16_f32";
const HIPCC_SELECT: &str = "indexed_attention_select_from_scores";
const PM_SCORE: &str = "indexed_attention_select_scores_rows16_f32_pm_gfx1151";
const PM_SELECT: &str = "indexed_attention_select_from_scores_pm_gfx1151";
const PM_SCORE_BF16: &str = "indexed_attention_select_scores_rows16_bf16_pm_gfx1151";
const SCORE_SCRATCH_BYTES: usize = 64 << 20;
const POISON: u8 = 0xa5;

fn snapshots(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(err)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "pre-prologue") { continue }
            snapshots(&p, out)?;
        } else if p.file_name().is_some_and(|n| n == "snapshot.json") {
            out.push(p);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Impl { Hipcc, Pm }

struct Pm { _module: Module, score: Option<Function>, score_bf16: Option<Function>, select: Option<Function> }

/// One original select call's geometry.
#[derive(Clone, Copy)]
struct Call { rows: usize, stride: usize, block_count: usize, dim: usize, compress: usize, position_start: usize, budget: usize, capacity: usize }

impl Call {
    fn group(&self) -> usize { (SCORE_SCRATCH_BYTES / (self.block_count * 4) / 16 * 16).max(16).min(self.rows) }
    fn row_blocks(&self, row: usize) -> usize { ((self.position_start + row + 1) / self.compress).min(self.block_count) }
}

struct Bufs { query: GpuTensor, pooled: GpuTensor, scores: GpuTensor, selected: GpuTensor, mirror: GpuTensor }

fn score_blob(c: &Call, b: &Bufs, g0: usize, n: usize) -> KernargBlob {
    let mut a = KernargBlob::new();
    a.push_ptr(unsafe { (b.query.buf.as_ptr() as *const f32).add(g0 * c.stride) } as *const _);
    a.push_ptr(b.pooled.buf.as_ptr());
    a.push_ptr(b.scores.buf.as_ptr());
    a.push_i32(n as i32);
    a.push_i32(c.stride as i32);
    a.push_i32(c.block_count as i32);
    a.push_i32(c.dim as i32);
    a.push_i32(c.compress as i32);
    a.push_i32((c.position_start + g0) as i32);
    a.push_i32(c.block_count as i32);
    a.pad_to(16);
    a
}

fn select_blob(c: &Call, b: &Bufs, g0: usize, n: usize, last: bool) -> KernargBlob {
    let mut a = KernargBlob::new();
    a.push_ptr(b.scores.buf.as_ptr());
    a.push_i32(c.block_count as i32);
    a.push_ptr(unsafe { (b.selected.buf.as_ptr() as *const i32).add(g0 * c.capacity) } as *const _);
    a.push_i32(n as i32);
    a.push_i32(c.block_count as i32);
    a.push_i32(c.budget as i32);
    a.push_i32(c.compress as i32);
    a.push_i32((c.position_start + g0) as i32);
    a.push_i32(c.capacity as i32);
    a.push_ptr(if last { b.mirror.buf.as_ptr() } else { std::ptr::null_mut() });
    a.pad_to(16);
    a
}

fn launch(gpu: &mut Gpu, pm: &Option<Pm>, which: Impl, score: bool, grid: [u32; 3], blob: &mut KernargBlob) -> Result<()> {
    match which {
        Impl::Hipcc => gpu
            .launch_kernel_blob(if score { HIPCC_SCORE } else { HIPCC_SELECT }, grid, [256, 1, 1], 0, blob.as_mut_slice())
            .map_err(err),
        Impl::Pm => {
            let pm = pm.as_ref().ok_or("--pm-image required for a pm arm")?;
            let f = if score { pm.score.as_ref() } else { pm.select.as_ref() }.ok_or("pm image lacks the symbol")?;
            unsafe { gpu.hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, blob.as_mut_slice()) }.map_err(err)
        }
    }
}

struct Timing { score_ms: f64, select_ms: f64 }

/// The production pair over all groups. `keep` downloads each group's
/// scratch after its score launch (rows x block_count, raw bytes).
fn run_call(gpu: &mut Gpu, pm: &Option<Pm>, c: &Call, b: &Bufs, arms: (Impl, Impl), mut keep: Option<&mut Vec<u8>>,
            events: Option<&[hip_bridge::Event; 3]>) -> Result<Timing> {
    let group = c.group();
    let (mut score_ms, mut select_ms) = (0.0f64, 0.0f64);
    let mut g0 = 0;
    while g0 < c.rows {
        let n = group.min(c.rows - g0);
        if keep.is_some() {
            gpu.hip.memset(&b.scores.buf, POISON as i32, b.scores.byte_size()).map_err(err)?;
        }
        let mut blob = score_blob(c, b, g0, n);
        if let Some(e) = events { gpu.hip.event_record(&e[0], None).map_err(err)?; }
        launch(gpu, pm, arms.0, true, [c.block_count.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], &mut blob)?;
        if let Some(e) = events { gpu.hip.event_record(&e[1], None).map_err(err)?; }
        if let Some(k) = keep.as_deref_mut() {
            gpu.hip.device_synchronize().map_err(err)?;
            let mut bytes = vec![0u8; n * c.block_count * 4];
            let len = bytes.len();
            gpu.hip.memcpy_dtoh(&mut bytes, &b.scores.buf.byte_view(0, len)).map_err(err)?;
            k.extend_from_slice(&bytes);
        }
        let mut blob = select_blob(c, b, g0, n, g0 + n == c.rows);
        launch(gpu, pm, arms.1, false, [n as u32, 1, 1], &mut blob)?;
        if let Some(e) = events {
            gpu.hip.event_record(&e[2], None).map_err(err)?;
            gpu.hip.event_synchronize(&e[2]).map_err(err)?;
            score_ms += gpu.hip.event_elapsed_ms(&e[0], &e[1]).map_err(err)? as f64;
            select_ms += gpu.hip.event_elapsed_ms(&e[1], &e[2]).map_err(err)? as f64;
        }
        g0 += n;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    Ok(Timing { score_ms, select_ms })
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).map_err(err)?;
    Ok(bytes)
}

fn poison_outputs(gpu: &Gpu, b: &Bufs) -> Result<()> {
    gpu.hip.memset(&b.selected.buf, POISON as i32, b.selected.byte_size()).map_err(err)?;
    gpu.hip.memset(&b.mirror.buf, POISON as i32, b.mirror.byte_size()).map_err(err)
}

// ------------------------------------------------------------ BF16 route

const FUSED_BF16: &str = "indexed_attention_select_bf16_batched";
const LAB_MODULE: &str = "qsa_bf16_lab";
const LAB_BF16: &str = "qsa_bf16_batched_score_only_lab";
/// `QSA_SELECT_GLOBAL_ROWS`, `QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES` and
/// `QSA_SELECT_BATCHED_STATIC_LDS_BYTES` of `rdna-compute/src/tensor_ops.rs`.
const GLOBAL_ROWS: usize = 256;
const LDS_LIMIT_BYTES: usize = 64 * 1024;
const STATIC_LDS_BYTES: usize = 4 * 1024;
const GUARD_BYTES: usize = 64 << 10;
const LAB_SIG: &str = "extern \"C\" __global__ void qsa_bf16_batched_score_only_lab(const float* query, \
    const __hip_bfloat16* pooled, int* selected, int rows, int query_row_stride, int block_count, \
    int index_heads, int index_dim, int budget_blocks, int compress, int position_start, int capacity, \
    int* mirror, float* scores_global, int scores_stride) {";

fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()
}

fn le_u32(bytes: &[u8]) -> Vec<u32> { bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect() }
fn le_i32(bytes: &[u8]) -> Vec<i32> { bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect() }

/// The appended lab kernel: the live source's `qsa_select_batched` body
/// through its first `__syncthreads` (the -1 fill and every block score),
/// behind an explicit BF16 extern signature, so it reproduces the fused
/// pre-selection phase exactly.
fn lab_source() -> Result<String> {
    let marker = "static __device__ __forceinline__ void qsa_select_batched(";
    let start = TENSOR_OPS_SRC.find(marker).ok_or("qsa_select_batched not found in tensor_ops source")?;
    let after = &TENSOR_OPS_SRC[start + marker.len()..];
    let open = after.find(") {").ok_or("qsa_select_batched signature end not found")?;
    let rest = &after[open + 3..];
    let sync = "__syncthreads();";
    let end = rest.find(sync).ok_or("qsa_select_batched first __syncthreads not found")? + sync.len();
    let mut s = String::from(TENSOR_OPS_SRC);
    s.push('\n');
    s.push_str(LAB_SIG);
    s.push_str(&rest[..end]);
    s.push_str("\n}\n");
    Ok(s)
}

/// Production fused-launch geometry for shape bound `shape` plus its score
/// scratch (global mode only).
struct Fused { shape: usize, stride: usize, global: bool, scratch: Option<GpuTensor> }

impl Fused {
    fn new(gpu: &Gpu, shape: usize) -> Result<Fused> {
        if shape == 0 { return Err("shape blocks must be > 0".into()) }
        let global = shape * 4 + STATIC_LDS_BYTES > LDS_LIMIT_BYTES;
        let stride = shape.div_ceil(4) * 4;
        let scratch = if global {
            let bytes = GLOBAL_ROWS * stride * 4;
            Some(gpu.upload_raw(&vec![POISON; bytes], &[bytes]).map_err(err)?)
        } else { None };
        Ok(Fused { shape, stride, global, scratch })
    }
    /// `(first, rows)` launch groups: 256-row groups past the LDS row, else one.
    fn groups(&self, c: &Call) -> Vec<(usize, usize)> {
        if self.global && c.rows > GLOBAL_ROWS {
            (0..c.rows).step_by(GLOBAL_ROWS).map(|first| (first, GLOBAL_ROWS.min(c.rows - first))).collect()
        } else {
            vec![(0, c.rows)]
        }
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        match self.scratch { Some(s) => gpu.free_tensor(s).map_err(err), None => Ok(()) }
    }
}

/// Query/pooled/selected/mirror of one fused launch (selected/mirror may be
/// guarded sub-views).
struct Io<'a> { query: &'a GpuTensor, pooled: &'a GpuTensor, selected: &'a GpuTensor, mirror: &'a GpuTensor }

/// Production raw ABI: ptr(query, pooled, selected), i32(rows, stride,
/// block_count, heads4, dim128, budget, compress, position_start, capacity),
/// ptr(mirror, scores_global), i32(scores_stride), pad16. A group's block
/// count is `(position_start + first + rows) / compress`.
fn fused_blob(c: &Call, io: &Io, f: &Fused, first: usize, rows: usize, last: bool) -> KernargBlob {
    let mut a = KernargBlob::new();
    a.push_ptr(unsafe { (io.query.buf.as_ptr() as *const f32).add(first * c.stride) } as *const _);
    a.push_ptr(io.pooled.buf.as_ptr());
    a.push_ptr(unsafe { (io.selected.buf.as_ptr() as *const u8).add(first * c.capacity * 4) } as *const _);
    a.push_i32(rows as i32);
    a.push_i32(c.stride as i32);
    a.push_i32(((c.position_start + first + rows) / c.compress) as i32);
    a.push_i32(4);
    a.push_i32(c.dim as i32);
    a.push_i32(c.budget as i32);
    a.push_i32(c.compress as i32);
    a.push_i32((c.position_start + first) as i32);
    a.push_i32(c.capacity as i32);
    a.push_ptr(if last { io.mirror.buf.as_ptr() } else { std::ptr::null_mut() });
    a.push_ptr(f.scratch.as_ref().map_or(std::ptr::null_mut(), |s| s.buf.as_ptr()));
    a.push_i32(f.stride as i32);
    a.pad_to(16);
    a
}

/// Copy each row's causal prefix of a poison-initialised score scratch into
/// the canonical `rows * block_count` layout (`canon` stays poison elsewhere)
/// and report whether every scratch byte outside the causal extents, including
/// every allocated row past `group_rows`, is still poison.
fn extract_scores(scratch: &[u8], row_bytes: usize, group_rows: usize, first: usize, c: &Call, canon: &mut [u8]) -> bool {
    let canon_row = c.block_count * 4;
    let mut ok = true;
    for r in 0..scratch.len() / row_bytes {
        let row = &scratch[r * row_bytes..(r + 1) * row_bytes];
        let live = if r < group_rows { c.row_blocks(first + r) * 4 } else { 0 };
        ok &= row[live..].iter().all(|&x| x == POISON);
        if live > 0 {
            let at = (first + r) * canon_row;
            canon[at..at + live].copy_from_slice(&row[..live]);
        }
    }
    ok
}

struct FusedRun { ms: f64, canon: Option<Vec<u8>>, poison_ok: bool }

/// The production fused launch(es) of `kernel` (the incumbent or the lab
/// kernel, identical blob/grouping/dynamic LDS). `capture` (global mode only)
/// poisons the scratch before every group, then extracts the group's scores.
/// `ms` sums the per-group event times.
fn run_fused(gpu: &mut Gpu, c: &Call, io: &Io, f: &Fused, kernel: &str, capture: bool, events: Option<&[hip_bridge::Event; 3]>)
    -> Result<FusedRun> {
    let capture = capture && f.global;
    let mut canon = capture.then(|| vec![POISON; c.rows * c.block_count * 4]);
    let (mut poison_ok, mut ms) = (true, 0.0f64);
    let shared = if f.global { 0 } else { (f.shape * 4) as u32 };
    for (first, rows) in f.groups(c) {
        let last = first + rows == c.rows;
        if capture {
            let s = f.scratch.as_ref().unwrap();
            gpu.hip.memset(&s.buf, POISON as i32, s.byte_size()).map_err(err)?;
        }
        let mut blob = fused_blob(c, io, f, first, rows, last);
        if let Some(e) = events { gpu.hip.event_record(&e[0], None).map_err(err)?; }
        gpu.launch_kernel_blob(kernel, [rows as u32, 1, 1], [256, 1, 1], shared, blob.as_mut_slice()).map_err(err)?;
        if let Some(e) = events {
            gpu.hip.event_record(&e[1], None).map_err(err)?;
            gpu.hip.event_synchronize(&e[1]).map_err(err)?;
            ms += gpu.hip.event_elapsed_ms(&e[0], &e[1]).map_err(err)? as f64;
        }
        if capture {
            gpu.hip.device_synchronize().map_err(err)?;
            let bytes = download(gpu, f.scratch.as_ref().unwrap())?;
            poison_ok &= extract_scores(&bytes, f.stride * 4, rows, first, c, canon.as_mut().unwrap());
        }
    }
    gpu.hip.device_synchronize().map_err(err)?;
    Ok(FusedRun { ms, canon, poison_ok })
}

fn launch_pm_score_bf16(gpu: &mut Gpu, pm: &Option<Pm>, grid: [u32; 3], blob: &mut KernargBlob) -> Result<()> {
    let pm = pm.as_ref().ok_or("--pm-image required for a pm arm")?;
    let f = pm.score_bf16.as_ref().ok_or("pm image lacks the BF16 score symbol")?;
    unsafe { gpu.hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, blob.as_mut_slice()) }.map_err(err)
}

/// The candidate pair (BF16 PM score + PM or hipcc from_scores select) over
/// the production 64 MiB / rows16 grouping. `capture` poisons the scratch
/// before every score launch and, after it, downloads the WHOLE scratch:
/// returns the canonical causal scores and whether everything outside the
/// causal extents (incl. unused allocated rows) stayed poison.
fn run_pm_bf16(gpu: &mut Gpu, pm: &Option<Pm>, c: &Call, b: &Bufs, arms: (Impl, Impl), capture: bool,
               events: Option<&[hip_bridge::Event; 3]>) -> Result<(Timing, Option<(Vec<u8>, bool)>)> {
    let group = c.group();
    let (mut score_ms, mut select_ms) = (0.0f64, 0.0f64);
    let mut canon = capture.then(|| vec![POISON; c.rows * c.block_count * 4]);
    let mut poison_ok = true;
    let mut g0 = 0;
    while g0 < c.rows {
        let n = group.min(c.rows - g0);
        if capture { gpu.hip.memset(&b.scores.buf, POISON as i32, b.scores.byte_size()).map_err(err)?; }
        let mut blob = score_blob(c, b, g0, n);
        if let Some(e) = events { gpu.hip.event_record(&e[0], None).map_err(err)?; }
        launch_pm_score_bf16(gpu, pm, [c.block_count.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], &mut blob)?;
        if let Some(e) = events { gpu.hip.event_record(&e[1], None).map_err(err)?; }
        if capture {
            gpu.hip.device_synchronize().map_err(err)?;
            let bytes = download(gpu, &b.scores)?;
            poison_ok &= extract_scores(&bytes, c.block_count * 4, n, g0, c, canon.as_mut().unwrap());
        }
        let mut blob = select_blob(c, b, g0, n, g0 + n == c.rows);
        launch(gpu, pm, arms.1, false, [n as u32, 1, 1], &mut blob)?;
        if let Some(e) = events {
            gpu.hip.event_record(&e[2], None).map_err(err)?;
            gpu.hip.event_synchronize(&e[2]).map_err(err)?;
            score_ms += gpu.hip.event_elapsed_ms(&e[0], &e[1]).map_err(err)? as f64;
            select_ms += gpu.hip.event_elapsed_ms(&e[1], &e[2]).map_err(err)? as f64;
        }
        g0 += n;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    Ok((Timing { score_ms, select_ms }, canon.map(|v| (v, poison_ok))))
}

/// Full BF16 snapshot replay (see the module doc); returns whether the
/// incumbent fused launch reproduced the snapshot bytes.
#[allow(clippy::too_many_arguments)]
fn replay_bf16(gpu: &mut Gpu, pm: &Option<Pm>, c: &Call, b: &Bufs, snap_sel: &[u8], snap_mirror: &[u8], qbytes: &[u8], pbytes: &[u8],
               arms: (Impl, Impl), shape_flag: usize, timed: usize, cpu_stride: usize, screen: bool,
               events: &[hip_bridge::Event; 3], rec: &mut Value, failures: &mut usize) -> Result<bool> {
    if arms == (Impl::Hipcc, Impl::Pm) { return Err("BF16 route rejects --score hipcc --select pm".into()) }
    if arms != (Impl::Hipcc, Impl::Hipcc) {
        let p = pm.as_ref().ok_or("--pm-image required for a pm arm")?;
        if p.score_bf16.is_none() || (arms.1 == Impl::Pm && p.select.is_none()) {
            return Err("pm image lacks the BF16 score/select symbols".into())
        }
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, FUSED_BF16).map_err(err)?;
    if shape_flag < c.block_count { return Err("--shape-blocks is below the active block count".into()) }
    let shape = shape_flag;
    let fused = Fused::new(gpu, shape)?;
    let io = Io { query: &b.query, pooled: &b.pooled, selected: &b.selected, mirror: &b.mirror };
    rec["route"] = json!("bf16");
    rec["shape_blocks"] = json!(shape);
    rec["fused_score_scratch"] = json!(if fused.global { "global" } else { "lds" });
    rec["fused_groups"] = json!(fused.groups(c).len());

    // Incumbent fused launch against the captured bytes.
    poison_outputs(gpu, b)?;
    let base = run_fused(gpu, c, &io, &fused, FUSED_BF16, true, None)?;
    let base_sel = download(gpu, &b.selected)?;
    let base_mirror = download(gpu, &b.mirror)?;
    let baseline_exact = base_sel == snap_sel && base_mirror == snap_mirror;
    rec["baseline_selected_equal"] = json!(base_sel == snap_sel);
    rec["baseline_mirror_equal"] = json!(base_mirror == snap_mirror);
    if !baseline_exact { *failures += 1; }
    if fused.global {
        rec["fused_scratch_outside_causal_poison_ok"] = json!(base.poison_ok);
        if !base.poison_ok { *failures += 1; }
    }
    if let Some(canon) = &base.canon {
        let score_keys = le_u32(canon);
        let query: Vec<f32> = qbytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        let pooled = widen_bf16(pbytes);
        let hipcc_sel = le_i32(&base_sel);
        if screen || cpu_stride > 0 {
            let (cpu, mut s) = cpu_checks(c, &query, &pooled, &score_keys, &hipcc_sel, cpu_stride.max(1), screen);
            if cpu["cpu_selection_rows_differ"] != 0 || cpu["rescore_bits_differ"] != 0 { *failures += 1; }
            rec["cpu"] = cpu;
            if screen { rec["screen"] = screen_json(&mut s); }
        }
    } else {
        let why = json!({"skipped":"lds_mode: fused scores live in LDS, no score capture; CPU rescoring/selection/screen not run"});
        if screen || cpu_stride > 0 { rec["cpu"] = why.clone(); }
        if screen { rec["screen"] = why; }
    }

    // Experimental pair against the fused bytes.
    if arms != (Impl::Hipcc, Impl::Hipcc) {
        poison_outputs(gpu, b)?;
        let (_, cap) = run_pm_bf16(gpu, pm, c, b, arms, true, None)?;
        let (arm_canon, poison_ok) = cap.unwrap();
        let sel_eq = download(gpu, &b.selected)? == base_sel;
        let mir_eq = download(gpu, &b.mirror)? == base_mirror;
        let score_eq = base.canon.as_ref().map(|bc| *bc == arm_canon);
        rec["arm"] = json!({"route":"bf16","score":"pm","select":if arms.1 == Impl::Pm {"pm"} else {"hipcc"},
            "selected_equal":sel_eq,"mirror_equal":mir_eq,
            "score_causal_bits_equal_fused":score_eq,
            "score_causal_comparison":if score_eq.is_some() {"performed"} else {"skipped_lds_mode_no_fused_score_capture"},
            "scratch_outside_causal_poison_ok":poison_ok});
        if !(sel_eq && mir_eq && poison_ok && score_eq.unwrap_or(true)) { *failures += 1; }
    }

    if timed > 0 {
        let median = |v: &[f64]| v[v.len() / 2];
        if arms == (Impl::Hipcc, Impl::Hipcc) {
            let lab = lab_source()?;
            gpu.ensure_kernel_public(LAB_MODULE, &lab, LAB_BF16).map_err(err)?;
            for _ in 0..2 {
                run_fused(gpu, c, &io, &fused, FUSED_BF16, false, Some(events))?;
                run_fused(gpu, c, &io, &fused, LAB_BF16, false, Some(events))?;
            }
            let (mut fu, mut sc, mut se) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..timed {
                let f = run_fused(gpu, c, &io, &fused, FUSED_BF16, false, Some(events))?.ms;
                let s = run_fused(gpu, c, &io, &fused, LAB_BF16, false, Some(events))?.ms;
                fu.push(f);
                sc.push(s);
                se.push(f - s);
            }
            fu.sort_by(f64::total_cmp);
            sc.sort_by(f64::total_cmp);
            se.sort_by(f64::total_cmp);
            rec["timing"] = json!({"reps":timed,"route":"bf16","fused_ms_median":median(&fu),"score_phase_ms_median":median(&sc),
                "select_phase_ms_median":median(&se),"fused_ms":fu,"score_phase_ms":sc,"select_phase_ms":se,
                "note":"fused_ms sums per-group launches; score_phase_ms is the appended lab kernel (fused pre-selection through the first __syncthreads, no static LDS); select_phase_ms = fused - score per rep"});
        } else {
            for _ in 0..2 { run_pm_bf16(gpu, pm, c, b, arms, false, Some(events))?; }
            let (mut sc, mut se) = (Vec::new(), Vec::new());
            for _ in 0..timed {
                let (t, _) = run_pm_bf16(gpu, pm, c, b, arms, false, Some(events))?;
                sc.push(t.score_ms);
                se.push(t.select_ms);
            }
            sc.sort_by(f64::total_cmp);
            se.sort_by(f64::total_cmp);
            rec["timing"] = json!({"reps":timed,"route":"bf16","score_ms_median":median(&sc),"select_ms_median":median(&se),
                "score_ms":sc,"select_ms":se});
        }
    }
    fused.free(gpu)?;
    Ok(baseline_exact)
}

/// An allocation `[guard | payload | guard]` of 0xa5 whose payload is exposed
/// as a non-owning sub-view; only `alloc` is ever freed.
struct Guarded { alloc: GpuTensor, view: GpuTensor, payload: usize }

impl Guarded {
    fn new(gpu: &Gpu, payload: usize) -> Result<Guarded> {
        let total = payload + 2 * GUARD_BYTES;
        let alloc = gpu.upload_raw(&vec![POISON; total], &[total]).map_err(err)?;
        let view = alloc.sub_offset(GUARD_BYTES, payload);
        Ok(Guarded { alloc, view, payload })
    }
    /// Poison guards and payload alike.
    fn poison(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip.memset(&self.alloc.buf, POISON as i32, self.alloc.byte_size()).map_err(err)
    }
    /// (both guards unchanged, payload bytes).
    fn read(&self, gpu: &Gpu) -> Result<(bool, Vec<u8>)> {
        let all = download(gpu, &self.alloc)?;
        let ok = all[..GUARD_BYTES].iter().all(|&x| x == POISON) && all[GUARD_BYTES + self.payload..].iter().all(|&x| x == POISON);
        Ok((ok, all[GUARD_BYTES..GUARD_BYTES + self.payload].to_vec()))
    }
}

/// `--tied-bf16-hipcc`: six all-tied fixtures through the fused launch (and
/// the PM pair iff an image is supplied) against the CPU stable-order oracle,
/// selected/mirror each inside 64 KiB guards.
fn run_tied_bf16(gpu: &mut Gpu, pm: &Option<Pm>, shape_flag: usize, out_f: &mut std::fs::File, failures: &mut usize) -> Result<()> {
    if let Some(p) = pm {
        if p.score_bf16.is_none() || p.select.is_none() { return Err("--tied-bf16-hipcc with --pm-image needs the BF16 score and select symbols".into()) }
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, FUSED_BF16).map_err(err)?;
    for blocks in [16384usize, 32768, 65536] {
        for vbits in [0x0000u16, 0x3f80] {
            let value = f32::from_bits((vbits as u32) << 16);
            let (rows, compress) = (3usize, 4usize);
            // The last row sees exactly `blocks` complete blocks plus a 3-token tail.
            let position_start = blocks * compress + 3 - rows;
            let c = Call { rows, stride: 512, block_count: blocks, dim: 128, compress, position_start, budget: 512, capacity: 2051 };
            let shape = blocks.max(shape_flag);
            let qbytes: Vec<u8> = (0..rows * 512).flat_map(|_| value.to_le_bytes()).collect();
            let pbytes: Vec<u8> = (0..blocks * 128).flat_map(|_| vbits.to_le_bytes()).collect();
            let query = gpu.upload_raw(&qbytes, &[qbytes.len()]).map_err(err)?;
            let pooled = gpu.upload_raw(&pbytes, &[pbytes.len()]).map_err(err)?;
            let sel_g = Guarded::new(gpu, rows * c.capacity * 4)?;
            let mir_g = Guarded::new(gpu, c.capacity * 4)?;
            let fused = Fused::new(gpu, shape)?;

            // CPU stable oracle over the (all-tied) scores.
            let score_bits = cpu_score(&vec![value; 512], &vec![value; 128], 128).0.to_bits();
            let keys = vec![score_bits; blocks];
            let check = |sel: &[u8], mir: &[u8]| -> (bool, bool) {
                let (got, mirror) = (le_i32(sel), le_i32(mir));
                let (mut s_ok, mut m_ok) = (true, true);
                for r in 0..rows {
                    let want = selection_row(&rank(&keys[..c.row_blocks(r)]), &c, r);
                    s_ok &= got[r * c.capacity..(r + 1) * c.capacity] == want[..];
                    if r == rows - 1 { m_ok &= mirror == want; }
                }
                (s_ok, m_ok)
            };

            sel_g.poison(gpu)?;
            mir_g.poison(gpu)?;
            let io = Io { query: &query, pooled: &pooled, selected: &sel_g.view, mirror: &mir_g.view };
            let run = run_fused(gpu, &c, &io, &fused, FUSED_BF16, true, None)?;
            let (sel_guard, sel) = sel_g.read(gpu)?;
            let (mir_guard, mir) = mir_g.read(gpu)?;
            let (s_ok, m_ok) = check(&sel, &mir);
            let uniform = run.canon.as_ref().map_or(true, |canon| {
                (0..rows).all(|r| le_u32(&canon[r * blocks * 4..r * blocks * 4 + c.row_blocks(r) * 4]).iter().all(|&k| k == score_bits))
            });
            let fused_ok = s_ok && m_ok && sel_guard && mir_guard && run.poison_ok;
            if !fused_ok { *failures += 1; }
            let mut rec = json!({"tied_bf16":true,"route":"bf16","blocks":blocks,"value_bf16":format!("0x{vbits:04x}"),
                "query_value":value,"rows":rows,"position_start":position_start,"shape_blocks":shape,
                "fused_score_scratch":if fused.global {"global"} else {"lds"},
                "fused":{"selected_equal_cpu_oracle":s_ok,"mirror_equal_cpu_oracle":m_ok,"selected_guards_ok":sel_guard,
                    "mirror_guards_ok":mir_guard,"scratch_outside_causal_poison_ok":run.poison_ok,
                    "scores_all_equal_cpu_bits":uniform,"ok":fused_ok}});

            if pm.is_some() {
                // Re-poison guards and payloads, then the candidate pair.
                sel_g.poison(gpu)?;
                mir_g.poison(gpu)?;
                let group = c.group();
                let sbytes = group * blocks * 4;
                let scores = gpu.upload_raw(&vec![POISON; sbytes], &[sbytes]).map_err(err)?;
                let b = Bufs { query: query.shallow_clone(), pooled: pooled.shallow_clone(), scores,
                    selected: sel_g.view.shallow_clone(), mirror: mir_g.view.shallow_clone() };
                let (_, cap) = run_pm_bf16(gpu, pm, &c, &b, (Impl::Pm, Impl::Pm), true, None)?;
                let (_, poison_ok) = cap.unwrap();
                let Bufs { scores, .. } = b;
                gpu.free_tensor(scores).map_err(err)?;
                let (sel_guard, sel) = sel_g.read(gpu)?;
                let (mir_guard, mir) = mir_g.read(gpu)?;
                let (s_ok, m_ok) = check(&sel, &mir);
                let pm_ok = s_ok && m_ok && sel_guard && mir_guard && poison_ok;
                if !pm_ok { *failures += 1; }
                rec["pm"] = json!({"selected_equal_cpu_oracle":s_ok,"mirror_equal_cpu_oracle":m_ok,"selected_guards_ok":sel_guard,
                    "mirror_guards_ok":mir_guard,"scratch_outside_causal_poison_ok":poison_ok,"ok":pm_ok});
            }
            eprintln!("{rec}");
            writeln!(out_f, "{rec}").map_err(err)?;
            fused.free(gpu)?;
            for t in [query, pooled, sel_g.alloc, mir_g.alloc] { gpu.free_tensor(t).map_err(err)?; }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- CPU side

fn f16_round(x: f32) -> f32 {
    if !x.is_finite() { return x; }
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        return (x * 16_777_216.0).round_ties_even() / 16_777_216.0;
    }
    let b = x.to_bits();
    let r = f32::from_bits((b + 0xfff + ((b >> 13) & 1)) & !0x1fff);
    if r.abs() > 65504.0 { f32::INFINITY.copysign(x) } else { r }
}

fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() { return x; }
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// Per head serial d-order F32 FMA chain from +0, `max(dot / sqrt(dim), 0)`
/// per head, heads added in order: the incumbent's source arithmetic.
/// Also reports per-head pre-ReLU signs (bit h set if dot > 0).
fn cpu_score(q: &[f32], key: &[f32], dim: usize) -> (f32, u8) {
    let scale = (dim as f32).sqrt();
    let mut score = 0.0f32;
    let mut signs = 0u8;
    for h in 0..4 {
        let qh = &q[h * dim..(h + 1) * dim];
        let mut dot = 0.0f32;
        for d in 0..dim { dot = qh[d].mul_add(key[d], dot); }
        let s = dot / scale;
        if s > 0.0 { signs |= 1 << h; }
        score += if s > 0.0 { s } else { 0.0 };
    }
    (score, signs)
}

fn rank(keys: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| keys[b as usize].cmp(&keys[a as usize]).then(a.cmp(&b)));
    order
}

fn selection_row(order: &[u32], c: &Call, row: usize) -> Vec<i32> {
    let visible = c.position_start + row + 1;
    let blocks = c.row_blocks(row);
    let chosen = c.budget.min(blocks);
    let mut out = vec![-1i32; c.capacity];
    for (slot, &block) in order.iter().take(chosen).enumerate() {
        for o in 0..c.compress {
            let (i, t) = (slot * c.compress + o, block as usize * c.compress + o);
            if i < c.capacity && t < visible { out[i] = t as i32; }
        }
    }
    let mut i = chosen * c.compress;
    for t in blocks * c.compress..visible {
        if i >= c.capacity { break; }
        out[i] = t as i32;
        i += 1;
    }
    out
}

#[derive(Default, Clone)]
struct Screen {
    rows: usize,
    finish: [usize; 5],          // index 0: chosen == visible blocks (no refinement)
    all_equal: usize,
    pad512: usize,
    pad1024: usize,
    over1024_after: [Vec<usize>; 4], // candidates (above + boundary) after byte b
    boundary_after: [Vec<usize>; 4],
}

/// Adaptive exact prefix selection replay over one row's keys.
fn screen_row(keys: &[u32], chosen: usize, s: &mut Screen) {
    s.rows += 1;
    let (mn, mx) = keys.iter().fold((u32::MAX, 0u32), |(a, b), &k| (a.min(k), b.max(k)));
    if mn == mx { s.all_equal += 1; }
    if chosen == keys.len() {
        s.finish[0] += 1;
        s.pad512 += 1;
        return;
    }
    let (mut prefix, mut need, mut above) = (0u32, chosen, 0usize);
    let mut done = false;
    for pass in 0..4 {
        let shift = 24 - 8 * pass;
        let high = if pass == 0 { 0 } else { !0u32 << (shift + 8) };
        let mut hist = [0usize; 256];
        for &k in keys { if k & high == prefix { hist[((k >> shift) & 255) as usize] += 1; } }
        let mut acc = 0usize;
        let mut digit = 255usize;
        loop {
            if acc + hist[digit] >= need { break; }
            acc += hist[digit];
            digit -= 1;
        }
        above += acc;
        need -= acc;
        prefix |= (digit as u32) << shift;
        let boundary = hist[digit];
        let cand = above + boundary;
        s.over1024_after[pass].push(cand);
        s.boundary_after[pass].push(boundary);
        if !done && (cand <= 1024 || pass == 3) {
            done = true;
            s.finish[pass + 1] += 1;
            let final_cand = if pass == 3 { above + need } else { cand };
            if final_cand <= 512 { s.pad512 += 1 } else { s.pad1024 += 1 }
        }
    }
}

fn quant(v: &mut [usize]) -> Value {
    if v.is_empty() { return Value::Null; }
    v.sort_unstable();
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    json!({"n":v.len(),"min":v[0],"p50":q(0.5),"p90":q(0.9),"p99":q(0.99),"max":v[v.len()-1]})
}

fn screen_json(s: &mut Screen) -> Value {
    let rows = s.rows.max(1) as f64;
    let by_byte2 = (s.finish[0] + s.finish[1] + s.finish[2]) as f64 / rows;
    let cands: Vec<Value> = s.over1024_after.iter_mut().map(|v| quant(v)).collect();
    let bounds: Vec<Value> = s.boundary_after.iter_mut().map(|v| quant(v)).collect();
    json!({"rows":s.rows,"finish_no_refine":s.finish[0],"finish_after_byte":[s.finish[1],s.finish[2],s.finish[3],s.finish[4]],
        "frac_finish_by_byte":[ (s.finish[0]+s.finish[1]) as f64/rows, by_byte2,
            (s.finish[0]+s.finish[1]+s.finish[2]+s.finish[3]) as f64/rows, 1.0],
        "all_equal":s.all_equal,"pad512":s.pad512,"pad1024":s.pad1024,
        "candidates_after_byte":cands,"boundary_after_byte":bounds,"by_byte2_frac":by_byte2})
}

#[derive(Default)]
struct Diag { rows: usize, score_bits_differ: u64, scores: u64, relu_sign_changes: u64, ordered_rows_differ: usize, ordered_elems_differ: u64, set_rows_differ: usize, set_elems_differ: u64 }

// ---------------------------------------------------------------- main

fn cpu_checks(c: &Call, query: &[f32], pooled: &[f32], scores: &[u32], hipcc_sel: &[i32], cpu_stride: usize, screen: bool)
    -> (Value, Screen) {
    let bc = c.block_count;
    // Stable CPU selection over the hipcc scores, every row.
    let sel_mismatch: Vec<usize> = (0..c.rows).into_par_iter().filter(|&r| {
        let keys = &scores[r * bc..r * bc + c.row_blocks(r)];
        selection_row(&rank(keys), c, r) != hipcc_sel[r * c.capacity..(r + 1) * c.capacity]
    }).collect();
    // Exact rescoring and WMMA operand-rounding diagnostic on sampled rows.
    let sampled: Vec<usize> = (0..c.rows).filter(|r| r % cpu_stride == 0 || *r == c.rows - 1).collect();
    let per_row: Vec<(u64, u64, [Diag; 2], f64)> = sampled.par_iter().map(|&r| {
        let q = &query[r * c.stride..r * c.stride + 4 * c.dim];
        let blocks = c.row_blocks(r);
        let qs = [q.iter().map(|&x| f16_round(x)).collect::<Vec<_>>(), q.iter().map(|&x| bf16_round(x)).collect::<Vec<_>>()];
        let mut exact = Vec::with_capacity(blocks);
        let mut signs = Vec::with_capacity(blocks);
        let mut alt = [Vec::with_capacity(blocks), Vec::with_capacity(blocks)];
        let mut alt_signs = [Vec::with_capacity(blocks), Vec::with_capacity(blocks)];
        let mut differ = 0u64;
        for b in 0..blocks {
            let key = &pooled[b * c.dim..(b + 1) * c.dim];
            let (s, sg) = cpu_score(q, key, c.dim);
            if s.to_bits() != scores[r * bc + b] { differ += 1; }
            exact.push(s.to_bits());
            signs.push(sg);
            let k16: Vec<f32> = key.iter().map(|&x| f16_round(x)).collect();
            for (v, (qv, kv)) in [(&qs[0], &k16), (&qs[1], &key.to_vec())].into_iter().enumerate() {
                let (s, sg) = cpu_score(qv, kv, c.dim);
                alt[v].push(s.to_bits());
                alt_signs[v].push(sg);
            }
        }
        let chosen = c.budget.min(blocks);
        let order = rank(&exact);
        let margin = if blocks > chosen && chosen > 0 {
            let (a, b) = (f32::from_bits(exact[order[chosen - 1] as usize]), f32::from_bits(exact[order[chosen] as usize]));
            ((a - b) / a.max(f32::MIN_POSITIVE)) as f64
        } else { f64::NAN };
        let mut diags = [Diag::default(), Diag::default()];
        for v in 0..2 {
            let d = &mut diags[v];
            d.rows = 1;
            d.scores = blocks as u64;
            d.score_bits_differ = exact.iter().zip(&alt[v]).filter(|(a, b)| a != b).count() as u64;
            d.relu_sign_changes = signs.iter().zip(&alt_signs[v]).map(|(a, b)| (a ^ b).count_ones() as u64).sum();
            let ao = rank(&alt[v]);
            let e = order[..chosen].iter().zip(&ao[..chosen]).filter(|(a, b)| a != b).count() as u64;
            d.ordered_elems_differ = e;
            d.ordered_rows_differ = usize::from(e > 0);
            let es: std::collections::BTreeSet<u32> = order[..chosen].iter().copied().collect();
            let miss = ao[..chosen].iter().filter(|x| !es.contains(x)).count() as u64;
            d.set_elems_differ = miss;
            d.set_rows_differ = usize::from(miss > 0);
        }
        (differ, blocks as u64, diags, margin)
    }).collect();
    let rescore_differ: u64 = per_row.iter().map(|x| x.0).sum();
    let rescored: u64 = per_row.iter().map(|x| x.1).sum();
    let mut margins: Vec<f64> = per_row.iter().map(|x| x.3).filter(|m| m.is_finite()).collect();
    margins.sort_by(f64::total_cmp);
    let mut wmma = Vec::new();
    for (v, name) in ["f16_operands", "bf16_operands"].iter().enumerate() {
        let mut t = Diag::default();
        for x in &per_row {
            let d = &x.2[v];
            t.rows += d.rows; t.scores += d.scores; t.score_bits_differ += d.score_bits_differ;
            t.relu_sign_changes += d.relu_sign_changes; t.ordered_rows_differ += d.ordered_rows_differ;
            t.ordered_elems_differ += d.ordered_elems_differ; t.set_rows_differ += d.set_rows_differ; t.set_elems_differ += d.set_elems_differ;
        }
        wmma.push(json!({"variant":name,"rows":t.rows,"scores":t.scores,"score_bits_differ":t.score_bits_differ,
            "relu_sign_changes":t.relu_sign_changes,"rows_ordered_selection_differ":t.ordered_rows_differ,
            "ordered_elements_differ":t.ordered_elems_differ,"rows_set_differ":t.set_rows_differ,"set_elements_differ":t.set_elems_differ}));
    }
    let mut s = Screen::default();
    if screen {
        let parts: Vec<Screen> = (0..c.rows).into_par_iter().fold(Screen::default, |mut s, r| {
            let blocks = c.row_blocks(r);
            let chosen = c.budget.min(blocks);
            if chosen > 0 { screen_row(&scores[r * bc..r * bc + blocks], chosen, &mut s); }
            s
        }).collect();
        for p in parts {
            s.rows += p.rows; s.all_equal += p.all_equal; s.pad512 += p.pad512; s.pad1024 += p.pad1024;
            for i in 0..5 { s.finish[i] += p.finish[i]; }
            for i in 0..4 { s.over1024_after[i].extend(&p.over1024_after[i]); s.boundary_after[i].extend(&p.boundary_after[i]); }
        }
    }
    let m = |p: f64| if margins.is_empty() { f64::NAN } else { margins[((margins.len() - 1) as f64 * p) as usize] };
    (json!({"cpu_selection_rows_differ":sel_mismatch.len(),"cpu_selection_first_differ":sel_mismatch.first(),
        "rescored_rows":sampled.len(),"rescored_scores":rescored,"rescore_bits_differ":rescore_differ,
        "kth_rel_margin":{"min":m(0.0),"p10":m(0.1),"p50":m(0.5)},
        "wmma_cpu_emulation":wmma,
        "wmma_note":"CPU emulation: F16/BF16 operand rounding with the incumbent serial F32 FMA chain kept; a real WMMA port also reassociates, so these are lower bounds"}), s)
}

fn main() -> Result<()> {
    if std::env::var_os("HIPFIRE_LOCK_DIR").is_none() { return Err("private HIPFIRE_LOCK_DIR required".into()) }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut dirs, mut out, mut timed, mut image) = (Vec::new(), None, 0usize, None::<PathBuf>);
    let (mut arms, mut cpu_stride, mut screen, mut tied) = ((Impl::Hipcc, Impl::Hipcc), 64usize, true, false);
    let (mut g3, mut beta_src) = (false, None::<PathBuf>);
    let (mut shape_flag, mut tied_bf16, mut g3_bf16) = (65536usize, false, false);
    let parse_impl = |v: Option<String>| -> Result<Impl> {
        match v.as_deref() { Some("hipcc") => Ok(Impl::Hipcc), Some("pm") => Ok(Impl::Pm), _ => Err("arm must be hipcc|pm".into()) }
    };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next(),
            "--time" => timed = it.next().ok_or("--time N")?.parse().map_err(err)?,
            "--pm-image" => image = Some(PathBuf::from(it.next().ok_or("--pm-image FILE")?)),
            "--score" => arms.0 = parse_impl(it.next())?,
            "--select" => arms.1 = parse_impl(it.next())?,
            "--cpu-rows" => cpu_stride = it.next().ok_or("--cpu-rows N")?.parse().map_err(err)?,
            "--no-screen" => screen = false,
            "--tied" => tied = true,
            "--g3" => g3 = true,
            "--beta-src" => beta_src = Some(PathBuf::from(it.next().ok_or("--beta-src ABS_FILE")?)),
            "--shape-blocks" => shape_flag = it.next().ok_or("--shape-blocks N")?.parse().map_err(err)?,
            "--tied-bf16-hipcc" => tied_bf16 = true,
            "--g3-bf16" => g3_bf16 = true,
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    let out = out.ok_or("--out OUT.jsonl required")?;
    if g3_bf16 {
        if g3 || beta_src.is_some() || image.is_some() || tied || tied_bf16 || timed > 0
            || arms != (Impl::Hipcc, Impl::Hipcc) || cpu_stride != 64 || !screen || shape_flag != 65536 {
            return Err("--g3-bf16 takes only SNAPSHOT_DIR... and --out (no --g3/--beta-src/--pm-image/--tied*/--time/--score/--select/--cpu-rows/--no-screen/--shape-blocks)".into())
        }
        #[cfg(feature = "lab")]
        return g3_bf16::run(&dirs, Path::new(&out));
        #[cfg(not(feature = "lab"))]
        return Err("--g3-bf16 needs --features lab".into());
    }
    if g3 {
        #[cfg(feature = "lab")]
        return g3::run(&dirs, Path::new(&out), beta_src, image.is_some());
        #[cfg(not(feature = "lab"))]
        return Err("--g3 needs --features lab".into());
    }
    let mut paths = Vec::new();
    for d in &dirs { snapshots(d, &mut paths)?; }
    if paths.is_empty() && !tied && !tied_bf16 { return Err("no snapshot.json found".into()) }
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch != "gfx1151" { return Err(format!("gfx1151 only, got {}", gpu.arch)) }
    for k in [HIPCC_SCORE, HIPCC_SELECT] { gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, k).map_err(err)?; }
    let pm = match &image {
        Some(p) => {
            if !p.is_absolute() { return Err("--pm-image must be absolute".into()) }
            let bytes = std::fs::read(p).map_err(err)?;
            let module = gpu.hip.module_load_data(&bytes).map_err(err)?;
            let score = gpu.hip.module_get_function(&module, PM_SCORE).ok();
            let score_bf16 = gpu.hip.module_get_function(&module, PM_SCORE_BF16).ok();
            let select = gpu.hip.module_get_function(&module, PM_SELECT).ok();
            eprintln!("pm image {} sha256 {} score={} score_bf16={} select={}", p.display(), sha256(&bytes), score.is_some(), score_bf16.is_some(), select.is_some());
            Some(Pm { _module: module, score, score_bf16, select })
        }
        None => None,
    };
    let mut out_f = std::fs::File::create(&out).map_err(err)?;
    let events = [gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?];
    let mut failures = 0usize;

    for path in &paths {
        let root = path.parent().unwrap();
        let header: Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if header["schema"] != "qsa-source-v1" { return Err(format!("{}: unsupported schema", path.display())) }
        let sel = &header["identity"]["selector"];
        if header["identity"]["arch"] != "gfx1151" { return Err("snapshot arch not gfx1151".into()) }
        let bf16 = sel["pooled_dtype"] == "BF16";
        if !(sel["pooled_dtype"] == "F32" || bf16) || sel["projection_dtype"] != "F32" { return Err(format!("{}: not the F32/BF16 pooled route", path.display())) }
        if sel["heads"] != 4 || sel["dim"] != 128 { return Err(format!("{}: uncovered selector geometry", path.display())) }
        let get = |name: &str| -> Result<Vec<u8>> {
            let b = std::fs::read(root.join(name)).map_err(err)?;
            if header["files"][name]["sha256"] != sha256(&b) { return Err(format!("{}: hash mismatch {name}", path.display())) }
            Ok(b)
        };
        let g = &header["geometry"];
        let n = |v: &Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("missing {k}"));
        let (rows, position_start, compress) = (n(g, "rows")?, n(g, "position_start")?, n(g, "compress")?);
        let c = Call { rows, stride: n(sel, "projection_stride")?, block_count: (position_start + rows) / compress, dim: 128,
            compress, position_start, budget: n(g, "budget_blocks")?, capacity: n(g, "capacity")? };
        if c.budget > 512 || c.block_count == 0 { return Err("uncovered budget/block count".into()) }
        let pooled_capacity = n(sel, "pooled_capacity")?;
        let qbytes = get("index-projection.f32")?;
        let pbytes = get("pooled.source")?;
        let snap_sel = get("selected.i32")?;
        let snap_mirror = get("selected-mirror.i32")?;
        let pooled_elem = if bf16 { 2 } else { 4 };
        if qbytes.len() != rows * c.stride * 4 || pbytes.len() != pooled_capacity * 128 * pooled_elem || snap_sel.len() != rows * c.capacity * 4
            || snap_mirror.len() != c.capacity * 4 || c.block_count > pooled_capacity {
            return Err(format!("{}: geometry/extent mismatch", path.display()))
        }
        let group = c.group();
        let bufs = Bufs {
            query: gpu.upload_raw(&qbytes, &[qbytes.len()]).map_err(err)?,
            pooled: gpu.upload_raw(&pbytes, &[pbytes.len()]).map_err(err)?,
            scores: gpu.upload_raw(&vec![0u8; group * c.block_count * 4], &[group * c.block_count * 4]).map_err(err)?,
            selected: gpu.upload_raw(&vec![0u8; snap_sel.len()], &[snap_sel.len()]).map_err(err)?,
            mirror: gpu.upload_raw(&vec![0u8; snap_mirror.len()], &[snap_mirror.len()]).map_err(err)?,
        };
        let mut rec = json!({"snapshot":path.display().to_string(),"ctx":header["identity"]["ctx"],"layer":header["identity"]["layer"],
            "position_start":position_start,"rows":rows,"block_count":c.block_count,"group":group,
            "groups":rows.div_ceil(group),"budget_blocks":c.budget,"capacity":c.capacity,"stride":c.stride});
        if bf16 {
            let exact = replay_bf16(&mut gpu, &pm, &c, &bufs, &snap_sel, &snap_mirror, &qbytes, &pbytes, arms, shape_flag, timed,
                cpu_stride, screen, &events, &mut rec, &mut failures)?;
            eprintln!("{} bf16 baseline_exact={} {}", path.display(), exact, rec.get("timing").map(|t| t.to_string()).unwrap_or_default());
            writeln!(out_f, "{rec}").map_err(err)?;
            out_f.flush().map_err(err)?;
            for t in [bufs.query, bufs.pooled, bufs.scores, bufs.selected, bufs.mirror] { gpu.free_tensor(t).map_err(err)?; }
            continue;
        }

        // Baseline hipcc/hipcc against the captured bytes, keeping the scores.
        poison_outputs(&gpu, &bufs)?;
        let mut base_scores = Vec::new();
        run_call(&mut gpu, &pm, &c, &bufs, (Impl::Hipcc, Impl::Hipcc), Some(&mut base_scores), None)?;
        let base_sel = download(&gpu, &bufs.selected)?;
        let base_mirror = download(&gpu, &bufs.mirror)?;
        let baseline_exact = base_sel == snap_sel && base_mirror == snap_mirror;
        rec["baseline_selected_equal"] = json!(base_sel == snap_sel);
        rec["baseline_mirror_equal"] = json!(base_mirror == snap_mirror);
        if !baseline_exact { failures += 1; }
        let score_keys: Vec<u32> = base_scores.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let query: Vec<f32> = qbytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let pooled: Vec<f32> = pbytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let hipcc_sel: Vec<i32> = base_sel.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        if screen || cpu_stride > 0 {
            let (cpu, mut s) = cpu_checks(&c, &query, &pooled, &score_keys, &hipcc_sel, cpu_stride.max(1), screen);
            if cpu["cpu_selection_rows_differ"] != 0 || cpu["rescore_bits_differ"] != 0 { failures += 1; }
            rec["cpu"] = cpu;
            if screen { rec["screen"] = screen_json(&mut s); }
        }

        // Experimental arm: byte identity against the baseline.
        if arms != (Impl::Hipcc, Impl::Hipcc) {
            poison_outputs(&gpu, &bufs)?;
            let mut arm_scores = Vec::new();
            run_call(&mut gpu, &pm, &c, &bufs, arms, Some(&mut arm_scores), None)?;
            let sel_eq = download(&gpu, &bufs.selected)? == base_sel;
            let mir_eq = download(&gpu, &bufs.mirror)? == base_mirror;
            // Scores: every valid (causal) score bit equal and all poison
            // outside them untouched (the baseline leaves the same poison).
            let score_eq = arm_scores == base_scores;
            rec["arm"] = json!({"score":if arms.0==Impl::Pm {"pm"} else {"hipcc"},"select":if arms.1==Impl::Pm {"pm"} else {"hipcc"},
                "selected_equal":sel_eq,"mirror_equal":mir_eq,"score_bytes_equal_incl_poison":score_eq});
            if !(sel_eq && mir_eq && score_eq) { failures += 1; }
        }

        if timed > 0 {
            for _ in 0..2 { run_call(&mut gpu, &pm, &c, &bufs, arms, None, Some(&events))?; }
            let mut sc = Vec::new();
            let mut se = Vec::new();
            for _ in 0..timed {
                let t = run_call(&mut gpu, &pm, &c, &bufs, arms, None, Some(&events))?;
                sc.push(t.score_ms);
                se.push(t.select_ms);
            }
            sc.sort_by(f64::total_cmp);
            se.sort_by(f64::total_cmp);
            rec["timing"] = json!({"reps":timed,"score_ms_median":sc[sc.len()/2],"select_ms_median":se[se.len()/2],
                "score_ms":sc,"select_ms":se});
        }
        eprintln!("{} baseline_exact={} {}", path.display(), baseline_exact, rec.get("timing").map(|t| t.to_string()).unwrap_or_default());
        writeln!(out_f, "{rec}").map_err(err)?;
        out_f.flush().map_err(err)?;
        for t in [bufs.query, bufs.pooled, bufs.scores, bufs.selected, bufs.mirror] { gpu.free_tensor(t).map_err(err)?; }
    }

    if tied {
        if arms.1 != Impl::Pm { return Err("--tied runs the PM selector only (never the pre-fix hipcc arm)".into()) }
        for (blocks, value) in [(16384usize, 1.0f32), (16384, 0.0), (32768, 1.0), (65536, 1.0), (65536, 0.0)] {
            let compress = 4;
            let rows = 3usize;
            // The last row sees exactly `blocks` complete blocks plus a 3-token tail.
            let position_start = blocks * compress + 3 - rows;
            let c = Call { rows, stride: 0, block_count: blocks, dim: 128, compress, position_start, budget: 512, capacity: 512 * compress + compress - 1 };
            let scores: Vec<u8> = (0..rows * blocks).flat_map(|_| value.to_le_bytes()).collect();
            let sel_len = rows * c.capacity * 4;
            let b = Bufs {
                query: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
                pooled: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
                scores: gpu.upload_raw(&scores, &[scores.len()]).map_err(err)?,
                selected: gpu.upload_raw(&vec![0u8; sel_len], &[sel_len]).map_err(err)?,
                mirror: gpu.upload_raw(&vec![0u8; c.capacity * 4], &[c.capacity * 4]).map_err(err)?,
            };
            poison_outputs(&gpu, &b)?;
            let mut blob = select_blob(&c, &b, 0, rows, true);
            launch(&mut gpu, &pm, Impl::Pm, false, [rows as u32, 1, 1], &mut blob)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let got: Vec<i32> = download(&gpu, &b.selected)?.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
            let mirror: Vec<i32> = download(&gpu, &b.mirror)?.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
            let keys = vec![value.to_bits(); blocks];
            let mut ok = true;
            for r in 0..rows {
                let rb = c.row_blocks(r);
                let want = selection_row(&rank(&keys[..rb]), &c, r);
                ok &= got[r * c.capacity..(r + 1) * c.capacity] == want[..];
                if r == rows - 1 { ok &= mirror == want; }
            }
            if !ok { failures += 1; }
            let rec = json!({"tied":true,"blocks":blocks,"value":value,"rows":rows,"position_start":position_start,"pm_select_equal_cpu_oracle":ok});
            eprintln!("{rec}");
            writeln!(out_f, "{rec}").map_err(err)?;
            for t in [b.query, b.pooled, b.scores, b.selected, b.mirror] { gpu.free_tensor(t).map_err(err)?; }
        }
    }
    if tied_bf16 { run_tied_bf16(&mut gpu, &pm, shape_flag, &mut out_f, &mut failures)?; }
    for e in events { gpu.hip.event_destroy(e).map_err(err)?; }
    if failures > 0 { return Err(format!("{failures} failing checks; see {out}")) }
    Ok(())
}

// ---------------------------------------------------------------- G3

/// `--g3 --beta-src ABS_FILE`: exact oracle + graph-safety gate of the PM QSA
/// selector pair as the default gfx1151 prefill route.  Needs
/// `--features lab` (the explicit-arm helper is lab-only):
/// `cargo build --release -p rdna-compute --features lab --example qsa_select_pm_check`.
///
/// `qsa_select_pm_check SNAPSHOT_DIR... --g3 --beta-src ABS_FILE --out OUT.jsonl`
///
/// Arms (one JSONL record per case x arm, then one summary line; nonzero exit
/// on any failing record or refusal):
/// - `raw-fixed` / `raw-pm` / `beta`: raw-ABI production grouping (own 64 MiB
///   scratch, scores downloaded) of the in-tree hipcc pair / embedded PM image /
///   the pre-fix hipcc pair compiled from `--beta-src` as a separate module;
/// - `fixed` / `pm` / `pm-score-only` / `pm-select-only`: the production helper
///   `indexed_attention_select_batch_pm_arm` with `Hipcc` / `Both` / `ScoreOnly` /
///   `SelectOnly`;
/// - `pm-graph`: the PM pair's raw launches captured into a HIP graph on a
///   private stream (buffers reserved and poisoned outside capture), replayed
///   twice and compared with the eager `pm` arm;
/// - `live`: the public route (`indexed_attention_select_batch[_mirrored]`),
///   default flags, which must equal the `pm` arm.
/// Every arm runs twice serially (`repeat_identical`).  Oracle: CPU stable
/// `(score desc, block asc)` selection + tail/fill/mirror over the fixed
/// arm's scores (sampled rows rescored on the CPU bit for bit); snapshot cases
/// are also compared with the captured `selected.i32` / `selected-mirror.i32`.
#[cfg(feature = "lab")]
mod g3 {
    use super::*;
    use hip_bridge::{Graph, GraphExec, Stream};
    use rdna_compute::tensor_ops::{
        indexed_attention_select_batch, indexed_attention_select_batch_mirrored,
        indexed_attention_select_batch_pm_arm, IndexedAttentionSelectBatch, QsaSelectPmArm,
    };

    const PM_IMAGE: &[u8] = include_bytes!("../../../kernels/qsa_select_pm_gfx1151.hxaco");
    const BETA_MODULE: &str = "qsa_select_beta_060cadcd3b";
    /// Poisoned bytes after `selected` / `mirror`: any stray write shows.
    const CANARY: usize = 4096;
    /// The pre-fix tie-count overflow range: beta never runs from here.
    const BETA_MAX_BLOCKS: usize = 32768;
    const PM_MAX_BLOCKS: usize = 511 * 8 * 32;
    const OOB: usize = 0x7fff_ff00;

    #[derive(Clone, Copy, PartialEq)]
    enum RawImpl { Fixed, Beta, Pm }

    struct Env {
        _beta: Module, beta_score: Function, beta_select: Function,
        _pm: Module, pm_score: Function, pm_select: Function,
    }

    struct Sink { f: std::fs::File, failures: usize, records: usize, cases: usize }
    impl Sink {
        fn put(&mut self, mut rec: Value, ok: bool) -> Result<()> {
            rec["ok"] = json!(ok);
            if !ok { self.failures += 1; eprintln!("FAIL {rec}"); }
            self.records += 1;
            writeln!(self.f, "{rec}").map_err(err)?;
            self.f.flush().map_err(err)
        }
    }

    struct Case {
        name: String,
        c: Call,
        shape_blocks: usize,
        qbytes: Vec<u8>,
        pbytes: Vec<u8>,
        qh: String,
        ph: String,
        snap: Option<(Vec<u8>, Vec<u8>)>,
        mirror: bool,
        beta_skip: Option<&'static str>,
    }

    struct Out { sel: Vec<u8>, mirror: Vec<u8>, scores: Vec<u8>, canary: bool, persisted: Option<bool> }
    fn same(a: &Out, b: &Out) -> bool { a.sel == b.sel && a.mirror == b.mirror && a.scores == b.scores && a.canary == b.canary }

    struct Expect<'a> { mirror: bool, sel: &'a [u8], mirror_bytes: &'a [u8], snap: Option<&'a (Vec<u8>, Vec<u8>)> }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn bytes_f32(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect() }
    fn word_diff(a: &[u8], b: &[u8]) -> u64 {
        if a.len() != b.len() { return u64::MAX }
        a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x != y).count() as u64
    }

    // ------------------------------------------------------------ launches

    fn raw_launch(gpu: &mut Gpu, env: &Env, which: RawImpl, score: bool, grid: [u32; 3], blob: &mut KernargBlob) -> Result<()> {
        match which {
            RawImpl::Fixed => gpu
                .launch_kernel_blob(if score { HIPCC_SCORE } else { HIPCC_SELECT }, grid, [256, 1, 1], 0, blob.as_mut_slice())
                .map_err(err),
            RawImpl::Beta => {
                let f = if score { &env.beta_score } else { &env.beta_select };
                unsafe { gpu.hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, blob.as_mut_slice()) }.map_err(err)
            }
            RawImpl::Pm => {
                let f = if score { &env.pm_score } else { &env.pm_select };
                unsafe { gpu.hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, blob.as_mut_slice()) }.map_err(err)
            }
        }
    }

    /// The production pair over all groups with the given kernels (`keep`
    /// downloads each group's scratch after its score launch).
    fn raw_call(gpu: &mut Gpu, env: &Env, which: RawImpl, c: &Call, b: &Bufs, mirror: bool, mut keep: Option<&mut Vec<u8>>) -> Result<()> {
        let group = c.group();
        let mut g0 = 0;
        while g0 < c.rows {
            let n = group.min(c.rows - g0);
            if keep.is_some() { gpu.hip.memset(&b.scores.buf, POISON as i32, b.scores.byte_size()).map_err(err)?; }
            let mut blob = score_blob(c, b, g0, n);
            raw_launch(gpu, env, which, true, [c.block_count.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], &mut blob)?;
            if let Some(k) = keep.as_deref_mut() {
                gpu.hip.device_synchronize().map_err(err)?;
                let mut bytes = vec![0u8; n * c.block_count * 4];
                let len = bytes.len();
                gpu.hip.memcpy_dtoh(&mut bytes, &b.scores.buf.byte_view(0, len)).map_err(err)?;
                k.extend_from_slice(&bytes);
            }
            let mut blob = select_blob(c, b, g0, n, mirror && g0 + n == c.rows);
            raw_launch(gpu, env, which, false, [n as u32, 1, 1], &mut blob)?;
            g0 += n;
        }
        gpu.hip.device_synchronize().map_err(err)
    }

    fn read_out(gpu: &Gpu, b: &Bufs, c: &Call, scores: Vec<u8>, persisted: Option<bool>) -> Result<Out> {
        let sel_len = c.rows * c.capacity * 4;
        let mir_len = c.capacity * 4;
        let mut sel = download(gpu, &b.selected)?;
        let mut mirror = download(gpu, &b.mirror)?;
        let canary = sel[sel_len..].iter().all(|&x| x == POISON) && mirror[mir_len..].iter().all(|&x| x == POISON);
        sel.truncate(sel_len);
        mirror.truncate(mir_len);
        Ok(Out { sel, mirror, scores, canary, persisted })
    }

    fn arm_raw(gpu: &mut Gpu, env: &Env, which: RawImpl, case: &Case, b: &Bufs) -> Result<Out> {
        poison_outputs(gpu, b)?;
        let mut scores = Vec::new();
        raw_call(gpu, env, which, &case.c, b, case.mirror, Some(&mut scores))?;
        read_out(gpu, b, &case.c, scores, None)
    }

    fn params<'a>(case: &'a Case, b: &'a Bufs) -> IndexedAttentionSelectBatch<'a> {
        let c = &case.c;
        IndexedAttentionSelectBatch {
            query: &b.query, pooled: &b.pooled, selected: &b.selected, rows: c.rows, query_row_stride: c.stride,
            block_count: c.block_count, index_heads: 4, index_dim: c.dim, budget_blocks: c.budget, compress: c.compress,
            position_start: c.position_start, capacity: c.capacity, shape_blocks: case.shape_blocks,
        }
    }

    /// `arm` Some: the lab helper with that arm; None: the live route.
    fn arm_helper(gpu: &mut Gpu, case: &Case, b: &Bufs, arm: Option<QsaSelectPmArm>) -> Result<Out> {
        poison_outputs(gpu, b)?;
        let p = params(case, b);
        let persisted = match arm {
            Some(a) => indexed_attention_select_batch_pm_arm(gpu, &p, case.mirror.then_some(&b.mirror), a).map_err(err)?,
            None if case.mirror => indexed_attention_select_batch_mirrored(gpu, &p, &b.mirror).map_err(err)?,
            None => { indexed_attention_select_batch(gpu, &p).map_err(err)?; false }
        };
        gpu.hip.device_synchronize().map_err(err)?;
        read_out(gpu, b, &case.c, Vec::new(), Some(persisted))
    }

    fn twice<F: FnMut(&mut Gpu) -> Result<Out>>(gpu: &mut Gpu, mut f: F) -> Result<(Out, Out)> {
        let a = f(gpu)?;
        let b = f(gpu)?;
        Ok((a, b))
    }

    // -------------------------------------------------------------- graph

    struct GraphRun { stream: Stream, graph: Graph, exec: GraphExec, _blobs: Vec<Vec<u8>> }

    /// Capture the PM pair's raw launches (score+select per group, or select
    /// only for injected scores) on a private non-blocking stream.  All kernarg
    /// blobs are built before capture and owned by the result; nothing but
    /// kernel launches is issued inside the capture window.
    fn build_graph(gpu: &mut Gpu, env: &Env, c: &Call, b: &Bufs, mirror: bool, with_score: bool) -> Result<GraphRun> {
        let group = c.group();
        let mut items: Vec<(&Function, [u32; 3], Vec<u8>)> = Vec::new();
        let mut g0 = 0;
        while g0 < c.rows {
            let n = group.min(c.rows - g0);
            if with_score {
                items.push((&env.pm_score, [c.block_count.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], score_blob(c, b, g0, n).into_vec()));
            }
            items.push((&env.pm_select, [n as u32, 1, 1], select_blob(c, b, g0, n, mirror && g0 + n == c.rows).into_vec()));
            g0 += n;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        let stream = gpu.hip.stream_create_non_blocking().map_err(err)?;
        gpu.hip.stream_begin_capture(&stream, 0).map_err(err)?;
        let mut launched = Ok(());
        for (f, grid, blob) in items.iter_mut() {
            launched = unsafe { gpu.hip.launch_kernel_blob(f, *grid, [256, 1, 1], 0, Some(&stream), blob.as_mut_slice()) };
            if launched.is_err() { break }
        }
        let graph = gpu.hip.stream_end_capture(&stream);
        launched.map_err(err)?;
        let graph = graph.map_err(err)?;
        let exec = gpu.hip.graph_instantiate(&graph).map_err(err)?;
        Ok(GraphRun { stream, graph, exec, _blobs: items.into_iter().map(|x| x.2).collect() })
    }

    fn replay(gpu: &Gpu, g: &GraphRun) -> Result<()> {
        gpu.hip.device_synchronize().map_err(err)?;
        gpu.hip.graph_launch(&g.exec, &g.stream).map_err(err)?;
        gpu.hip.stream_synchronize(&g.stream).map_err(err)
    }

    fn free_graph(gpu: &Gpu, g: GraphRun) -> Result<()> {
        gpu.hip.stream_synchronize(&g.stream).map_err(err)?;
        gpu.hip.graph_exec_destroy(g.exec).map_err(err)?;
        gpu.hip.graph_destroy(g.graph).map_err(err)?;
        gpu.hip.stream_destroy(g.stream).map_err(err)
    }

    // ------------------------------------------------------------- oracle

    /// CPU stable-order selection of every row over the raw score bytes.
    fn oracle(c: &Call, scores: &[u8]) -> Vec<u8> {
        let bc = c.block_count;
        let rows: Vec<Vec<u8>> = (0..c.rows).into_par_iter().map(|r| {
            let keys: Vec<u32> = scores[r * bc * 4..(r * bc + c.row_blocks(r)) * 4]
                .chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
            selection_row(&rank(&keys), c, r).iter().flat_map(|v| v.to_le_bytes()).collect()
        }).collect();
        rows.concat()
    }

    /// Sampled rows rescored on the CPU (per-head serial FMA chain): number
    /// of causal score words that differ from the GPU's.
    fn rescore_diff(c: &Call, q: &[f32], p: &[f32], scores: &[u8]) -> u64 {
        let bc = c.block_count;
        (0..c.rows).into_par_iter().filter(|r| r % 61 == 0 || *r == c.rows - 1).map(|r| {
            let qr = &q[r * c.stride..r * c.stride + 4 * c.dim];
            (0..c.row_blocks(r)).filter(|&b| {
                let (s, _) = cpu_score(qr, &p[b * c.dim..(b + 1) * c.dim], c.dim);
                let o = (r * bc + b) * 4;
                s.to_bits() != u32::from_le_bytes(scores[o..o + 4].try_into().unwrap())
            }).count() as u64
        }).sum()
    }

    fn judge(name: &str, arm: &str, e: &Expect, o1: &Out, o2: &Out, inputs_ok: bool, extra: Vec<(&str, Value)>, extra_ok: bool) -> (Value, bool) {
        let eq_oracle = o1.sel == e.sel;
        let eq_snap = e.snap.map(|s| o1.sel == s.0 && o1.mirror == s.1);
        let mirror_ok = (if e.mirror { o1.mirror == e.mirror_bytes } else { o1.mirror.iter().all(|&x| x == POISON) })
            && o1.persisted.is_none_or(|p| p == e.mirror);
        let repeat = same(o1, o2);
        let mut rec = json!({"case":name,"arm":arm,"bytes_equal_oracle":eq_oracle,"mirror_ok":mirror_ok,
            "canary_ok":o1.canary,"inputs_unchanged":inputs_ok,"repeat_identical":repeat});
        if let Some(s) = eq_snap { rec["bytes_equal_snapshot"] = json!(s); }
        for (k, v) in extra { rec[k] = v; }
        let ok = eq_oracle && eq_snap.unwrap_or(true) && mirror_ok && o1.canary && inputs_ok && repeat && extra_ok;
        (rec, ok)
    }

    fn inputs_unchanged(gpu: &Gpu, b: &Bufs, qh: &str, ph: &str) -> Result<bool> {
        Ok(sha256(&download(gpu, &b.query)?) == qh && sha256(&download(gpu, &b.pooled)?) == ph)
    }

    fn alloc_bufs(gpu: &mut Gpu, case: &Case) -> Result<Bufs> {
        let c = &case.c;
        let (q, p) = (bytes_f32(&case.qbytes), bytes_f32(&case.pbytes));
        let scores = c.group() * c.block_count * 4;
        let sel = c.rows * c.capacity * 4 + CANARY;
        let mir = c.capacity * 4 + CANARY;
        Ok(Bufs {
            query: gpu.upload_f32(&q, &[q.len()]).map_err(err)?,
            pooled: gpu.upload_f32(&p, &[p.len()]).map_err(err)?,
            scores: gpu.upload_raw(&vec![0u8; scores], &[scores]).map_err(err)?,
            selected: gpu.upload_raw(&vec![POISON; sel], &[sel]).map_err(err)?,
            mirror: gpu.upload_raw(&vec![POISON; mir], &[mir]).map_err(err)?,
        })
    }

    fn free_bufs(gpu: &mut Gpu, b: Bufs) -> Result<()> {
        for t in [b.query, b.pooled, b.scores, b.selected, b.mirror] { gpu.free_tensor(t).map_err(err)?; }
        Ok(())
    }

    /// Whether the builder pair covers the call (mirrors the route's gate).
    fn covered(c: &Call) -> bool {
        let below = |n: Option<usize>| n.is_some_and(|x| x < OOB);
        c.rows >= 512 && c.budget <= 512 && c.block_count > 0 && c.block_count <= PM_MAX_BLOCKS
            && below(c.rows.checked_mul(c.stride).and_then(|n| n.checked_mul(4)))
            && below(c.block_count.checked_mul(128 * 4))
            && below(c.rows.checked_mul(c.capacity).and_then(|n| n.checked_mul(4)))
    }

    // -------------------------------------------------------------- cases

    fn run_case(gpu: &mut Gpu, env: &Env, sink: &mut Sink, case: &Case) -> Result<()> {
        let c = &case.c;
        sink.cases += 1;
        if !covered(c) {
            sink.put(json!({"case":case.name,"arm":"*","refused":"call not covered by the PM pair / production helper (rows < 512, budget > 512 or extent)"}), false)?;
            return Ok(());
        }
        let b = alloc_bufs(gpu, case)?;
        let (q, p) = (bytes_f32(&case.qbytes), bytes_f32(&case.pbytes));
        eprintln!("case {} rows={} bc={} group={} compress={} budget={} capacity={} mirror={}", case.name, c.rows, c.block_count,
            c.group(), c.compress, c.budget, c.capacity, case.mirror);

        // raw-fixed: the oracle's scores.
        let (fx1, fx2) = twice(gpu, |g| arm_raw(g, env, RawImpl::Fixed, case, &b))?;
        let orc = oracle(c, &fx1.scores);
        let orc_mirror = orc[(c.rows - 1) * c.capacity * 4..].to_vec();
        let rescore = rescore_diff(c, &q, &p, &fx1.scores);
        let exp = Expect { mirror: case.mirror, sel: &orc, mirror_bytes: &orc_mirror, snap: case.snap.as_ref() };
        let oracle_snap = case.snap.as_ref().map(|s| orc == s.0 && orc_mirror == s.1);
        let ok_in = inputs_unchanged(gpu, &b, &case.qh, &case.ph)?;
        let (rec, ok) = judge(&case.name, "raw-fixed", &exp, &fx1, &fx2, ok_in,
            vec![("cpu_rescore_bits_diff", json!(rescore)), ("oracle_equal_snapshot", json!(oracle_snap))],
            rescore == 0 && oracle_snap.unwrap_or(true));
        sink.put(rec, ok)?;

        // raw-pm and raw beta: same score bits (incl. untouched poison) and bytes.
        for (which, arm) in [(RawImpl::Pm, "raw-pm"), (RawImpl::Beta, "beta")] {
            if which == RawImpl::Beta {
                if let Some(reason) = case.beta_skip {
                    sink.put(json!({"case":case.name,"arm":"beta","skipped":reason}), true)?;
                    continue;
                }
            }
            let (o1, o2) = twice(gpu, |g| arm_raw(g, env, which, case, &b))?;
            let diff = word_diff(&o1.scores, &fx1.scores);
            let ok_in = inputs_unchanged(gpu, &b, &case.qh, &case.ph)?;
            let (mut rec, mut ok) = judge(&case.name, arm, &exp, &o1, &o2, ok_in, vec![("score_bits_diff", json!(diff))], diff == 0);
            if which == RawImpl::Beta { rec["beta_src"] = json!("pre-fix hipcc pair"); }
            ok &= o1.scores.len() == fx1.scores.len();
            sink.put(rec, ok)?;
        }

        // helper arms (production grouping / scratch / mirror persistence).
        let mut eager_pm: Option<Out> = None;
        for (arm, name) in [
            (QsaSelectPmArm::Hipcc, "fixed"), (QsaSelectPmArm::Both, "pm"),
            (QsaSelectPmArm::ScoreOnly, "pm-score-only"), (QsaSelectPmArm::SelectOnly, "pm-select-only"),
        ] {
            let (o1, o2) = twice(gpu, |g| arm_helper(g, case, &b, Some(arm)))?;
            let ok_in = inputs_unchanged(gpu, &b, &case.qh, &case.ph)?;
            let (rec, ok) = judge(&case.name, name, &exp, &o1, &o2, ok_in, Vec::new(), true);
            sink.put(rec, ok)?;
            if arm == QsaSelectPmArm::Both { eager_pm = Some(o1); }
        }
        let eager_pm = eager_pm.unwrap();

        // pm-graph: captured raw PM launches, replayed twice.
        {
            let g = build_graph(gpu, env, c, &b, case.mirror, true)?;
            let mut outs = Vec::new();
            for _ in 0..2 {
                poison_outputs(gpu, &b)?;
                gpu.hip.memset(&b.scores.buf, POISON as i32, b.scores.byte_size()).map_err(err)?;
                replay(gpu, &g)?;
                outs.push(read_out(gpu, &b, c, Vec::new(), None)?);
            }
            free_graph(gpu, g)?;
            let o2 = outs.pop().unwrap();
            let o1 = outs.pop().unwrap();
            let ok_in = inputs_unchanged(gpu, &b, &case.qh, &case.ph)?;
            let eq_eager = o1.sel == eager_pm.sel && o1.mirror == eager_pm.mirror;
            let (rec, ok) = judge(&case.name, "pm-graph", &exp, &o1, &o2, ok_in, vec![("equal_eager_pm", json!(eq_eager))], eq_eager);
            sink.put(rec, ok)?;
        }

        // live route, default flags: bytes of the pm arm.
        {
            let (o1, o2) = twice(gpu, |g| arm_helper(g, case, &b, None))?;
            let ok_in = inputs_unchanged(gpu, &b, &case.qh, &case.ph)?;
            let eq_pm = o1.sel == eager_pm.sel && o1.mirror == eager_pm.mirror;
            let (rec, ok) = judge(&case.name, "live", &exp, &o1, &o2, ok_in, vec![("equal_pm", json!(eq_pm))], eq_pm);
            sink.put(rec, ok)?;
        }
        free_bufs(gpu, b)
    }

    fn load_snapshot(path: &Path) -> Result<Case> {
        let root = path.parent().unwrap();
        let header: Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if header["schema"] != "qsa-source-v1" { return Err(format!("{}: unsupported schema", path.display())) }
        let sel = &header["identity"]["selector"];
        if header["identity"]["arch"] != "gfx1151" { return Err("snapshot arch not gfx1151".into()) }
        if sel["pooled_dtype"] != "F32" || sel["projection_dtype"] != "F32" { return Err(format!("{}: not the F32 pooled route", path.display())) }
        if sel["heads"] != 4 || sel["dim"] != 128 { return Err(format!("{}: uncovered selector geometry", path.display())) }
        let get = |name: &str| -> Result<Vec<u8>> {
            let b = std::fs::read(root.join(name)).map_err(err)?;
            if header["files"][name]["sha256"] != sha256(&b) { return Err(format!("{}: hash mismatch {name}", path.display())) }
            Ok(b)
        };
        let g = &header["geometry"];
        let n = |v: &Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("missing {k}"));
        let (rows, position_start, compress) = (n(g, "rows")?, n(g, "position_start")?, n(g, "compress")?);
        if compress == 0 { return Err(format!("{}: zero compress", path.display())) }
        let c = Call { rows, stride: n(sel, "projection_stride")?, block_count: (position_start + rows) / compress, dim: 128,
            compress, position_start, budget: n(g, "budget_blocks")?, capacity: n(g, "capacity")? };
        if c.budget > 512 || c.block_count == 0 { return Err("uncovered budget/block count".into()) }
        let pooled_capacity = n(sel, "pooled_capacity")?;
        let qbytes = get("index-projection.f32")?;
        let pbytes = get("pooled.source")?;
        let snap_sel = get("selected.i32")?;
        let snap_mirror = get("selected-mirror.i32")?;
        if qbytes.len() != rows * c.stride * 4 || pbytes.len() != pooled_capacity * 128 * 4 || snap_sel.len() != rows * c.capacity * 4
            || snap_mirror.len() != c.capacity * 4 || c.block_count > pooled_capacity {
            return Err(format!("{}: geometry/extent mismatch", path.display()))
        }
        let name = format!("{} ctx={} layer={}", path.display(), header["identity"]["ctx"], header["identity"]["layer"]);
        Ok(Case { name, c, shape_blocks: pooled_capacity, qh: sha256(&qbytes), ph: sha256(&pbytes), qbytes, pbytes,
            snap: Some((snap_sel, snap_mirror)), mirror: true,
            beta_skip: (c.block_count >= BETA_MAX_BLOCKS).then_some("block_count >= 32768: pre-fix tie-count overflow range, beta never runs") })
    }

    // --------------------------------------------------------- synthetic

    #[derive(Clone, Copy)]
    enum Tie { Dense, Sparse, Dup, Float, Zero, Same }

    struct Spec { name: &'static str, compress: usize, bc: usize, rows: usize, tail: usize, budget: usize, cap_extra: usize, mirror: bool, tie: Tie }

    fn int(r: &mut Rng) -> f32 { (r.next() % 3) as f32 - 1.0 }
    fn flt(r: &mut Rng) -> f32 { (r.next() >> 40) as f32 / 8_388_608.0 - 1.0 }

    /// Deterministic case.  Query/pooled values are small integers (exact,
    /// order-independent dots: dense score ties) or floats; `Dup` draws every
    /// block from 40 key vectors (exact equal-threshold ties).  `tail` is the
    /// last row's partial block, `position_start = bc*compress + tail - rows`.
    fn synth_case(idx: usize, s: &Spec) -> Case {
        let (compress, bc, rows) = (s.compress, s.bc, s.rows);
        assert!(s.tail < compress && bc * compress + s.tail >= rows, "{}", s.name);
        let position_start = bc * compress + s.tail - rows;
        let capacity = s.budget * compress + compress - 1 + s.cap_extra;
        let stride = 640usize;
        let shape_blocks = bc + 7;
        let mut rng = Rng(0x5eed_0000_0000 + idx as u64);
        let float_mode = matches!(s.tie, Tie::Float);
        let mut q = Vec::with_capacity(rows * stride);
        for _ in 0..rows {
            for d in 0..stride { q.push(if d < 512 && !float_mode { int(&mut rng) } else { flt(&mut rng) }); }
        }
        let mut p = Vec::with_capacity(shape_blocks * 128);
        match s.tie {
            Tie::Dense => for _ in 0..bc * 128 { p.push(int(&mut rng)); },
            Tie::Float => for _ in 0..bc * 128 { p.push(flt(&mut rng)); },
            Tie::Sparse => for _ in 0..bc * 128 {
                let v = rng.next();
                p.push(if v % 16 == 0 { if v & 16 == 0 { 1.0 } else { -1.0 } } else { 0.0 });
            },
            Tie::Zero => p.resize(bc * 128, 0.0),
            Tie::Dup => {
                let keys: Vec<Vec<f32>> = (0..40).map(|_| (0..128).map(|_| int(&mut rng)).collect()).collect();
                for _ in 0..bc { let k = &keys[(rng.next() % 40) as usize]; p.extend_from_slice(k); }
            }
            Tie::Same => {
                let k: Vec<f32> = (0..128).map(|_| int(&mut rng)).collect();
                for _ in 0..bc { p.extend_from_slice(&k); }
            }
        }
        // Blocks past the active count are never read: garbage the hash guards.
        for _ in 0..(shape_blocks - bc) * 128 { p.push(flt(&mut rng)); }
        let (qbytes, pbytes) = (f32_bytes(&q), f32_bytes(&p));
        let c = Call { rows, stride, block_count: bc, dim: 128, compress, position_start, budget: s.budget, capacity };
        Case { name: format!("synthetic:{}", s.name), c, shape_blocks, qh: sha256(&qbytes), ph: sha256(&pbytes), qbytes, pbytes, snap: None, mirror: s.mirror,
            beta_skip: (bc >= BETA_MAX_BLOCKS).then_some("block_count >= 32768: pre-fix tie-count overflow range, beta never runs") }
    }

    fn specs() -> Vec<Spec> {
        use Tie::*;
        let s = |name, compress, bc, rows, tail, budget, cap_extra, mirror, tie| Spec { name, compress, bc, rows, tail, budget, cap_extra, mirror, tie };
        vec![
            // position % 4 residues (every row also walks all residues)
            s("c4-res0", 4, 700, 512, 0, 512, 0, true, Dense),
            s("c4-res1", 4, 700, 512, 1, 512, 0, false, Dense),
            s("c4-res2", 4, 700, 512, 2, 512, 37, true, Sparse),
            s("c4-res3", 4, 700, 512, 3, 512, 0, false, Dup),
            // visible blocks fewer / equal / more than the budget
            s("c4-fewer-than-budget", 4, 128, 512, 0, 512, 0, true, Dense),
            s("c4-equal-budget", 4, 512, 512, 1, 512, 0, true, Float),
            s("c4-more-than-budget", 4, 1024, 1024, 0, 512, 0, false, Sparse),
            // budgets 0 / 1 / 511
            s("c4-budget0", 4, 600, 600, 2, 0, 0, true, Dense),
            s("c4-budget1", 4, 2000, 700, 3, 1, 5, true, Dup),
            s("c4-budget511", 4, 2100, 2100, 1, 511, 0, true, Dense),
            // all-zero and identical keys (everything tied)
            s("c4-zero-keys", 4, 4096, 640, 3, 512, 0, true, Zero),
            s("c4-same-keys", 4, 4096, 640, 2, 512, 0, true, Same),
            // ragged groups (64 MiB scratch: group = 16 MiB / (4 bc) rounded to 16)
            s("c4-ragged-1027-g1024+3", 4, 16384, 1027, 3, 512, 0, true, Dense),
            s("c4-ragged-2020-g1024+996", 4, 16384, 2020, 0, 512, 0, true, Float),
            s("c4-ragged-2051-g2048+3", 4, 8192, 2051, 1, 511, 0, false, Dup),
            s("c4-ragged-1019-single-group", 4, 8192, 1019, 2, 512, 0, true, Sparse),
            s("c4-ragged-2100-g1392+708", 4, 12000, 2100, 3, 1, 0, true, Dense),
            s("c4-ragged-1668-g832x2+4", 4, 20000, 1668, 3, 512, 0, true, Dup),
            // > 32768 blocks (beta never runs)
            s("c4-bc40000-g416x2+3", 4, 40000, 835, 1, 512, 0, true, Float),
            s("c4-bc65536-g64x8+3", 4, 65536, 515, 3, 512, 0, true, Dup),
            // compress 128
            s("c128-res0", 128, 16, 512, 0, 32, 0, true, Dense),
            s("c128-res1", 128, 16, 512, 1, 32, 0, false, Dense),
            s("c128-res63", 128, 16, 512, 63, 32, 3, true, Float),
            s("c128-res127-fewer-than-budget", 128, 16, 512, 127, 512, 0, true, Dense),
            s("c128-more-than-budget", 128, 600, 512, 64, 512, 0, true, Sparse),
            s("c128-budget0", 128, 64, 520, 9, 0, 0, true, Dense),
            s("c128-ragged-1027-g1024+3", 128, 16384, 1027, 5, 64, 0, true, Dup),
        ]
    }

    // ----------------------------------------------------------- all-tied

    /// Scores injected directly (every score equal): select-only raw
    /// launches of the fixed hipcc / embedded PM / (16384 only) pre-fix hipcc
    /// select symbol, and the PM select captured into a graph.
    fn run_tied(gpu: &mut Gpu, env: &Env, sink: &mut Sink, blocks: usize, value: f32) -> Result<()> {
        sink.cases += 1;
        let (compress, rows) = (4usize, 3usize);
        // Every row sees exactly `blocks` complete blocks plus a 1..3-token tail.
        let position_start = blocks * compress + 3 - rows;
        let c = Call { rows, stride: 0, block_count: blocks, dim: 128, compress, position_start, budget: 512, capacity: 512 * compress + compress - 1 };
        let name = format!("tied:blocks={blocks}:value={value}");
        let fill: Vec<u8> = (0..rows * blocks).flat_map(|_| value.to_le_bytes()).collect();
        let sel_len = rows * c.capacity * 4;
        let (sel_b, mir_b) = (sel_len + CANARY, c.capacity * 4 + CANARY);
        let b = Bufs {
            query: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
            pooled: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
            scores: gpu.upload_raw(&fill, &[fill.len()]).map_err(err)?,
            selected: gpu.upload_raw(&vec![POISON; sel_b], &[sel_b]).map_err(err)?,
            mirror: gpu.upload_raw(&vec![POISON; mir_b], &[mir_b]).map_err(err)?,
        };
        let orc = oracle(&c, &fill);
        let orc_mirror = orc[(rows - 1) * c.capacity * 4..].to_vec();
        let exp = Expect { mirror: true, sel: &orc, mirror_bytes: &orc_mirror, snap: None };
        let fill_hash = sha256(&fill);
        let mut eager_pm: Option<Out> = None;
        let mut arms = vec![("fixed", Some(RawImpl::Fixed)), ("pm", Some(RawImpl::Pm))];
        if blocks == 16384 { arms.push(("beta", Some(RawImpl::Beta))); } else {
            sink.put(json!({"case":name,"arm":"beta","skipped":"block_count >= 32768: pre-fix tie-count overflow range, beta never runs"}), true)?;
        }
        arms.push(("pm-graph", None));
        for (arm, which) in arms {
            let one = |g: &mut Gpu, graph: Option<&GraphRun>| -> Result<Out> {
                poison_outputs(g, &b)?;
                match (which, graph) {
                    (Some(w), _) => {
                        let mut blob = select_blob(&c, &b, 0, rows, true);
                        raw_launch(g, env, w, false, [rows as u32, 1, 1], &mut blob)?;
                        g.hip.device_synchronize().map_err(err)?;
                    }
                    (None, Some(gr)) => replay(g, gr)?,
                    (None, None) => return Err("graph arm without a graph".into()),
                }
                read_out(g, &b, &c, Vec::new(), None)
            };
            let (o1, o2, graph_run) = if which.is_none() {
                let gr = build_graph(gpu, env, &c, &b, true, false)?;
                let r = (one(gpu, Some(&gr))?, one(gpu, Some(&gr))?);
                free_graph(gpu, gr)?;
                (r.0, r.1, true)
            } else {
                (one(gpu, None)?, one(gpu, None)?, false)
            };
            let ok_in = sha256(&download(gpu, &b.scores)?) == fill_hash;
            let mut extra = vec![("blocks", json!(blocks)), ("value", json!(value)), ("rows", json!(rows)), ("position_start", json!(position_start))];
            let mut extra_ok = true;
            if graph_run {
                let eq = eager_pm.as_ref().is_some_and(|p| p.sel == o1.sel && p.mirror == o1.mirror);
                extra.push(("equal_eager_pm", json!(eq)));
                extra_ok = eq;
            }
            let (rec, ok) = judge(&name, arm, &exp, &o1, &o2, ok_in, extra, extra_ok);
            sink.put(rec, ok)?;
            if arm == "pm" { eager_pm = Some(o1); }
        }
        free_bufs(gpu, b)
    }

    // --------------------------------------------------------------- run

    fn load_env(gpu: &mut Gpu, beta_src: &Path) -> Result<Env> {
        if !beta_src.is_absolute() { return Err("--beta-src must be absolute".into()) }
        let src = std::fs::read_to_string(beta_src).map_err(err)?;
        if !src.contains(HIPCC_SCORE) || !src.contains(HIPCC_SELECT) { return Err("--beta-src lacks the QSA select symbols".into()) }
        if src == TENSOR_OPS_SRC { return Err("--beta-src equals the in-tree (fixed) source; the beta arm must be the pre-fix pair".into()) }
        let mut compiler = rdna_compute::KernelCompiler::new(&gpu.arch, gpu.flags.hipcc_extra_flags.clone()).map_err(err)?;
        let obj = compiler.compile_for_symbol(BETA_MODULE, &src, HIPCC_SCORE).map_err(err)?.to_path_buf();
        let beta = gpu.hip.module_load(obj.to_str().ok_or("non-utf8 beta object path")?).map_err(err)?;
        let beta_score = gpu.hip.module_get_function(&beta, HIPCC_SCORE).map_err(err)?;
        let beta_select = gpu.hip.module_get_function(&beta, HIPCC_SELECT).map_err(err)?;
        eprintln!("beta src {} sha256 {} object {}", beta_src.display(), sha256(src.as_bytes()), obj.display());
        let pm = gpu.hip.module_load_data(PM_IMAGE).map_err(err)?;
        let pm_score = gpu.hip.module_get_function(&pm, PM_SCORE).map_err(err)?;
        let pm_select = gpu.hip.module_get_function(&pm, PM_SELECT).map_err(err)?;
        eprintln!("embedded pm image sha256 {}", sha256(PM_IMAGE));
        for k in [HIPCC_SCORE, HIPCC_SELECT] { gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, k).map_err(err)?; }
        Ok(Env { _beta: beta, beta_score, beta_select, _pm: pm, pm_score, pm_select })
    }

    pub fn run(dirs: &[PathBuf], out: &Path, beta_src: Option<PathBuf>, pm_image_given: bool) -> Result<()> {
        if pm_image_given { return Err("--pm-image is not used by --g3 (the embedded certified image is tested)".into()) }
        let beta_src = beta_src.ok_or("--g3 requires --beta-src ABS_FILE (mq_fwht256.h + tensor_ops.hip of 060cadcd3b)")?;
        for v in ["HIPFIRE_QWEN4_QSA_SCORE_PM", "HIPFIRE_QWEN4_QSA_SELECT_PM", "HIPFIRE_QWEN4_QSA_SELECT_EXACT"] {
            if std::env::var_os(v).is_some() { return Err(format!("{v} is set: --g3 checks the live route with default flags") ) }
        }
        let mut paths = Vec::new();
        for d in dirs { snapshots(d, &mut paths)?; }
        if paths.is_empty() { return Err("no snapshot.json found".into()) }
        let mut gpu = Gpu::init().map_err(err)?;
        if gpu.arch != "gfx1151" { return Err(format!("gfx1151 only, got {}", gpu.arch)) }
        let env = load_env(&mut gpu, &beta_src)?;
        let mut sink = Sink { f: std::fs::File::create(out).map_err(err)?, failures: 0, records: 0, cases: 0 };

        for path in &paths {
            let case = load_snapshot(path)?;
            run_case(&mut gpu, &env, &mut sink, &case)?;
        }
        for (i, s) in specs().iter().enumerate() {
            let case = synth_case(i, s);
            run_case(&mut gpu, &env, &mut sink, &case)?;
        }
        for blocks in [16384usize, 32768, 65536, 70000] {
            for value in [1.0f32, 0.0] { run_tied(&mut gpu, &env, &mut sink, blocks, value)?; }
        }
        let summary = json!({"summary":true,"cases":sink.cases,"records":sink.records,"failures":sink.failures,"pass":sink.failures == 0});
        eprintln!("{summary}");
        writeln!(sink.f, "{summary}").map_err(err)?;
        sink.f.flush().map_err(err)?;
        if sink.failures > 0 { return Err(format!("{} failing G3 records; see {}", sink.failures, out.display())) }
        Ok(())
    }
}

// ------------------------------------------------------------- G3 BF16

/// `--g3-bf16 [SNAPSHOT_DIR...] --out OUT.jsonl`: exact-oracle + HIP-graph gate
/// of the BF16 production selector (`QsaKvFormat::index_dtype` BF16 pooled
/// keys, F32 query).  Needs `--features lab` (the explicit-arm helper is
/// lab-only):
/// `cargo build --release -p rdna-compute --features lab --example qsa_select_pm_check`.
///
/// Cases: every BF16 `qsa-source-v1` snapshot found under the directories
/// (hash-checked `selected.i32`, `selected-mirror.i32`, raw BF16
/// `pooled.source`, F32 `index-projection.f32`; gfx1151, pooled BF16,
/// projection F32, heads 4 / dim 128), then synthetic all-tied fixtures (every
/// key `0x0000` or `0x3f80`, query 1.0; 16384/32768/65536 blocks, 512 rows,
/// compress 4, budget 512, capacity 2051, stride 512, `position_start =
/// blocks * 4 + 3 - rows`; budgets 0/1/511 at 16384 blocks; 128 blocks =
/// fewer visible blocks than the budget).  Synthetic expectations are the CPU
/// `selection_row(&rank(keys))` of every row (tails included); snapshot
/// expectations are the captured bytes.  With no directory the run is
/// synthetic-only.
///
/// Every case runs at `shape_blocks = pooled_capacity` and again at 65536 when
/// that covers the active blocks (also when identical).  One JSONL record per
/// case and shape holds every arm.  Mirrored (`Hipcc`, `ScoreOnly`, `Both`
/// through `indexed_attention_select_batch_pm_arm` with a mirror, and the
/// default `indexed_attention_select_batch_mirrored`): selected and mirror
/// equal the expectation, `persisted` equals the `Hipcc` arm's, canaries
/// intact.  No-mirror (same arms with `None`, and `indexed_attention_select_batch`):
/// selected equals the expectation, the mirror buffer stays poison, `persisted`
/// is null.  Every output payload and its 4096-byte trailing canary is
/// poisoned before each arm; `Both` runs twice and must repeat bit for bit.
/// After all arms the default mirrored production call is captured into a HIP
/// graph (kernels, scratch and buffers warmed by the eager arms; capture on a
/// private non-blocking stream routed through `gpu.active_stream` with
/// `gpu.graphs.capture_mode` set; capture state restored on every path), the
/// instantiated graph is launched and synchronised, and its output must equal
/// the eager arm, the expectation, the canaries and `persisted`.  The
/// captured kernel symbol and launch count are recorded.  Finally query and
/// pooled are downloaded and compared with the originals.  The process
/// environment is neither read nor modified.
#[cfg(feature = "lab")]
mod g3_bf16 {
    use super::*;
    use rdna_compute::tensor_ops::{
        indexed_attention_select_batch, indexed_attention_select_batch_mirrored,
        indexed_attention_select_batch_pm_arm, IndexedAttentionSelectBatch, QsaSelectPmArm,
    };
    use rdna_compute::DType;

    /// Poisoned bytes after `selected` / `mirror`: any stray write shows.
    const CANARY: usize = 4096;
    /// The second shape pass.
    const FIXED_SHAPE: usize = 65536;
    const ARMS: [(&str, Option<QsaSelectPmArm>); 4] = [
        ("hipcc", Some(QsaSelectPmArm::Hipcc)),
        ("score-only", Some(QsaSelectPmArm::ScoreOnly)),
        ("both", Some(QsaSelectPmArm::Both)),
        ("default", None),
    ];

    struct Sink { f: std::fs::File, failures: usize, records: usize, cases: usize }
    impl Sink {
        fn put(&mut self, mut rec: Value, ok: bool) -> Result<()> {
            rec["ok"] = json!(ok);
            if !ok { self.failures += 1; eprintln!("FAIL {rec}"); } else { eprintln!("{rec}"); }
            self.records += 1;
            writeln!(self.f, "{rec}").map_err(err)?;
            self.f.flush().map_err(err)
        }
    }

    struct Case {
        name: String,
        kind: &'static str,
        c: Call,
        pooled_capacity: usize,
        qbytes: Vec<u8>,
        pbytes: Vec<u8>,
        qh: String,
        ph: String,
        /// Expected `selected` rows / final-row mirror (snapshot bytes or CPU oracle).
        sel: Vec<u8>,
        mirror: Vec<u8>,
    }

    /// Production tensors: F32 query, BF16 pooled, Raw selected / mirror with
    /// trailing canaries.
    struct Dev { query: GpuTensor, pooled: GpuTensor, selected: GpuTensor, mirror: GpuTensor }

    struct Out { sel: Vec<u8>, mirror: Vec<u8>, canary: bool, persisted: Option<bool> }
    fn same(a: &Out, b: &Out) -> bool { a.sel == b.sel && a.mirror == b.mirror && a.canary == b.canary && a.persisted == b.persisted }

    fn le_f32(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect() }

    fn alloc_dev(gpu: &mut Gpu, case: &Case) -> Result<Dev> {
        let c = &case.c;
        let q = le_f32(&case.qbytes);
        let query = gpu.upload_f32(&q, &[q.len()]).map_err(err)?;
        let mut pooled = gpu.upload_raw(&case.pbytes, &[case.pbytes.len()]).map_err(err)?;
        pooled.dtype = DType::BF16;
        pooled.shape = vec![case.pbytes.len() / 2];
        let (sel, mir) = (c.rows * c.capacity * 4 + CANARY, c.capacity * 4 + CANARY);
        let selected = gpu.upload_raw(&vec![POISON; sel], &[sel]).map_err(err)?;
        let mirror = gpu.upload_raw(&vec![POISON; mir], &[mir]).map_err(err)?;
        Ok(Dev { query, pooled, selected, mirror })
    }

    fn free_dev(gpu: &mut Gpu, d: Dev) -> Result<()> {
        for t in [d.query, d.pooled, d.selected, d.mirror] { gpu.free_tensor(t).map_err(err)?; }
        Ok(())
    }

    /// Poison the whole selected / mirror allocations (payload and canary).
    fn poison_dev(gpu: &Gpu, d: &Dev) -> Result<()> {
        for t in [&d.selected, &d.mirror] { gpu.hip.memset(&t.buf, POISON as i32, t.byte_size()).map_err(err)?; }
        Ok(())
    }

    fn read_out(gpu: &Gpu, case: &Case, d: &Dev, persisted: Option<bool>) -> Result<Out> {
        let c = &case.c;
        let (sel_len, mir_len) = (c.rows * c.capacity * 4, c.capacity * 4);
        let mut sel = download(gpu, &d.selected)?;
        let mut mirror = download(gpu, &d.mirror)?;
        let canary = sel[sel_len..].iter().all(|&x| x == POISON) && mirror[mir_len..].iter().all(|&x| x == POISON);
        sel.truncate(sel_len);
        mirror.truncate(mir_len);
        Ok(Out { sel, mirror, canary, persisted })
    }

    fn params<'a>(case: &'a Case, d: &'a Dev, shape: usize) -> IndexedAttentionSelectBatch<'a> {
        let c = &case.c;
        IndexedAttentionSelectBatch {
            query: &d.query, pooled: &d.pooled, selected: &d.selected, rows: c.rows, query_row_stride: c.stride,
            block_count: c.block_count, index_heads: 4, index_dim: c.dim, budget_blocks: c.budget, compress: c.compress,
            position_start: c.position_start, capacity: c.capacity, shape_blocks: shape,
        }
    }

    /// One eager arm: `arm` Some = lab helper, None = the public route;
    /// `mirror` selects the mirrored / no-mirror entry.  `persisted` is the
    /// mirrored entry's return, null for no-mirror.
    fn run_arm(gpu: &mut Gpu, case: &Case, d: &Dev, shape: usize, mirror: bool, arm: Option<QsaSelectPmArm>) -> Result<Out> {
        poison_dev(gpu, d)?;
        let p = params(case, d, shape);
        let r = match arm {
            Some(a) => indexed_attention_select_batch_pm_arm(gpu, &p, mirror.then_some(&d.mirror), a).map_err(err)?,
            None if mirror => indexed_attention_select_batch_mirrored(gpu, &p, &d.mirror).map_err(err)?,
            None => { indexed_attention_select_batch(gpu, &p).map_err(err)?; false }
        };
        gpu.hip.device_synchronize().map_err(err)?;
        read_out(gpu, case, d, mirror.then_some(r))
    }

    fn judge(case: &Case, mirror: bool, o: &Out, reference: Option<&Out>, repeat: Option<bool>) -> (Value, bool) {
        let sel_ok = o.sel == case.sel;
        let mirror_ok = if mirror { o.mirror == case.mirror } else { o.mirror.iter().all(|&x| x == POISON) };
        let persisted_ok = o.persisted.is_some() == mirror;
        let eq_hipcc = reference.is_some_and(|r| o.sel == r.sel && o.mirror == r.mirror && o.persisted == r.persisted);
        let repeat_ok = repeat.unwrap_or(true);
        let mut rec = json!({"selected_equal_expected":sel_ok,"persisted":o.persisted,"persisted_kind_ok":persisted_ok,
            "canary_ok":o.canary,"equal_hipcc_arm":eq_hipcc});
        rec[if mirror { "mirror_equal_expected" } else { "mirror_untouched_poison" }] = json!(mirror_ok);
        if let Some(r) = repeat { rec["repeat_identical"] = json!(r); }
        (rec, sel_ok && mirror_ok && persisted_ok && o.canary && eq_hipcc && repeat_ok)
    }

    struct Mode { rec: Value, ok: bool, hipcc: Option<Out>, default: Option<Out> }

    /// All four arms of one entry kind (mirrored or no-mirror).
    fn run_mode(gpu: &mut Gpu, case: &Case, d: &Dev, shape: usize, mirror: bool) -> Mode {
        let mut arms = serde_json::Map::new();
        let (mut ok, mut hipcc, mut default) = (true, None::<Out>, None::<Out>);
        for (name, arm) in ARMS {
            let first = run_arm(gpu, case, d, shape, mirror, arm);
            let second = (arm == Some(QsaSelectPmArm::Both)).then(|| run_arm(gpu, case, d, shape, mirror, arm));
            let (mut rec, arm_ok, out) = match first {
                Err(e) => (json!({"error":e}), false, None),
                Ok(o1) => {
                    let repeat = match &second {
                        None => Ok(None),
                        Some(Ok(o2)) => Ok(Some(same(&o1, o2))),
                        Some(Err(e)) => Err(e.clone()),
                    };
                    match repeat {
                        Err(e) => (json!({"error":format!("second run: {e}")}), false, Some(o1)),
                        Ok(rep) => {
                            let reference = if name == "hipcc" { Some(&o1) } else { hipcc.as_ref() };
                            let (rec, arm_ok) = judge(case, mirror, &o1, reference, rep);
                            (rec, arm_ok, Some(o1))
                        }
                    }
                }
            };
            rec["ok"] = json!(arm_ok);
            ok &= arm_ok;
            arms.insert(name.to_string(), rec);
            match name { "hipcc" => hipcc = out, "default" => default = out, _ => {} }
        }
        Mode { rec: Value::Object(arms), ok, hipcc, default }
    }

    // -------------------------------------------------------------- graph

    /// Production fused launches of one call: 256-row groups in global-score
    /// mode with more than 256 rows, else one.
    fn expected_launches(rows: usize, shape: usize) -> usize {
        let global = shape * 4 + STATIC_LDS_BYTES > LDS_LIMIT_BYTES;
        if global && rows > GLOBAL_ROWS { rows.div_ceil(GLOBAL_ROWS) } else { 1 }
    }

    fn replay_graph(gpu: &Gpu, case: &Case, d: &Dev, stream: &hip_bridge::Stream, persisted: bool) -> Result<Out> {
        poison_dev(gpu, d)?;
        gpu.hip.device_synchronize().map_err(err)?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stream).map_err(err)?;
        gpu.hip.stream_synchronize(stream).map_err(err)?;
        read_out(gpu, case, d, Some(persisted))
    }

    /// Capture the default mirrored production call, launch the instantiated
    /// graph and compare with the eager `default` / `hipcc` arms.  Eager arms of
    /// the same shape have already loaded the kernels and sized every scratch.
    fn run_graph(gpu: &mut Gpu, case: &Case, d: &Dev, shape: usize, eager: &Out, hipcc: &Out) -> Result<(Value, bool)> {
        if gpu.active_stream.is_some() || gpu.graphs.capture_mode { return Err("capture state not clean before graph".into()) }
        poison_dev(gpu, d)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let stream = gpu.hip.stream_create_non_blocking().map_err(err)?;
        if let Err(e) = gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, &stream) {
            gpu.graphs.abort_graph_capture(&gpu.hip, gpu.device_id, &stream);
            let _ = gpu.hip.stream_destroy(stream);
            return Err(format!("begin capture: {}", err(e)));
        }
        gpu.active_stream = Some(stream);
        let called = {
            let p = params(case, d, shape);
            indexed_attention_select_batch_mirrored(gpu, &p, &d.mirror)
        };
        let symbol = gpu.last_launched_kernel().map(str::to_string);
        let blobs = gpu.graphs.capture_blobs.len();
        let stream = gpu.active_stream.take().ok_or("active stream vanished during capture")?;
        let ended = match &called {
            Ok(_) => gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, &stream),
            Err(_) => Ok(()),
        };
        if called.is_err() || ended.is_err() {
            gpu.graphs.abort_graph_capture(&gpu.hip, gpu.device_id, &stream);
            let _ = gpu.hip.stream_destroy(stream);
            return Err(format!("capture failed: call={:?} end={:?}", called.map_err(err), ended.map_err(err)));
        }
        let persisted = called.map_err(err)?;
        let state_restored = !gpu.graphs.capture_mode && gpu.active_stream.is_none();
        let replayed = replay_graph(gpu, case, d, &stream, persisted);
        gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
        let destroyed = gpu.hip.stream_destroy(stream).map_err(err);
        let o = replayed?;
        destroyed?;

        let launches = expected_launches(case.c.rows, shape);
        let (sel_ok, mirror_ok) = (o.sel == case.sel, o.mirror == case.mirror);
        let eq_eager = o.sel == eager.sel && o.mirror == eager.mirror && o.persisted == eager.persisted;
        let persisted_hipcc = o.persisted == hipcc.persisted;
        let symbol_ok = symbol.as_deref() == Some(FUSED_BF16);
        let launches_ok = blobs == launches;
        let ok = sel_ok && mirror_ok && eq_eager && persisted_hipcc && o.canary && symbol_ok && launches_ok && state_restored;
        let rec = json!({"captured_symbol":symbol,"captured_symbol_ok":symbol_ok,"captured_launches":blobs,
            "expected_launches":launches,"captured_launches_ok":launches_ok,"capture_state_restored":state_restored,
            "capture_persisted":persisted,"persisted":o.persisted,"persisted_equal_hipcc_arm":persisted_hipcc,
            "selected_equal_expected":sel_ok,"mirror_equal_expected":mirror_ok,"equal_eager_default":eq_eager,
            "canary_ok":o.canary,"ok":ok});
        Ok((rec, ok))
    }

    // -------------------------------------------------------------- cases

    fn run_pass(gpu: &mut Gpu, sink: &mut Sink, case: &Case, d: &Dev, shape: usize, label: &str) -> Result<()> {
        let c = &case.c;
        eprintln!("case {} {label} shape_blocks={shape} rows={} bc={} compress={} budget={} capacity={}",
            case.name, c.rows, c.block_count, c.compress, c.budget, c.capacity);
        let mut rec = json!({"case":case.name,"kind":case.kind,"pass":label,"shape_blocks":shape,"pooled_capacity":case.pooled_capacity,
            "rows":c.rows,"block_count":c.block_count,"compress":c.compress,"position_start":c.position_start,"budget_blocks":c.budget,
            "capacity":c.capacity,"projection_stride":c.stride,"heads":4,"dim":c.dim,"pooled_dtype":"BF16","projection_dtype":"F32",
            "pooled_dtype_tensor_is_bf16":d.pooled.dtype == DType::BF16,"query_sha256":case.qh,"pooled_sha256":case.ph});
        let mirrored = run_mode(gpu, case, d, shape, true);
        let plain = run_mode(gpu, case, d, shape, false);
        let mut ok = mirrored.ok && plain.ok;
        rec["mirrored"] = mirrored.rec;
        rec["no_mirror"] = plain.rec;
        if let Err(e) = gpu.hip.device_synchronize() { rec["sync_error"] = json!(err(e)); ok = false; }
        match (&mirrored.default, &mirrored.hipcc) {
            (Some(eager), Some(hipcc)) => match run_graph(gpu, case, d, shape, eager, hipcc) {
                Ok((g, g_ok)) => { rec["graph"] = g; ok &= g_ok; }
                Err(e) => { rec["graph"] = json!({"error":e,"ok":false}); ok = false; }
            },
            _ => { rec["graph"] = json!({"skipped":"mirrored hipcc/default arm failed","ok":false}); ok = false; }
        }
        match (download(gpu, &d.query), download(gpu, &d.pooled)) {
            (Ok(q_back), Ok(p_back)) => {
                let inputs_ok = q_back == case.qbytes && p_back == case.pbytes;
                rec["inputs_unchanged"] = json!(inputs_ok);
                ok &= inputs_ok;
            }
            (q, p) => {
                rec["inputs_unchanged"] = json!(false);
                rec["input_download_error"] = json!(format!("query: {:?} pooled: {:?}", q.err(), p.err()));
                ok = false;
            }
        }
        sink.put(rec, ok)
    }

    fn run_case(gpu: &mut Gpu, sink: &mut Sink, case: &Case) -> Result<()> {
        sink.cases += 1;
        let d = match alloc_dev(gpu, case) {
            Ok(d) => d,
            Err(e) => return sink.put(json!({"case":case.name,"kind":case.kind,"arm":"*","error":format!("alloc: {e}")}), false),
        };
        let mut passes = vec![(case.pooled_capacity, "pooled_capacity")];
        if FIXED_SHAPE >= case.c.block_count { passes.push((FIXED_SHAPE, "shape_65536")); }
        for (shape, label) in passes { run_pass(gpu, sink, case, &d, shape, label)?; }
        match free_dev(gpu, d) {
            Ok(()) => Ok(()),
            Err(e) => sink.put(json!({"case":case.name,"kind":case.kind,"arm":"*","error":format!("free: {e}")}), false),
        }
    }

    fn load_snapshot(path: &Path) -> Result<Case> {
        let root = path.parent().unwrap();
        let header: Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if header["schema"] != "qsa-source-v1" { return Err(format!("{}: unsupported schema", path.display())) }
        let sel = &header["identity"]["selector"];
        if header["identity"]["arch"] != "gfx1151" { return Err(format!("{}: snapshot arch not gfx1151", path.display())) }
        if sel["pooled_dtype"] != "BF16" || sel["projection_dtype"] != "F32" { return Err(format!("{}: not the BF16 pooled / F32 projection route", path.display())) }
        if sel["heads"] != 4 || sel["dim"] != 128 { return Err(format!("{}: uncovered selector geometry", path.display())) }
        let get = |name: &str| -> Result<Vec<u8>> {
            let b = std::fs::read(root.join(name)).map_err(err)?;
            if header["files"][name]["sha256"] != sha256(&b) { return Err(format!("{}: hash mismatch {name}", path.display())) }
            Ok(b)
        };
        let g = &header["geometry"];
        let n = |v: &Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("missing {k}"));
        let (rows, position_start, compress) = (n(g, "rows")?, n(g, "position_start")?, n(g, "compress")?);
        if compress == 0 || rows == 0 { return Err(format!("{}: zero compress/rows", path.display())) }
        let c = Call { rows, stride: n(sel, "projection_stride")?, block_count: (position_start + rows) / compress, dim: 128,
            compress, position_start, budget: n(g, "budget_blocks")?, capacity: n(g, "capacity")? };
        if c.budget > 512 || c.block_count == 0 || c.capacity == 0 || c.stride < 4 * c.dim { return Err(format!("{}: uncovered budget/block count/capacity/stride", path.display())) }
        let pooled_capacity = n(sel, "pooled_capacity")?;
        let qbytes = get("index-projection.f32")?;
        let pbytes = get("pooled.source")?;
        let snap_sel = get("selected.i32")?;
        let snap_mirror = get("selected-mirror.i32")?;
        if qbytes.len() != rows * c.stride * 4 || pbytes.len() != pooled_capacity * 128 * 2 || snap_sel.len() != rows * c.capacity * 4
            || snap_mirror.len() != c.capacity * 4 || c.block_count > pooled_capacity {
            return Err(format!("{}: geometry/extent mismatch", path.display()))
        }
        let name = format!("{} ctx={} layer={}", path.display(), header["identity"]["ctx"], header["identity"]["layer"]);
        Ok(Case { name, kind: "snapshot", c, pooled_capacity, qh: sha256(&qbytes), ph: sha256(&pbytes), qbytes, pbytes, sel: snap_sel, mirror: snap_mirror })
    }

    /// All-tied fixture: every BF16 key `vbits`, query 1.0 in every element.
    /// The last row sees `blocks` complete blocks plus a 3-token tail; the
    /// expectation is the CPU stable selection of every row.
    fn synth_case(blocks: usize, vbits: u16, budget: usize, tag: &str) -> Case {
        let (rows, compress, stride, capacity) = (512usize, 4usize, 512usize, 2051usize);
        assert!(blocks * compress + 3 >= rows, "{tag}");
        let position_start = blocks * compress + 3 - rows;
        let c = Call { rows, stride, block_count: blocks, dim: 128, compress, position_start, budget, capacity };
        debug_assert_eq!((position_start + rows) / compress, blocks);
        let value = f32::from_bits((vbits as u32) << 16);
        let qbytes: Vec<u8> = (0..rows * stride).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let pbytes: Vec<u8> = (0..blocks * 128).flat_map(|_| vbits.to_le_bytes()).collect();
        let key_bits = cpu_score(&vec![1.0f32; 4 * c.dim], &vec![value; c.dim], c.dim).0.to_bits();
        let per_row: Vec<Vec<u8>> = (0..rows).into_par_iter().map(|r| {
            let keys = vec![key_bits; c.row_blocks(r)];
            selection_row(&rank(&keys), &c, r).iter().flat_map(|v| v.to_le_bytes()).collect()
        }).collect();
        let sel = per_row.concat();
        let mirror = sel[(rows - 1) * capacity * 4..].to_vec();
        Case { name: format!("synthetic:{tag}:blocks={blocks}:value=0x{vbits:04x}:budget={budget}"), kind: "synthetic", c, pooled_capacity: blocks,
            qh: sha256(&qbytes), ph: sha256(&pbytes), qbytes, pbytes, sel, mirror }
    }

    fn synthetic_cases() -> Vec<Case> {
        let mut v = Vec::new();
        for blocks in [16384usize, 32768, 65536] {
            for vbits in [0x0000u16, 0x3f80] { v.push(synth_case(blocks, vbits, 512, "tied")); }
        }
        for budget in [0usize, 1, 511] {
            for vbits in [0x0000u16, 0x3f80] { v.push(synth_case(16384, vbits, budget, "budget")); }
        }
        // position_start 3: every row sees at most 128 blocks, fewer than the budget.
        for vbits in [0x0000u16, 0x3f80] { v.push(synth_case(128, vbits, 512, "fewer-visible-than-budget")); }
        v
    }

    pub fn run(dirs: &[PathBuf], out: &Path) -> Result<()> {
        let mut paths = Vec::new();
        for d in dirs { snapshots(d, &mut paths)?; }
        if !dirs.is_empty() && paths.is_empty() { return Err("no snapshot.json found".into()) }
        let mut gpu = Gpu::init().map_err(err)?;
        if gpu.arch != "gfx1151" { return Err(format!("gfx1151 only, got {}", gpu.arch)) }
        let mut sink = Sink { f: std::fs::File::create(out).map_err(err)?, failures: 0, records: 0, cases: 0 };
        for path in &paths {
            match load_snapshot(path) {
                Ok(case) => run_case(&mut gpu, &mut sink, &case)?,
                Err(e) => {
                    sink.cases += 1;
                    sink.put(json!({"case":path.display().to_string(),"kind":"snapshot","arm":"*","error":format!("load: {e}")}), false)?;
                }
            }
        }
        for case in synthetic_cases() { run_case(&mut gpu, &mut sink, &case)?; }
        let summary = json!({"summary":true,"mode":"g3-bf16","snapshots":paths.len(),"cases":sink.cases,"records":sink.records,
            "failures":sink.failures,"pass":sink.failures == 0});
        eprintln!("{summary}");
        writeln!(sink.f, "{summary}").map_err(err)?;
        sink.f.flush().map_err(err)?;
        if sink.failures > 0 { return Err(format!("{} failing G3 BF16 records; see {}", sink.failures, out.display())) }
        Ok(())
    }
}
