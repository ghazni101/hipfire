// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exactness of the BF16-pooled QSA block-score kernels
//! (`kernels/src/qsa_bf16_scores.hip`: `indexed_attention_select_scores_rows{8,16}_bf16`).
//!
//! * Raw scores: each kernel, launched directly (grid x wider than the live
//!   block count, ragged row tiles, padded query rows, `score_stride` != block
//!   count), must write the exact bytes of an appended lab kernel that calls the
//!   fused BF16 selector's own `indexed_attention_select_score4_f32<bf16>` for
//!   every visible `(row, block)` cell, leave every other cell and the canary
//!   past the score rows untouched, and write exactly the visible cells.
//!   Rows 1/8/17, 255..65536 blocks, full / partial / clipped (`block_count`
//!   below the visible count) cover.
//! * End to end: `indexed_attention_select_batch_filter_arm` with the hipcc
//!   score (`score_pm = false`, which scores BF16 arenas with these kernels) and
//!   the filter select must write the bytes of the fused BF16 incumbent
//!   (`indexed_attention_select_batch_pm_arm(.., Hipcc)`) -- selection, mirror,
//!   persisted flag, canaries -- for 2048/8192/32768/65536 blocks (8K..256K
//!   tokens at compress 4), rows 1/8/17, heads 4, dim 128, budget 128,
//!   capacity 528, over dense and tie-rich keys, full and partial visibility
//!   (blocks past the visible prefix hold a huge key, so a read past it changes
//!   the selection), with and without a mirror. Forced repair (`cap = 16`) must
//!   report every row repaired; `target = 1` must stay exact. Repeats are
//!   bit-identical.
//! * Split route arms (`indexed_attention_select_batch_bf16_split_arm`):
//!   select-from-scores and the filter trio, eager and in the retained-replay
//!   shape (stride, grids and scratch from `shape_blocks` far above the live
//!   count, one multi-group case), at the production geometry (budget 512,
//!   capacity 2051), against the same fused incumbent.
//! * Recorded replay: the split route recorded under a `ReplayController`
//!   window at a tiny and at a large position, replayed through the recorded-HIP
//!   oracle (`Gpu::replay_recorded_hip_prefix_at`, the identical declared
//!   binding set the retained PM4 route applies) at positions from 0 to the
//!   capacity bound, each compared with the fused incumbent run eagerly at that
//!   position (blocks past the visible prefix are rewritten as huge keys per
//!   position). The tape's kernels and grids must not depend on the capture
//!   position.
//! * Production route (default on for exact gfx1201; `HIPFIRE_QWEN4_QSA_BF16_SPLIT=0`
//!   must force the incumbent): the public mirrored select eager and recorded,
//!   below and above the crossover.
//! * `#[ignore]`d `qsa_bf16_split_crossover_table`: graph-replayed GPU time of
//!   the fused incumbent, split select and split filter.
//!
//! Every GPU test skips without a gfx11+/gfx12 device.
//!
//!   cargo test -p rdna-compute --release --features lab --test qsa_bf16_scores -- --nocapture
//!   HIPFIRE_QWEN4_QSA_BF16_SPLIT=0 cargo test -p rdna-compute --release --features lab --test qsa_bf16_scores -- --nocapture
//!   cargo test -p rdna-compute --release --features lab --test qsa_bf16_scores -- --ignored --nocapture qsa_bf16_split_crossover_table
#![cfg(feature = "lab")]

use hip_bridge::KernargBlob;
use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
use rdna_compute::tensor_ops::{
    indexed_attention_select_batch_bf16_split_arm, indexed_attention_select_batch_filter_arm,
    indexed_attention_select_batch_mirrored, indexed_attention_select_batch_pm_arm,
    reserve_qsa_bf16_split_scratch, IndexedAttentionSelectBatch, QsaBf16SplitArm, QsaFilterKnobs,
    QsaFilterRowStat, QsaSelectPmArm,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const SCORES_SRC: &str = include_str!("../../../kernels/src/qsa_bf16_scores.hip");
const SCORES_MODULE: &str = "qsa_bf16_scores_test";
const ROWS8: &str = "indexed_attention_select_scores_rows8_bf16";
const ROWS16: &str = "indexed_attention_select_scores_rows16_bf16";

const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const ORACLE_MODULE: &str = "qsa_bf16_scores_fused_oracle_test";
const ORACLE: &str = "qsa_bf16_scores_fused_oracle";
/// Appended lab kernel: one thread per `(row, block)` cell, the fused BF16
/// selector's own per-block score (`qsa_select_batched` stores exactly this),
/// same visibility rule and ABI as the score kernels under test.
const ORACLE_KERNEL: &str = r#"
extern "C" __global__ __launch_bounds__(256) void qsa_bf16_scores_fused_oracle(
    const float* query, const __hip_bfloat16* pooled, float* scores, int rows,
    int query_row_stride, int block_count, int index_dim, int compress,
    int position_start, int score_stride) {
    const int row = (int)blockIdx.y;
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (row >= rows || block >= block_count) return;
    if (block >= (position_start + row + 1) / compress) return;
    scores[row * score_stride + block] = indexed_attention_select_score4_f32(
        query + row * query_row_stride, pooled, block, 4, index_dim);
}
"#;

const HEADS: usize = 4;
const DIM: usize = 128;
const COMPRESS: usize = 4;
const QUERY_STRIDE: usize = HEADS * DIM;

/// Selector geometry: `budget` blocks of `COMPRESS` tokens plus the tail.
#[derive(Clone, Copy)]
struct Geometry {
    budget: usize,
    capacity: usize,
}

const SMALL: Geometry = Geometry { budget: 128, capacity: 528 };
/// The production Qwen4 selector geometry.
const PRODUCTION: Geometry = Geometry { budget: 512, capacity: 2051 };
const MAX_ROWS: usize = 17;
const POISON: u8 = 0xa5;
const POISON_WORD: u32 = 0xa5a5_a5a5;
const CANARY: usize = 4096;
/// BF16 bits of 8.0: a block of these scores far above any real key.
const HOT: u16 = 0x4100;

// ---------------------------------------------------------------------------
// helpers

fn gpu_or_skip() -> Option<Gpu> {
    let gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skip: no GPU ({e})");
            return None;
        }
    };
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        eprintln!("skip: needs a gfx11+/gfx12 device (got {})", gpu.arch);
        return None;
    }
    Some(gpu)
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
}

fn mix(x: usize) -> u32 {
    let mut h = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (h >> 32) as u32
}

/// Round-to-nearest-even F32 -> BF16 bits (finite inputs).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn poisoned(gpu: &Gpu, len: usize) -> GpuTensor {
    gpu.upload_raw(&vec![POISON; len], &[len]).expect("poisoned upload")
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .memset(&t.buf, POISON as i32, t.byte_size())
        .expect("poison outputs");
}

fn upload_bf16(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[bits.len()], DType::BF16).expect("alloc bf16");
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("bf16 upload");
    t
}

fn stat_tuples(stats: &[QsaFilterRowStat]) -> Vec<(u64, u32, bool)> {
    stats.iter().map(|s| (s.threshold, s.count, s.repaired)).collect()
}

fn all_poison(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| *b == POISON)
}

// ---------------------------------------------------------------------------
// key generators

#[derive(Clone, Copy, Debug)]
enum Keys {
    /// Random keys, 10% all-negative (ReLU zero score).
    Dense,
    /// Exact score ties: 40% of blocks copy one of four palette rows (two
    /// constants), 10% all-negative, the rest random.
    Ties,
    /// Dense with a per-block power-of-two scale (-3..=4): exercises BF16
    /// exponents and product rounding. Raw-score test only.
    Wide,
}

fn keys_f32(kind: Keys, blocks: usize) -> Vec<f32> {
    let mut rng = Lcg(0x7d3b_91a5 ^ blocks as u32);
    let mut keys = vec![0f32; blocks * DIM];
    match kind {
        Keys::Dense | Keys::Wide => {
            for block in 0..blocks {
                let h = mix(block);
                let scale = if matches!(kind, Keys::Wide) { 2f32.powi(((h >> 4) & 7) as i32 - 3) } else { 1.0 };
                let negative = h % 10 == 0;
                for v in &mut keys[block * DIM..(block + 1) * DIM] {
                    *v = if negative {
                        -(rng.unit() + 0.01) * scale
                    } else {
                        (rng.unit() - 0.25) * scale
                    };
                }
            }
        }
        Keys::Ties => {
            let palette: Vec<Vec<f32>> = (0..4)
                .map(|k| match k {
                    0 => vec![1.0; DIM],
                    1 => vec![0.9; DIM],
                    _ => (0..DIM).map(|_| rng.unit()).collect(),
                })
                .collect();
            for block in 0..blocks {
                let h = mix(block);
                let row = &mut keys[block * DIM..(block + 1) * DIM];
                match h % 10 {
                    0..=3 => row.copy_from_slice(&palette[((h >> 8) % 4) as usize]),
                    4 => row.iter_mut().for_each(|v| *v = -rng.unit() - 0.01),
                    _ => row.iter_mut().for_each(|v| *v = rng.unit() - 0.25),
                }
            }
        }
    }
    keys
}

/// BF16 keys; blocks at or past `hot_from` hold [`HOT`] in every element.
fn keys_bf16(kind: Keys, blocks: usize, hot_from: usize) -> Vec<u16> {
    let mut bits: Vec<u16> = keys_f32(kind, blocks).into_iter().map(bf16_bits).collect();
    if hot_from < blocks {
        bits[hot_from * DIM..].fill(HOT);
    }
    bits
}

#[derive(Clone, Copy, Debug)]
enum Cover {
    /// `(position_start + rows) / compress == blocks`, plus `tail` tokens.
    Full(usize),
    /// Three quarters of the blocks are visible; the rest are [`HOT`].
    Partial(usize),
}

impl Cover {
    fn position_start(self, blocks: usize, rows: usize) -> usize {
        match self {
            Cover::Full(tail) => blocks * COMPRESS + tail - rows,
            Cover::Partial(tail) => blocks * 3 + tail - rows,
        }
    }

    /// First block no row can see (independent of `rows`).
    fn hot_from(self, blocks: usize) -> usize {
        match self {
            Cover::Full(_) => blocks,
            Cover::Partial(_) => (self.position_start(blocks, 1) + 1) / COMPRESS,
        }
    }
}

// ---------------------------------------------------------------------------
// raw scores against the fused per-block score

fn query_padded(rows: usize, stride: usize) -> Vec<f32> {
    let mut rng = Lcg(0x51ca_17b3 ^ (rows * 1031 + stride) as u32);
    // Padding between rows is huge: a read of it corrupts the dot.
    let mut q = vec![3.0e19f32; (rows - 1) * stride + QUERY_STRIDE];
    for r in 0..rows {
        for v in &mut q[r * stride..r * stride + QUERY_STRIDE] {
            *v = (rng.unit() - 0.3) * 1.7;
        }
    }
    q
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Full(usize),
    Partial(usize),
    /// `block_count` below the visible prefix (stale-grid shape).
    Clip,
}

fn launch_scores(
    gpu: &Gpu,
    name: &str,
    grid: [u32; 3],
    query: &GpuTensor,
    pooled: &GpuTensor,
    scores: &GpuTensor,
    ints: [i32; 7],
) {
    let mut args = KernargBlob::new();
    args.push_ptr(query.buf.as_ptr());
    args.push_ptr(pooled.buf.as_ptr());
    args.push_ptr(scores.buf.as_ptr());
    for v in ints {
        args.push_i32(v);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(name, grid, [256, 1, 1], 0, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {name}: {e}"));
    gpu.hip.device_synchronize().expect("sync");
}

#[test]
fn qsa_bf16_score_kernels_match_fused_scores_bit_for_bit() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let mut oracle_src = String::from(TENSOR_OPS_SRC);
    oracle_src.push('\n');
    oracle_src.push_str(ORACLE_KERNEL);
    gpu.ensure_kernel_public(ORACLE_MODULE, &oracle_src, ORACLE).expect("load fused score oracle");
    for name in [ROWS8, ROWS16] {
        gpu.ensure_kernel_public(SCORES_MODULE, SCORES_SRC, name).expect("load bf16 score kernel");
    }
    let mut cells_checked = 0usize;
    let mut cases = 0usize;
    for blocks in [255usize, 257, 2048, 8192, 32768, 65536] {
        for kind in [Keys::Dense, Keys::Ties, Keys::Wide] {
            let pooled = upload_bf16(&mut gpu, &keys_bf16(kind, blocks, blocks));
            for rows in [1usize, 8, 17] {
                for qstride in [QUERY_STRIDE, QUERY_STRIDE + 128] {
                    let q = query_padded(rows, qstride);
                    let query = gpu.upload_f32(&q, &[q.len()]).expect("query upload");
                    for scenario in
                        [Scenario::Full(0), Scenario::Full(3), Scenario::Partial(2), Scenario::Clip]
                    {
                        let (ps, live) = match scenario {
                            Scenario::Full(t) => (Cover::Full(t).position_start(blocks, rows), blocks),
                            Scenario::Partial(t) => (Cover::Partial(t).position_start(blocks, rows), blocks),
                            Scenario::Clip => (Cover::Full(3).position_start(blocks, rows), blocks / 2),
                        };
                        let stride = live + 7;
                        let bytes = rows * stride * 4 + CANARY;
                        let scores = poisoned(&gpu, bytes);
                        let ints = [
                            rows as i32, qstride as i32, live as i32, DIM as i32, COMPRESS as i32,
                            ps as i32, stride as i32,
                        ];
                        let label = format!(
                            "blocks={blocks} keys={kind:?} rows={rows} qstride={qstride} {scenario:?} live={live}"
                        );

                        launch_scores(
                            &gpu, ORACLE, [live.div_ceil(256) as u32, rows as u32, 1], &query,
                            &pooled, &scores, ints,
                        );
                        let want = gpu.download_raw_bytes(&scores).expect("oracle download");
                        let expected_cells: usize =
                            (0..rows).map(|r| live.min((ps + r + 1) / COMPRESS)).sum();
                        let written = want[..rows * stride * 4]
                            .chunks_exact(4)
                            .filter(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) != POISON_WORD)
                            .count();
                        assert_eq!(written, expected_cells, "{label}: oracle wrote the visible cells");
                        assert!(all_poison(&want[rows * stride * 4..]), "{label}: oracle canary");

                        for (name, tile_rows) in [(ROWS8, 8usize), (ROWS16, 16usize)] {
                            // `extra` > 0: the fixed grid is wider than the live count.
                            for extra in [0usize, 700] {
                                poison(&gpu, &scores);
                                launch_scores(
                                    &gpu,
                                    name,
                                    [
                                        (live + extra).div_ceil(256) as u32,
                                        rows.div_ceil(tile_rows) as u32,
                                        1,
                                    ],
                                    &query,
                                    &pooled,
                                    &scores,
                                    ints,
                                );
                                let got = gpu.download_raw_bytes(&scores).expect("scores download");
                                assert!(
                                    got == want,
                                    "{name} extra={extra} {label}: scores differ from the fused per-block score"
                                );
                                cases += 1;
                            }
                        }
                        cells_checked += expected_cells;
                        gpu.free_tensor(scores).expect("free scores");
                    }
                    gpu.free_tensor(query).expect("free query");
                }
            }
            gpu.free_tensor(pooled).expect("free pooled");
        }
    }
    println!("{cases} score launches bit-exact over {cells_checked} cells");
}

// ---------------------------------------------------------------------------
// end to end: BF16 filter arm vs the fused BF16 incumbent

struct Fixture {
    query: GpuTensor,
    selected: GpuTensor,
    mirror: GpuTensor,
    geo: Geometry,
}

impl Fixture {
    fn new(gpu: &mut Gpu, geo: Geometry) -> Self {
        Self::with_max_rows(gpu, geo, MAX_ROWS)
    }

    fn with_max_rows(gpu: &mut Gpu, geo: Geometry, max_rows: usize) -> Self {
        let mut rng = Lcg(0x51ca_17b3);
        // Positive queries: an all-negative block scores 0.
        let query: Vec<f32> = (0..max_rows * QUERY_STRIDE).map(|_| rng.unit()).collect();
        Self {
            query: gpu.upload_f32(&query, &[query.len()]).expect("query upload"),
            selected: poisoned(gpu, max_rows * geo.capacity * 4 + CANARY),
            mirror: poisoned(gpu, geo.capacity * 4 + CANARY),
            geo,
        }
    }

    fn params<'a>(
        &'a self,
        pooled: &'a GpuTensor,
        blocks: usize,
        rows: usize,
        position_start: usize,
    ) -> IndexedAttentionSelectBatch<'a> {
        self.params_shaped(pooled, blocks, blocks, rows, position_start)
    }

    /// `shape_blocks` above `blocks`: the pooling capacity of a retained tape.
    fn params_shaped<'a>(
        &'a self,
        pooled: &'a GpuTensor,
        blocks: usize,
        shape_blocks: usize,
        rows: usize,
        position_start: usize,
    ) -> IndexedAttentionSelectBatch<'a> {
        IndexedAttentionSelectBatch {
            query: &self.query,
            pooled,
            pooled_bf16: None,
            selected: &self.selected,
            rows,
            query_row_stride: QUERY_STRIDE,
            block_count: blocks,
            index_heads: HEADS,
            index_dim: DIM,
            budget_blocks: self.geo.budget,
            compress: COMPRESS,
            position_start,
            capacity: self.geo.capacity,
            shape_blocks,
        }
    }

    fn poison(&self, gpu: &Gpu) {
        poison(gpu, &self.selected);
        poison(gpu, &self.mirror);
    }

    fn download(&self, gpu: &Gpu, persisted: bool, stats: Vec<QsaFilterRowStat>) -> Out {
        gpu.hip.device_synchronize().expect("sync");
        Out {
            persisted,
            selected: gpu.download_raw_bytes(&self.selected).expect("selected download"),
            mirror: gpu.download_raw_bytes(&self.mirror).expect("mirror download"),
            stats,
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.query, self.selected, self.mirror] {
            gpu.free_tensor(t).expect("free");
        }
    }
}

struct Out {
    persisted: bool,
    selected: Vec<u8>,
    mirror: Vec<u8>,
    stats: Vec<QsaFilterRowStat>,
}

fn run_reference(
    gpu: &mut Gpu,
    fx: &Fixture,
    p: &IndexedAttentionSelectBatch<'_>,
    with_mirror: bool,
) -> Out {
    fx.poison(gpu);
    let persisted = indexed_attention_select_batch_pm_arm(
        gpu,
        p,
        with_mirror.then_some(&fx.mirror),
        QsaSelectPmArm::Hipcc,
    )
    .unwrap_or_else(|e| {
        panic!("fused bf16 reference rows={} blocks={} ps={}: {e}", p.rows, p.block_count, p.position_start)
    });
    fx.download(gpu, persisted, Vec::new())
}

fn run_filter(
    gpu: &mut Gpu,
    fx: &Fixture,
    p: &IndexedAttentionSelectBatch<'_>,
    with_mirror: bool,
    score_pm: bool,
    knobs: QsaFilterKnobs,
) -> Out {
    fx.poison(gpu);
    let (persisted, stats) = indexed_attention_select_batch_filter_arm(
        gpu,
        p,
        with_mirror.then_some(&fx.mirror),
        score_pm,
        knobs,
    )
    .unwrap_or_else(|e| {
        panic!(
            "bf16 filter pm_score={score_pm} rows={} blocks={} ps={} mirror={with_mirror}: {e}",
            p.rows, p.block_count, p.position_start
        )
    });
    fx.download(gpu, persisted, stats)
}

fn run_split(
    gpu: &mut Gpu,
    fx: &Fixture,
    p: &IndexedAttentionSelectBatch<'_>,
    with_mirror: bool,
    arm: QsaBf16SplitArm,
    bounded: bool,
) -> Out {
    fx.poison(gpu);
    let collect = matches!(arm, QsaBf16SplitArm::Filter(_));
    let (persisted, stats) = indexed_attention_select_batch_bf16_split_arm(
        gpu,
        p,
        with_mirror.then_some(&fx.mirror),
        arm,
        bounded,
        collect,
    )
    .unwrap_or_else(|e| {
        panic!(
            "bf16 split {arm:?} bounded={bounded} rows={} blocks={} shape={} ps={} mirror={with_mirror}: {e}",
            p.rows, p.block_count, p.shape_blocks, p.position_start
        )
    });
    fx.download(gpu, persisted, stats)
}

/// `out` must be the fused reference's bytes, flag included, with intact
/// canaries and a mirror only when one was given. `stat_rows`: the filter's
/// one-stat-per-row count (`None`: no stats expected).
fn check_against(
    label: &str,
    out: &Out,
    reference: &Out,
    geo: Geometry,
    rows: usize,
    with_mirror: bool,
    stat_rows: Option<usize>,
) {
    let used = rows * geo.capacity * 4;
    assert!(out.selected == reference.selected, "{label}: selection differs from the fused BF16 incumbent");
    assert!(out.mirror == reference.mirror, "{label}: mirror differs from the fused BF16 incumbent");
    assert_eq!(out.persisted, reference.persisted, "{label}: persisted flag");
    assert_eq!(out.persisted, with_mirror, "{label}: persisted without/with mirror");
    assert!(all_poison(&out.selected[used..]), "{label}: wrote past the selected rows");
    assert!(all_poison(&out.mirror[geo.capacity * 4..]), "{label}: wrote past the mirror");
    if with_mirror {
        assert!(
            out.mirror[..geo.capacity * 4] == out.selected[used - geo.capacity * 4..used],
            "{label}: mirror is not the final row's selection"
        );
    } else {
        assert!(all_poison(&out.mirror), "{label}: wrote a mirror it was not given");
    }
    assert_eq!(out.stats.len(), stat_rows.unwrap_or(0), "{label}: one stat per row");
}

fn score_arms(gpu: &Gpu) -> Vec<bool> {
    if gpu.arch_caps.is_gfx1151() { vec![false, true] } else { vec![false] }
}

#[test]
fn qsa_bf16_filter_arm_end_to_end() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu, SMALL);
    let arms = score_arms(&gpu);
    let default_knobs = QsaFilterKnobs::default();
    let cap16 = QsaFilterKnobs { cap: 16, ..default_knobs };
    let underfill = QsaFilterKnobs { target: 1, ..default_knobs };
    let knob_sets = [("default", default_knobs), ("cap16", cap16), ("underfill", underfill)];
    let mut cases = 0usize;
    for blocks in [2048usize, 8192, 32768, 65536] {
        for kind in [Keys::Dense, Keys::Ties] {
            for cover in [Cover::Full(0), Cover::Full(3), Cover::Partial(1)] {
                let pooled = upload_bf16(&mut gpu, &keys_bf16(kind, blocks, cover.hot_from(blocks)));
                for rows in [1usize, 8, 17] {
                    let ps = cover.position_start(blocks, rows);
                    // The live count is `(position_start + rows) / compress`; a
                    // partial cover's pooled arena is longer (blocks past it are hot).
                    let p = fx.params_shaped(&pooled, (ps + rows) / COMPRESS, blocks, rows, ps);
                    let with = run_reference(&mut gpu, &fx, &p, true);
                    let without = run_reference(&mut gpu, &fx, &p, false);
                    assert!(with.persisted && !without.persisted);
                    assert!(with.selected == without.selected, "reference differs with a mirror");
                    for &score_pm in &arms {
                        for (knob_name, knobs) in knob_sets {
                            for with_mirror in [true, false] {
                                let label = format!(
                                    "blocks={blocks} keys={kind:?} {cover:?} rows={rows} pm_score={score_pm} knobs={knob_name} mirror={with_mirror}"
                                );
                                let reference = if with_mirror { &with } else { &without };
                                let out = run_filter(&mut gpu, &fx, &p, with_mirror, score_pm, knobs);
                                check_against(&label, &out, reference, SMALL, rows, with_mirror, Some(rows));
                                if knob_name == "cap16" {
                                    // The chosen-th largest key bounds the candidate
                                    // count below by the budget, past a cap of 16.
                                    for (row, stat) in out.stats.iter().enumerate() {
                                        assert!(stat.repaired, "{label}: row {row} was not reported repaired");
                                    }
                                }
                                let again = run_filter(&mut gpu, &fx, &p, with_mirror, score_pm, knobs);
                                check_against(&format!("{label} repeat"), &again, reference, SMALL, rows, with_mirror, Some(rows));
                                assert_eq!(
                                    stat_tuples(&again.stats),
                                    stat_tuples(&out.stats),
                                    "{label}: stats differ between repeats"
                                );
                                cases += 1;
                            }
                        }
                    }
                }
                gpu.free_tensor(pooled).expect("free pooled");
            }
        }
    }
    fx.free(&mut gpu);
    println!("{cases} BF16 filter cases exact");
}

/// The pooling capacity of the longest supported context (256K tokens).
const CAPACITY_BLOCKS: usize = 65536;

#[test]
fn qsa_bf16_split_arms_and_shape_bounds_end_to_end() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu, PRODUCTION);
    let default_knobs = QsaFilterKnobs::default();
    let arms = [
        ("select", QsaBf16SplitArm::Select),
        ("filter", QsaBf16SplitArm::Filter(default_knobs)),
        ("filter-cap16", QsaBf16SplitArm::Filter(QsaFilterKnobs { cap: 16, ..default_knobs })),
        ("filter-underfill", QsaBf16SplitArm::Filter(QsaFilterKnobs { target: 1, ..default_knobs })),
    ];
    let mut cases = 0usize;
    for blocks in [2048usize, 8192, 32768, 65536] {
        for kind in [Keys::Dense, Keys::Ties] {
            for cover in [Cover::Full(0), Cover::Full(3), Cover::Partial(1)] {
                let pooled = upload_bf16(&mut gpu, &keys_bf16(kind, blocks, cover.hot_from(blocks)));
                for rows in [1usize, 8, 17] {
                    let ps = cover.position_start(blocks, rows);
                    let live = (ps + rows) / COMPRESS;
                    let p = fx.params_shaped(&pooled, live, blocks, rows, ps);
                    let with = run_reference(&mut gpu, &fx, &p, true);
                    let without = run_reference(&mut gpu, &fx, &p, false);
                    // Eager (live count) and the retained shape at the tight bound
                    // and at a capacity bound well above the live count.
                    let mut shapes = vec![(false, blocks)];
                    shapes.push((true, blocks));
                    shapes.push((true, CAPACITY_BLOCKS.max(blocks) + 777));
                    for (arm_name, arm) in arms {
                        for &(bounded, shape) in &shapes {
                            let q = fx.params_shaped(&pooled, live, shape, rows, ps);
                            for with_mirror in [true, false] {
                                let label = format!(
                                    "blocks={blocks} keys={kind:?} {cover:?} rows={rows} arm={arm_name} bounded={bounded} shape={shape} mirror={with_mirror}"
                                );
                                let reference = if with_mirror { &with } else { &without };
                                let out = run_split(&mut gpu, &fx, &q, with_mirror, arm, bounded);
                                let stats = matches!(arm, QsaBf16SplitArm::Filter(_)).then_some(rows);
                                check_against(&label, &out, reference, PRODUCTION, rows, with_mirror, stats);
                                if arm_name == "filter-cap16" {
                                    for (row, stat) in out.stats.iter().enumerate() {
                                        assert!(stat.repaired, "{label}: row {row} was not reported repaired");
                                    }
                                }
                                cases += 1;
                            }
                        }
                    }
                }
                gpu.free_tensor(pooled).expect("free pooled");
            }
        }
    }
    // Multi-group retained shape: a 1M-block bound leaves room for 16 score
    // rows, so 17 rows run as two groups with grouped position addends.
    for blocks in [2048usize, 8192] {
        let pooled = upload_bf16(&mut gpu, &keys_bf16(Keys::Ties, blocks, blocks));
        for rows in [17usize] {
            let ps = Cover::Full(2).position_start(blocks, rows);
            let p = fx.params(&pooled, blocks, rows, ps);
            let with = run_reference(&mut gpu, &fx, &p, true);
            let q = fx.params_shaped(&pooled, blocks, 1_000_000, rows, ps);
            for (arm_name, arm) in arms {
                let label = format!("multi-group blocks={blocks} rows={rows} arm={arm_name}");
                let out = run_split(&mut gpu, &fx, &q, true, arm, true);
                let stats = matches!(arm, QsaBf16SplitArm::Filter(_)).then_some(rows);
                check_against(&label, &out, &with, PRODUCTION, rows, true, stats);
                cases += 1;
            }
        }
        gpu.free_tensor(pooled).expect("free pooled");
    }
    fx.free(&mut gpu);
    println!("{cases} BF16 split cases exact");
}

// ---------------------------------------------------------------------------
// recorded replay

/// Opens a recording window, runs `body`, closes it; returns the tape.
fn record_tape(gpu: &mut Gpu, body: impl FnOnce(&mut Gpu)) -> Vec<(String, [u32; 3], [u32; 3])> {
    gpu.replay = ReplayController::new_armed(ReplayBackendRequest::Auto);
    gpu.replay.begin_capture().expect("open recording window");
    body(gpu);
    gpu.replay.finish_capture().expect("close recording window");
    gpu.replay
        .recorded_launches()
        .iter()
        .map(|l| (l.kernel.clone(), l.grid, l.block))
        .collect()
}

/// Visible-prefix keys for a replay at `position_start`: blocks no row can see
/// are huge.
fn keys_for_position(base: &[u16], blocks: usize, position_start: usize, rows: usize) -> Vec<u16> {
    let live = ((position_start + rows) / COMPRESS).min(blocks);
    let mut bits = base.to_vec();
    bits[live * DIM..].fill(HOT);
    bits
}

fn replay_positions(blocks: usize, rows: usize) -> Vec<usize> {
    let max = blocks * COMPRESS - rows;
    let mut v = vec![0usize, 2, 7, 4095, blocks * 2, blocks * 3 + 1, max.saturating_sub(5), max.saturating_sub(1), max];
    v.retain(|&p| p <= max);
    v.sort_unstable();
    v.dedup();
    v
}

/// Records `arm` at each capture position, checks the tape is position
/// independent and replays it at every position against the fused incumbent.
/// `record` runs the call under test (recording window open).
fn replay_matches_fused(
    gpu: &mut Gpu,
    fx: &Fixture,
    label: &str,
    blocks: usize,
    rows: usize,
    kind: Keys,
    expect_tape: &dyn Fn(&[(String, [u32; 3], [u32; 3])]),
    record: &dyn Fn(&mut Gpu, &IndexedAttentionSelectBatch<'_>),
) -> usize {
    let base = keys_bf16(kind, blocks, blocks);
    let pooled = upload_bf16(gpu, &base);
    let max = blocks * COMPRESS - rows;
    let mut checked = 0usize;
    let mut first_tape: Option<Vec<(String, [u32; 3], [u32; 3])>> = None;
    for capture_at in [7usize.min(max), max / 2, max] {
        let bits = keys_for_position(&base, blocks, capture_at, rows);
        gpu.hip
            .memcpy_htod(&pooled.buf, &bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())
            .expect("rewrite pooled");
        let live = (capture_at + rows) / COMPRESS;
        let p = fx.params_shaped(&pooled, live, blocks, rows, capture_at);
        fx.poison(gpu);
        let tape = record_tape(gpu, |g| record(g, &p));
        expect_tape(&tape);
        match &first_tape {
            None => first_tape = Some(tape.clone()),
            Some(first) => assert!(
                *first == tape,
                "{label}: the tape recorded at position {capture_at} differs from the first (kernels or grids depend on the capture position)"
            ),
        }
        let count = tape.len();
        for position in replay_positions(blocks, rows) {
            let bits = keys_for_position(&base, blocks, position, rows);
            gpu.hip
                .memcpy_htod(&pooled.buf, &bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())
                .expect("rewrite pooled");
            let live = (position + rows) / COMPRESS;
            let q = fx.params_shaped(&pooled, live, blocks, rows, position);
            let reference = run_reference(gpu, fx, &q, true);
            fx.poison(gpu);
            gpu.replay_recorded_hip_prefix_at(count, position)
                .unwrap_or_else(|e| panic!("{label}: replay at {position}: {e}"));
            let out = fx.download(gpu, true, Vec::new());
            check_against(
                &format!("{label} captured_at={capture_at} replayed_at={position}"),
                &out,
                &reference,
                fx.geo,
                rows,
                true,
                None,
            );
            checked += 1;
        }
    }
    gpu.free_tensor(pooled).expect("free pooled");
    checked
}

#[test]
fn qsa_bf16_split_recorded_replay_matches_fused_at_positions() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu, PRODUCTION);
    let default_knobs = QsaFilterKnobs::default();
    let mut replays = 0usize;
    for blocks in [8192usize, 65536] {
        // Warm every scratch before the first recording, as the pre-capture
        // reserve does for the model: the fused reference grows the same slot
        // (global score rows) and the recorded pointers must not move.
        let warm = upload_bf16(&mut gpu, &keys_bf16(Keys::Dense, blocks, blocks));
        for rows in [1usize, 17] {
            let p = fx.params_shaped(&warm, blocks, blocks, rows, blocks * COMPRESS - rows);
            run_reference(&mut gpu, &fx, &p, true);
            run_split(&mut gpu, &fx, &p, true, QsaBf16SplitArm::Select, true);
            run_split(&mut gpu, &fx, &p, true, QsaBf16SplitArm::Filter(default_knobs), true);
        }
        gpu.free_tensor(warm).expect("free warm");
        for kind in [Keys::Dense, Keys::Ties] {
            for rows in [1usize, 8, 17] {
                for (name, arm, kernels) in [
                    (
                        "select",
                        QsaBf16SplitArm::Select,
                        vec!["indexed_attention_select_scores_rows16_bf16", "indexed_attention_select_from_scores"],
                    ),
                    (
                        "filter",
                        QsaBf16SplitArm::Filter(default_knobs),
                        vec![
                            "indexed_attention_select_scores_rows16_bf16",
                            "qsa_filter_threshold",
                            "qsa_filter_compact",
                            "qsa_filter_finish",
                        ],
                    ),
                ] {
                    let label = format!("blocks={blocks} keys={kind:?} rows={rows} arm={name}");
                    let score_rows = if rows > 8 { "rows16" } else { "rows8" };
                    let expected: Vec<String> = kernels
                        .iter()
                        .map(|k| k.replace("rows16", score_rows))
                        .collect();
                    replays += replay_matches_fused(
                        &mut gpu,
                        &fx,
                        &label,
                        blocks,
                        rows,
                        kind,
                        &|tape| {
                            let names: Vec<&str> = tape.iter().map(|t| t.0.as_str()).collect();
                            assert_eq!(names, expected, "{label}: recorded kernels");
                            // Position-independent bound grids.
                            assert_eq!(tape[0].1[0] as usize, blocks.div_ceil(256), "{label}: score grid x");
                            if name == "filter" {
                                assert_eq!(tape[2].1[0] as usize, blocks.div_ceil(4096), "{label}: compact grid x");
                            }
                        },
                        &|g, p| {
                            let (persisted, _) = indexed_attention_select_batch_bf16_split_arm(
                                g,
                                p,
                                Some(&fx.mirror),
                                arm,
                                true,
                                false,
                            )
                            .expect("recorded split call");
                            assert!(persisted);
                        },
                    );
                }
            }
        }
    }
    fx.free(&mut gpu);
    println!("{replays} recorded replays exact");
}

// ---------------------------------------------------------------------------
// production route

/// `false`: `HIPFIRE_QWEN4_QSA_BF16_SPLIT=0` (the kill switch; must force the
/// incumbent); `true`: unset or any other value (the default route on exact
/// gfx1201).
fn split_switch() -> bool {
    std::env::var("HIPFIRE_QWEN4_QSA_BF16_SPLIT").as_deref() != Ok("0")
}

#[test]
fn qsa_bf16_split_production_route_matches_fused() {
    let split_on = split_switch();
    let Some(mut gpu) = gpu_or_skip() else { return };
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("skip: the production route is exact gfx1201 only (got {})", gpu.arch);
        return;
    }
    let fx = Fixture::new(&mut gpu, PRODUCTION);
    let reserved =
        reserve_qsa_bf16_split_scratch(&mut gpu, DType::BF16).expect("reserve split scratch");
    assert_eq!(reserved > 0, split_on, "the reserve is live exactly when the route is");
    assert_eq!(
        reserve_qsa_bf16_split_scratch(&mut gpu, DType::F32).expect("F32 reserve"),
        0,
        "F32 pooled keys never reserve"
    );
    let mut cases = 0usize;
    for blocks in [256usize, 2048, 8192, 32768, 65536] {
        for kind in [Keys::Dense, Keys::Ties] {
            let cover = Cover::Full(1);
            let pooled = upload_bf16(&mut gpu, &keys_bf16(kind, blocks, blocks));
            for rows in [1usize, 8, 17] {
                let ps = cover.position_start(blocks, rows);
                let p = fx.params_shaped(&pooled, blocks, CAPACITY_BLOCKS.max(blocks), rows, ps);
                let reference = run_reference(&mut gpu, &fx, &p, true);
                fx.poison(&gpu);
                let persisted =
                    indexed_attention_select_batch_mirrored(&mut gpu, &p, &fx.mirror).expect("production select");
                let last = gpu.last_launched_kernel().map(str::to_owned);
                let out = fx.download(&gpu, persisted, Vec::new());
                check_against(
                    &format!("production eager blocks={blocks} keys={kind:?} rows={rows} last={last:?}"),
                    &out,
                    &reference,
                    PRODUCTION,
                    rows,
                    true,
                    None,
                );
                if blocks == 65536 {
                    let want = if split_on { "qsa_filter_finish" } else { "indexed_attention_select_bf16_batched" };
                    assert_eq!(last.as_deref(), Some(want), "65536 blocks, split switch {split_on}");
                }
                if blocks == 256 {
                    assert_eq!(
                        last.as_deref(),
                        Some("indexed_attention_select_bf16_batched"),
                        "256 blocks keep the fused incumbent"
                    );
                }
                // Recorded: the decision reads the shape bound only.
                let tape = record_tape(&mut gpu, |g| {
                    fx.poison(g);
                    indexed_attention_select_batch_mirrored(g, &p, &fx.mirror).expect("recorded production select");
                });
                let names: Vec<&str> = tape.iter().map(|t| t.0.as_str()).collect();
                let split_tape =
                    names.iter().any(|n| n.starts_with("indexed_attention_select_scores_rows") && n.ends_with("_bf16"));
                assert_eq!(
                    split_tape, split_on,
                    "recorded production tape at shape {CAPACITY_BLOCKS}, split switch {split_on}: {names:?}"
                );
                let count = tape.len();
                let position = ps / 2;
                let live = (position + rows) / COMPRESS;
                let q = fx.params_shaped(&pooled, live, CAPACITY_BLOCKS.max(blocks), rows, position);
                let reference = run_reference(&mut gpu, &fx, &q, true);
                fx.poison(&gpu);
                gpu.replay_recorded_hip_prefix_at(count, position).expect("production replay");
                let out = fx.download(&gpu, true, Vec::new());
                check_against(
                    &format!("production replay blocks={blocks} keys={kind:?} rows={rows} at {position}"),
                    &out,
                    &reference,
                    PRODUCTION,
                    rows,
                    true,
                    None,
                );
                cases += 1;
            }
            gpu.free_tensor(pooled).expect("free pooled");
        }
    }
    fx.free(&mut gpu);
    println!("{cases} production-route cases exact");
}

// ---------------------------------------------------------------------------
// crossover timing

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Per-call GPU time in microseconds of `call`, measured as the replay of a
/// stream-captured graph of `CALLS` back-to-back calls (the retained replay's
/// shape: no host launch cost, kernel gaps only); median of nine replays.
fn graph_time_us(gpu: &mut Gpu, call: &mut dyn FnMut(&mut Gpu)) -> f64 {
    const CALLS: usize = 16;
    gpu.ensure_capture_stream().expect("capture stream");
    call(gpu);
    gpu.hip.device_synchronize().expect("warm");
    gpu.begin_stream_capture().expect("begin capture");
    for _ in 0..CALLS {
        call(gpu);
    }
    let graph = gpu.end_stream_capture().expect("end capture");
    let exec = gpu.hip.graph_instantiate(&graph).expect("instantiate");
    let events = [gpu.hip.event_create().expect("event"), gpu.hip.event_create().expect("event")];
    gpu.launch_graph(&exec).expect("warm replay");
    gpu.hip.device_synchronize().expect("sync");
    let mut samples = Vec::new();
    for _ in 0..9 {
        gpu.hip.event_record(&events[0], None).expect("event");
        gpu.launch_graph(&exec).expect("replay");
        gpu.hip.event_record(&events[1], None).expect("event");
        gpu.hip.event_synchronize(&events[1]).expect("sync");
        samples.push(gpu.hip.event_elapsed_ms(&events[0], &events[1]).expect("elapsed") as f64 * 1000.0 / CALLS as f64);
    }
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(graph).expect("destroy graph");
    median(samples)
}

#[test]
#[ignore = "timing table; run on an idle card"]
fn qsa_bf16_split_crossover_table() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu, PRODUCTION);
    let knobs = QsaFilterKnobs::default();
    println!("arch={}", gpu.arch);
    println!("shape_blocks  blocks  rows  fused_us  select_us  filter_us   (select/filter: bounded split route)");
    for shape_kind in ["capacity", "tight"] {
        for blocks in [512usize, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536] {
            let pooled = upload_bf16(&mut gpu, &keys_bf16(Keys::Dense, blocks, blocks));
            let shape = if shape_kind == "capacity" { CAPACITY_BLOCKS } else { blocks };
            for rows in [1usize, 8, 16] {
                let ps = Cover::Full(1).position_start(blocks, rows);
                let p = fx.params_shaped(&pooled, blocks, shape, rows, ps);
                let fused = graph_time_us(&mut gpu, &mut |g| {
                    indexed_attention_select_batch_pm_arm(g, &p, Some(&fx.mirror), QsaSelectPmArm::Hipcc).expect("fused");
                });
                let select = graph_time_us(&mut gpu, &mut |g| {
                    indexed_attention_select_batch_bf16_split_arm(g, &p, Some(&fx.mirror), QsaBf16SplitArm::Select, true, false)
                        .expect("split select");
                });
                let filter = graph_time_us(&mut gpu, &mut |g| {
                    indexed_attention_select_batch_bf16_split_arm(
                        g,
                        &p,
                        Some(&fx.mirror),
                        QsaBf16SplitArm::Filter(knobs),
                        true,
                        false,
                    )
                    .expect("split filter");
                });
                println!("{shape:>9}  {blocks:>6}  {rows:>4}  {fused:>8.1}  {select:>9.1}  {filter:>9.1}");
            }
            gpu.free_tensor(pooled).expect("free pooled");
        }
    }
    fx.free(&mut gpu);
}

#[test]
#[ignore = "timing table; run on an idle card"]
fn qsa_bf16_split_prefill_crossover_table() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::with_max_rows(&mut gpu, PRODUCTION, 2048);
    let knobs = QsaFilterKnobs::default();
    println!("arch={}", gpu.arch);
    println!("eager shape (shape_blocks = blocks); graph-replayed GPU time per call");
    println!("blocks  rows  fused_us  select_us  filter_us");
    for blocks in [1024usize, 2048, 4096, 8192, 16384, 32768, 65536] {
        let pooled = upload_bf16(&mut gpu, &keys_bf16(Keys::Dense, blocks, blocks));
        for rows in [64usize, 256, 1024, 2048] {
            let ps = Cover::Full(1).position_start(blocks, rows);
            let p = fx.params(&pooled, blocks, rows, ps);
            let fused = graph_time_us(&mut gpu, &mut |g| {
                indexed_attention_select_batch_pm_arm(g, &p, Some(&fx.mirror), QsaSelectPmArm::Hipcc).expect("fused");
            });
            let select = graph_time_us(&mut gpu, &mut |g| {
                indexed_attention_select_batch_bf16_split_arm(g, &p, Some(&fx.mirror), QsaBf16SplitArm::Select, false, false)
                    .expect("split select");
            });
            let filter = graph_time_us(&mut gpu, &mut |g| {
                indexed_attention_select_batch_bf16_split_arm(
                    g,
                    &p,
                    Some(&fx.mirror),
                    QsaBf16SplitArm::Filter(knobs),
                    false,
                    false,
                )
                .expect("split filter");
            });
            println!("{blocks:>6}  {rows:>4}  {fused:>8.1}  {select:>9.1}  {filter:>9.1}");
        }
        gpu.free_tensor(pooled).expect("free pooled");
    }
    fx.free(&mut gpu);
}
