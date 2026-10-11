// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the gfx1151 lane-rows PM kernels
//! (`qwen4_lane_rows_pm`, `rdna_compute::lane_rows_pm`).
//!
//! `lane_rows_pm_oracle [--bench N]`: for each export and every column count
//! `B` in `1..=8`, on seeded random weights and activations (random packed
//! payloads, finite f16 headers incl. zero and subnormal scales, x columns of
//! very different magnitudes with `+-0` and F32 subnormals, one all-zero
//! column and one column large enough to reach the k8 `hc_act` exp guards),
//! the PM launcher's whole `y` allocation (guards and unused columns
//! included) must equal the expectation built from
//!
//! 1. `B` launches of the 1-row incumbent (`gemv_mq6g256v2`,
//!    `gemv_mq6g256v2` per matrix for the x4 form, `gemv_q8_0_k8`,
//!    `gemv_q8_0` at K = 320), one per column, and
//! 2. the incumbent rows kernel at `rows = B` (`gemm_mq6g256v2_f32_rows`,
//!    `gemm_mq6g256v2_f32_rows_x4`, `gemv_q8_0_k8_rows`,
//!    `gemv_q8_0_staged_rows`; `B >= 2`).
//!
//! The first differing byte is reported as guard / (column, row, byte). The
//! process exits 0 only if every byte of every case matches.
//!
//! `--bench N`: after the oracle, median HIP-event microseconds (5 warmups, N
//! samples) of one PM launch vs one incumbent rows launch at the C8 lane-step
//! shapes (`fn-batch/s7/DAG.md` §5) at `B = 4` and `B = 8`, plus the weight
//! bytes per PM launch as GB/s.
//!
//! The incumbents run with `lane_rows_scope = false`, so they are always the
//! hipcc kernels whatever `HIPFIRE_FN_LANE_PM_ROWS` says.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXB: usize = 8;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
const SENTINEL: u8 = 0x7f;
const HC_SCALE: f32 = 0.25;
/// Column magnitudes: `0.0` is the all-zero (+-0) column, `1024` reaches the
/// k8 `hc_act` exp guards.
const AMPS: [f32; MAXB] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for c in &mut chunks {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let tail = self.next().to_le_bytes();
        rest.copy_from_slice(&tail[..rest.len()]);
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// A finite f16: mostly `2^-lo..2^hi`, sometimes `+-0` or subnormal.
    fn f16(&mut self, lo_exp: u16, hi_exp: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => sign | ((15 - lo_exp + self.below(u64::from(lo_exp + hi_exp + 1)) as u16) << 10) | self.below(0x400) as u16,
        }
    }
}

/// MQ6G256V2 weights: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16)
/// and 192 packed payload bytes.
fn gen_mq6(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.chunks_exact_mut(200) {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.f16(9, 2), rng.f16(7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// Q8_0 weights: `[m][k/32]` blocks of an f16 scale and 32 int8.
fn gen_q8(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 32) * 34];
    for blk in bytes.chunks_exact_mut(34) {
        rng.fill(&mut blk[2..]);
        blk[..2].copy_from_slice(&rng.f16(8, 0).to_le_bytes());
    }
    bytes
}

/// `[MAXB][k]` activations with the per-column magnitudes of [`AMPS`].
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXB * k);
    for amp in AMPS {
        for _ in 0..k {
            x.push(match rng.below(64) {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                0 => 0.0,
                1 => -0.0,
                2..=5 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => rng.unit() * amp,
            });
        }
    }
    x
}

/// One output allocation: `GUARD | MAXB * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu.zeros(&[2 * GUARD + MAXB * m], DType::F32).map_err(err)?;
        Ok(Self { full, m })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXB * self.m)
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAXB * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip.memset(&self.full.buf, i32::from(SENTINEL), self.bytes()).map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.bytes());
        Ok(v)
    }
    /// Expected bytes: sentinel everywhere, `reference`'s first `b` columns in place.
    fn expected(&self, reference: &[u8], b: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + b * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXB * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (col, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("column {col} row {} byte {}", rem / 4, rem % 4)
        }
    }
}

type Launch = Box<dyn Fn(&mut Gpu, usize) -> Result<()>>;

/// One shape: `single(gpu, c)` is the 1-row incumbent on column `c` writing
/// column `c` of every output, `rows(gpu, b)` the incumbent rows kernel (b >=
/// 2), `pm(gpu, b)` the PM launcher.
struct Case {
    name: String,
    outs: Vec<Out>,
    /// Packed weight bytes one launch reads.
    weight_bytes: usize,
    single: Launch,
    rows: Launch,
    pm: Launch,
}

impl Case {
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.outs.iter().try_for_each(|o| o.reset(gpu))
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.outs.iter().map(|o| o.download(gpu)).collect()
    }
}

fn mq6_case(gpu: &mut Gpu, m: usize, k: usize, rng: &mut Rng) -> Result<Case> {
    let weights = gen_mq6(m, k, rng);
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = gen_x(k, rng);
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    Ok(Case {
        name: format!("mq6 m={m} k={k}"),
        outs: vec![out],
        weight_bytes: weights.len(),
        single: Box::new(move |gpu, c| {
            gpu.gemv_mq6g256v2(&w1, &x1.sub_offset(c * k, k), &y1.sub_offset(c * m, m), m, k).map_err(err)
        }),
        rows: Box::new(move |gpu, b| gpu.gemm_mq6g256v2_f32_rows(&w2, &x2, &y2, m, k, b).map_err(err)),
        pm: Box::new(move |gpu, b| gpu.gemv_mq6g256v2_xbatch_pm(&w, &x, &y, m, k, b).map_err(err)),
    })
}

fn mq6_x4_case(gpu: &mut Gpu, ms: [usize; 4], k: usize, rng: &mut Rng) -> Result<Case> {
    let mut ws = Vec::new();
    let mut outs = Vec::new();
    let mut weight_bytes = 0;
    for &m in &ms {
        // An unused slot still needs a valid pointer.
        let bytes = if m == 0 { vec![0u8; 256] } else { gen_mq6(m, k, rng) };
        weight_bytes += if m == 0 { 0 } else { bytes.len() };
        ws.push(Rc::new(gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?));
        outs.push(Out::new(gpu, m)?);
    }
    let xs = gen_x(k, rng);
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let ys: Vec<Rc<GpuTensor>> = outs.iter().map(|o| Rc::new(o.view())).collect();
    let (ws1, ys1, x1) = (ws.clone(), ys.clone(), x.clone());
    let (ws2, ys2, x2) = (ws.clone(), ys.clone(), x.clone());
    Ok(Case {
        name: format!("mq6 x4 ms={ms:?} k={k}"),
        outs,
        weight_bytes,
        single: Box::new(move |gpu, c| {
            for i in 0..4 {
                if ms[i] > 0 {
                    let (m, yv) = (ms[i], &ys1[i]);
                    gpu.gemv_mq6g256v2(&ws1[i], &x1.sub_offset(c * k, k), &yv.sub_offset(c * m, m), m, k).map_err(err)?;
                }
            }
            Ok(())
        }),
        rows: Box::new(move |gpu, b| {
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] = std::array::from_fn(|i| (&*ws2[i], &*ys2[i], ms[i]));
            gpu.gemm_mq6g256v2_f32_rows_x4(parts, &x2, k, b).map_err(err)
        }),
        pm: Box::new(move |gpu, b| {
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] = std::array::from_fn(|i| (&*ws[i], &*ys[i], ms[i]));
            gpu.gemv_mq6g256v2_x4_xbatch_pm(parts, &x, k, b).map_err(err)
        }),
    })
}

fn q8_k8_case(gpu: &mut Gpu, m: usize, k: usize, hc_act: bool, rng: &mut Rng) -> Result<Case> {
    let weights = gen_q8(m, k, rng);
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = gen_x(k, rng);
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let scale = hc_act.then_some(HC_SCALE);
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    Ok(Case {
        name: format!("q8 k8 m={m} k={k} hc_act={}", u8::from(hc_act)),
        outs: vec![out],
        weight_bytes: weights.len(),
        single: Box::new(move |gpu, c| {
            gpu.gemv_q8_0_k8(&w1, &x1.sub_offset(c * k, k), &y1.sub_offset(c * m, m), m, k, scale).map_err(err)
        }),
        rows: Box::new(move |gpu, b| gpu.gemv_q8_0_k8_rows(&w2, &x2, &y2, m, k, scale, b).map_err(err)),
        pm: Box::new(move |gpu, b| gpu.gemv_q8_0_k8_xbatch_pm(&w, &x, &y, m, k, scale, b).map_err(err)),
    })
}

fn q8_k320_case(gpu: &mut Gpu, m: usize, rng: &mut Rng) -> Result<Case> {
    const K: usize = 320;
    let weights = gen_q8(m, K, rng);
    let w = Rc::new(gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?);
    let xs = gen_x(K, rng);
    let x = Rc::new(gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?);
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let (w1, x1, y1) = (w.clone(), x.clone(), y.clone());
    let (w2, x2, y2) = (w.clone(), x.clone(), y.clone());
    Ok(Case {
        name: format!("q8 k320 m={m}"),
        outs: vec![out],
        weight_bytes: weights.len(),
        single: Box::new(move |gpu, c| {
            gpu.gemv_q8_0(&w1, &x1.sub_offset(c * K, K), &y1.sub_offset(c * m, m), m, K).map_err(err)
        }),
        rows: Box::new(move |gpu, b| gpu.gemv_q8_0_staged_rows(&w2, &x2, &y2, m, K, b).map_err(err)),
        pm: Box::new(move |gpu, b| gpu.gemv_q8_0_k320_xbatch_pm(&w, &x, &y, m, b).map_err(err)),
    })
}

/// `(mismatching byte count, first mismatching index)`.
fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

#[derive(Default)]
struct Tally {
    cases: usize,
    launches: usize,
    pm_vs_single: usize,
    pm_vs_rows: usize,
    rows_vs_single: usize,
}

impl Tally {
    fn failed(&self) -> bool {
        self.pm_vs_single + self.pm_vs_rows + self.rows_vs_single > 0
    }
}

/// Diff every output of `got` against `want`; the first difference is described.
fn compare(case: &Case, got: &[Vec<u8>], want: &[Vec<u8>]) -> (usize, Option<String>) {
    let mut bad = 0;
    let mut first = None;
    for (i, ((g, w), out)) in got.iter().zip(want).zip(&case.outs).enumerate() {
        let (n, at) = diff(g, w);
        bad += n;
        if let (None, Some(at)) = (&first, at) {
            first = Some(format!("matrix {i} {} (got {:#04x}, want {:#04x})", out.locate(at), g[at], w[at]));
        }
    }
    (bad, first)
}

fn verify(gpu: &mut Gpu, case: &Case, tally: &mut Tally) -> Result<()> {
    case.reset(gpu)?;
    for c in 0..MAXB {
        (case.single)(gpu, c)?;
    }
    let reference = case.download(gpu)?;
    let mut line = format!("{:<44}", case.name);
    let mut bad_case = 0;
    for b in 1..=MAXB {
        let expected: Vec<Vec<u8>> = case.outs.iter().zip(&reference).map(|(o, r)| o.expected(r, b)).collect();
        case.reset(gpu)?;
        (case.pm)(gpu, b)?;
        let got_pm = case.download(gpu)?;
        let (bad, first) = compare(case, &got_pm, &expected);
        tally.pm_vs_single += bad;
        tally.launches += 1;
        let mut note = String::new();
        if let Some(first) = first {
            note.push_str(&format!(" B={b}: PM differs from 1-row incumbent at {first};"));
        }
        if b >= 2 {
            case.reset(gpu)?;
            (case.rows)(gpu, b)?;
            let got_rows = case.download(gpu)?;
            let (rows_bad, rows_first) = compare(case, &got_rows, &expected);
            let (pm_rows_bad, pm_rows_first) = compare(case, &got_pm, &got_rows);
            tally.rows_vs_single += rows_bad;
            tally.pm_vs_rows += pm_rows_bad;
            tally.launches += 1;
            if let Some(first) = rows_first {
                note.push_str(&format!(" B={b}: incumbent rows differs from 1-row incumbent at {first};"));
            }
            if let Some(first) = pm_rows_first {
                note.push_str(&format!(" B={b}: PM differs from incumbent rows at {first};"));
            }
            bad_case += rows_bad + pm_rows_bad;
        }
        bad_case += bad;
        line.push_str(&note);
    }
    tally.cases += 1;
    println!("{line} {}", if bad_case == 0 { "ok (B=1..8)".to_string() } else { format!("MISMATCH ({bad_case} bytes)") });
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_us(gpu: &mut Gpu, n: usize, f: &dyn Fn(&mut Gpu) -> Result<()>) -> Result<f64> {
    let (start, stop) = (gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?);
    for _ in 0..5 {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        gpu.hip.event_record(&start, None).map_err(err)?;
        f(gpu)?;
        gpu.hip.event_record(&stop, None).map_err(err)?;
        gpu.hip.event_synchronize(&stop).map_err(err)?;
        samples.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1000.0);
    }
    gpu.hip.event_destroy(start).map_err(err)?;
    gpu.hip.event_destroy(stop).map_err(err)?;
    Ok(median(samples))
}

fn bench(gpu: &mut Gpu, case: &Case, n: usize) -> Result<()> {
    for b in [4, 8] {
        let pm = time_us(gpu, n, &|g| (case.pm)(g, b))?;
        let rows = time_us(gpu, n, &|g| (case.rows)(g, b))?;
        println!(
            "{:<44} B={b} PM {pm:>9.1} us  incumbent rows {rows:>9.1} us  x{:.2}  PM weights {:.0} GB/s",
            case.name,
            rows / pm,
            case.weight_bytes as f64 / (pm * 1e-6) / 1e9
        );
    }
    Ok(())
}

/// Every verification shape of the contract.
fn verify_cases(gpu: &mut Gpu) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut next_rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    for (m, k) in [(1, 256), (5, 2560), (13, 2560), (2560, 6144), (4099, 2560)] {
        cases.push(mq6_case(gpu, m, k, &mut next_rng())?);
    }
    for ms in [[10240, 6144, 48, 48], [12288, 640, 512, 512], [3, 0, 5, 0]] {
        cases.push(mq6_x4_case(gpu, ms, 2560, &mut next_rng())?);
    }
    for (m, k) in [(320, 10240), (7, 2048)] {
        for hc_act in [false, true] {
            cases.push(q8_k8_case(gpu, m, k, hc_act, &mut next_rng())?);
        }
    }
    for m in [10240, 17, 1] {
        cases.push(q8_k320_case(gpu, m, &mut next_rng())?);
    }
    Ok(cases)
}

fn run() -> Result<bool> {
    let mut bench_n = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--bench" => bench_n = Some(args.next().ok_or("--bench needs N")?.parse::<usize>().map_err(err)?),
            other => return Err(format!("unknown argument {other} (usage: lane_rows_pm_oracle [--bench N])")),
        }
    }
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch.as_str() != "gfx1151" {
        return Err(format!("needs exact gfx1151, found {}", gpu.arch.as_str()));
    }
    gpu.lane_rows_scope = false;
    if gpu.lane_rows_pm_enabled() {
        return Err("the incumbents must run unrouted (lane_rows_scope is off)".into());
    }
    let mut tally = Tally::default();
    for case in verify_cases(&mut gpu)? {
        verify(&mut gpu, &case, &mut tally)?;
    }
    println!(
        "{} cases, {} launches: PM vs 1-row incumbent {} bad bytes, PM vs incumbent rows {} bad bytes, incumbent rows vs 1-row {} bad bytes",
        tally.cases, tally.launches, tally.pm_vs_single, tally.pm_vs_rows, tally.rows_vs_single
    );
    if let Some(n) = bench_n.filter(|&n| n > 0) {
        let mut rng = Rng(0x00c8_0c8c_8c8c_8c8d);
        // The C8 lane-step shapes (DAG.md section 5), one case alive at a time.
        let case = mq6_x4_case(&mut gpu, [12288, 640, 512, 512], 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_x4_case(&mut gpu, [10240, 6144, 48, 48], 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_case(&mut gpu, 248_320, 2560, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = mq6_case(&mut gpu, 2560, 6144, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = q8_k8_case(&mut gpu, 320, 10240, true, &mut rng)?;
        bench(&mut gpu, &case, n)?;
        let case = q8_k320_case(&mut gpu, 10240, &mut rng)?;
        bench(&mut gpu, &case, n)?;
    }
    Ok(!tally.failed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("lane_rows_pm_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("lane_rows_pm_oracle: {e}");
            std::process::exit(2);
        }
    }
}
