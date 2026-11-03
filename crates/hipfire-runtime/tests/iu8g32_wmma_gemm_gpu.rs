// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the compact IU8G128 (qt59) batched residual GEMM family.
//!
//! qt59 is `[K signed i8 codes][K/128 f16 scales]`. The GEMM kernels round the
//! dequantized weight to f16 and run f16 WMMA; this test checks every
//! candidate against an independent f64 reference of the same compact row,
//! and checks determinism. `IU8_REAL=<.iu4p>` sweeps every real qt59 tensor.
//!
//!   cargo test -p hipfire-runtime --release --test iu8g32_wmma_gemm_gpu -- --ignored --nocapture

use half::f16;
use rdna_compute::Gpu;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random compact row: one f16 scale per 128 codes.
fn iu8_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 128, 0);
    let g = k / 128;
    let mut out = vec![0u8; m * (k + 2 * g)];
    for r in 0..m {
        let row = &mut out[r * (k + 2 * g)..(r + 1) * (k + 2 * g)];
        for b in &mut row[..k] {
            *b = (rng.next() % 255) as u8 ^ 0x80;
        }
        for b in 0..g {
            let sc = 0.001 + 0.02 * rng.f().abs();
            row[k + 2 * b..k + 2 * b + 2]
                .copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
        }
    }
    out
}

fn dequant(iu: &[u8], m: usize, k: usize) -> Vec<f64> {
    let g = k / 128;
    let row_b = k + 2 * g;
    let mut w = vec![0f64; m * k];
    for r in 0..m {
        let row = &iu[r * row_b..(r + 1) * row_b];
        for b in 0..g {
            let sc =
                f16::from_bits(u16::from_le_bytes([row[k + 2 * b], row[k + 2 * b + 1]])).to_f64();
            for i in 0..128 {
                w[r * k + b * 128 + i] = sc * (row[b * 128 + i] as i8) as f64;
            }
        }
    }
    w
}

fn check(
    gpu: &mut Gpu,
    label: &str,
    iu: &[u8],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut Rng,
) -> f64 {
    let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
    let seed: Vec<f32> = (0..n * m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let wi = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
    gpu.invalidate_fp16_cache();
    let yi = gpu.upload_f32(&seed, &[n * m]).unwrap();
    gpu.gemm_iu8g32_residual_wmma(&wi, &xd, &yi, m, k, n)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let b = gpu.download_f32(&yi).unwrap();
    let y2 = gpu.upload_f32(&seed, &[n * m]).unwrap();
    gpu.gemm_iu8g32_residual_wmma(&wi, &xd, &y2, m, k, n)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let b2 = gpu.download_f32(&y2).unwrap();
    assert!(
        b.iter().zip(&b2).all(|(u, v)| u.to_bits() == v.to_bits()),
        "{label} m={m} k={k} n={n}: nondeterministic"
    );

    let wd = dequant(iu, m, k);
    let xh: Vec<f64> = x.iter().map(|v| f16::from_f32(*v).to_f64()).collect();
    let mut abs = 0f64;
    let mut sc = 1e-12f64;
    for t in 0..n {
        for r in 0..m {
            let wx: f64 = (0..k).map(|c| wd[r * k + c] * xh[t * k + c]).sum();
            let l1: f64 = (0..k).map(|c| (wd[r * k + c] * xh[t * k + c]).abs()).sum();
            sc = sc.max(l1);
            abs = abs.max((b[t * m + r] as f64 - (wx + seed[t * m + r] as f64)).abs());
        }
    }
    let rel = abs / sc;
    eprintln!("{label:<30} m={m:<6} k={k:<6} n={n:<4} abs={abs:.3e} rel={rel:.3e}");
    assert!(rel < 1e-2, "{label} rel {rel:.3e}");
    for tt in [wi, xd, yi, y2] {
        gpu.free_tensor(tt).ok();
    }
    rel
}

#[test]
#[ignore]
fn iu8g32_wmma_matches_reference() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    // Probe with a width the compact row admits: K=32 is refused by design
    // and used to turn this whole gate into a silent SKIP.
    if !gpu.iu8g32_wmma_eligible(128) {
        eprintln!("SKIP — not gfx1100/gfx1151.");
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0f64;
    let mut cases = 0;
    for &m in &[1usize, 7, 129, 256] {
        // 384 and 2688 are K%128 but not K%256: the twin is their only GEMM.
        for &k in &[128usize, 256, 384, 2560, 2688, 4096] {
            let iu = iu8_random(&mut rng, m, k);
            for &n in &[1usize, 2, 15, 16, 17, 47, 48, 77] {
                worst = worst.max(check(&mut gpu, "random", &iu, m, k, n, &mut rng));
                cases += 1;
            }
        }
    }
    if let Ok(p) = std::env::var("IU8_REAL") {
        use hipfire_runtime::hfq::HfqFile;
        let f = HfqFile::open(std::path::Path::new(&p)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for t in f.tensors().to_vec() {
            let (ti, ib) = f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 59 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen.insert((k, m)) {
                continue;
            }
            assert_eq!(ib.len(), m * (k + k / 64), "{}: bad qt59 row size", t.name);
            for &n in &[1usize, 77] {
                check(
                    &mut gpu,
                    &format!("real:{}", &t.name[t.name.len().saturating_sub(24)..]),
                    &ib,
                    m,
                    k,
                    n,
                    &mut rng,
                );
                cases += 1;
            }
        }
    }
    eprintln!("iu8g32 wmma: {cases} cases, worst rel vs f64 = {worst:.3e}");
}

fn w8a8_weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let stride = k + k / 64;
    let mut out = vec![0u8; m * stride];
    for r in 0..m {
        for kb in 0..k / 128 {
            let sc = 0.001 + 0.02 * rng.f().abs();
            let off = r * stride + k + kb * 2;
            out[off..off + 2].copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
            for j in 0..128 {
                out[r * stride + kb * 128 + j] = (rng.next() % 255) as u8;
            }
        }
    }
    out
}

#[test]
#[ignore]
fn w8a8_lds_min_matches_reference_and_times() {
    use std::time::Instant;
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    let (m, k, n) = (64usize, 512usize, 32usize);
    let w8 = w8a8_weights(&mut rng, m, k);
    let wd = gpu.upload_raw(&w8, &[w8.len()]).unwrap();
    let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
    let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
    let xq = gpu.ensure_int8_mmq_linear_x(&xd, n, k).unwrap();
    let yd = gpu.upload_f32(&vec![0.0f32; n * m], &[n * m]).unwrap();
    gpu.gemm_iu8g32_w8a8_min(&wd, xq, &yd, m, k, n, false)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&yd).unwrap();
    let xq_len = (k / 128) * n * 136;
    let xt = gpu.upload_raw(&vec![0u8; xq_len], &[xq_len]).unwrap();
    unsafe {
        gpu.hip
            .memcpy_dtod_raw(xt.buf.as_ptr(), xq, xq_len)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let xqb = gpu.download_raw_bytes(&xt).unwrap();
    let stride = k + k / 64;
    let mut want = vec![0f32; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            for kb in 0..k / 128 {
                let blk = &xqb[(kb * n + t) * 136..(kb * n + t + 1) * 136];
                let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                let sc = f16::from_bits(u16::from_le_bytes([
                    w8[r * stride + k + kb * 2],
                    w8[r * stride + k + kb * 2 + 1],
                ]))
                .to_f32();
                let mut dot = 0i64;
                for j in 0..128 {
                    dot += (w8[r * stride + kb * 128 + j] as i8) as i64 * (blk[8 + j] as i8) as i64;
                }
                acc = (sc * d).mul_add(dot as f32, acc);
            }
            want[t * m + r] = acc;
        }
    }
    let max_ref = want
        .iter()
        .fold(0f64, |a, v| a.max(v.abs() as f64))
        .max(1e-30);
    let rel = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0f64, f64::max)
        / max_ref;
    eprintln!("w8a8 min: m={m} k={k} n={n} worst rel={rel:.3e}");
    assert!(rel < 1e-5, "w8a8 min rel {rel:.3e}");

    // The residual epilogue (`add = true`) is the role both qt59 prefill
    // bodies use; it must equal seed + the set result exactly.
    let seed: Vec<f32> = (0..n * m).map(|i| 0.11 * (i % 13) as f32 - 0.7).collect();
    let y_add = gpu.upload_f32(&seed, &[n * m]).unwrap();
    gpu.gemm_iu8g32_w8a8_min(&wd, xq, &y_add, m, k, n, true)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let add = gpu.download_f32(&y_add).unwrap();
    assert!(
        add.iter()
            .zip(&got)
            .zip(&seed)
            .all(|((a, s), b)| *a == *s + *b),
        "w8a8 add epilogue != seed + set"
    );
    gpu.free_tensor(y_add).ok();

    let (tm, tk, tn) = (5120usize, 6144usize, 512usize);
    let tw = w8a8_weights(&mut rng, tm, tk);
    let twd = gpu.upload_raw(&tw, &[tw.len()]).unwrap();
    let tx: Vec<f32> = (0..tn * tk).map(|_| rng.f() * 2.0).collect();
    let txd = gpu.upload_f32(&tx, &[tn * tk]).unwrap();
    let txq = gpu.ensure_int8_mmq_linear_x(&txd, tn, tk).unwrap();
    let ty = gpu.upload_f32(&vec![0.0f32; tn * tm], &[tn * tm]).unwrap();
    for _ in 0..2 {
        gpu.gemm_iu8g32_w8a8_min(&twd, txq, &ty, tm, tk, tn, false)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..8 {
        gpu.gemm_iu8g32_w8a8_min(&twd, txq, &ty, tm, tk, tn, false)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 8.0;
    let flops = 2.0 * (tm as f64) * (tk as f64) * (tn as f64);
    eprintln!(
        "w8a8 min wo shape M={tm} K={tk} N={tn}: {ms:.3} ms ({:.1} TFLOPS)",
        flops / 1e9 / ms
    );
}

/// The W8A8 GEMM against an f64 reference of its own quantized activations
/// on ragged shapes: partial 128-row M and N tiles, several N tiles, and
/// every K the launcher admits up to a real `wo`. Both epilogues.
#[test]
#[ignore]
fn w8a8_lds_min_ragged_shapes() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu8g32_w8a8_eligible(256) {
        eprintln!("SKIP — no IU8 W8A8 GEMM on {}.", gpu.arch);
        return;
    }
    let mut rng = Rng(0x2468_ACE0_1357_9BDF);
    let mut worst = 0f64;
    for &(m, k) in &[(1usize, 256usize), (129, 512), (300, 2560), (130, 6144)] {
        let w8 = w8a8_weights(&mut rng, m, k);
        let wd = gpu.upload_raw(&w8, &[w8.len()]).unwrap();
        let stride = k + k / 64;
        for &n in &[1usize, 16, 127, 129, 300] {
            let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
            let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
            let xq = gpu.ensure_int8_mmq_linear_x(&xd, n, k).unwrap();
            let xq_len = (k / 128) * n * 136;
            let xt = gpu.upload_raw(&vec![0u8; xq_len], &[xq_len]).unwrap();
            unsafe {
                gpu.hip
                    .memcpy_dtod_raw(xt.buf.as_ptr(), xq, xq_len)
                    .unwrap();
            }
            let seed: Vec<f32> = (0..n * m).map(|i| 0.13 * (i % 7) as f32 - 0.4).collect();
            let ys = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).unwrap();
            let ya = gpu.upload_f32(&seed, &[n * m]).unwrap();
            gpu.gemm_iu8g32_w8a8_min(&wd, xq, &ys, m, k, n, false)
                .unwrap();
            gpu.gemm_iu8g32_w8a8_min(&wd, xq, &ya, m, k, n, true)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let xqb = gpu.download_raw_bytes(&xt).unwrap();
            let set = gpu.download_f32(&ys).unwrap();
            let add = gpu.download_f32(&ya).unwrap();
            let mut abs = 0f64;
            let mut scale = 1e-12f64;
            for t in 0..n {
                for r in 0..m {
                    let mut acc = 0f64;
                    let mut l1 = 0f64;
                    for kb in 0..k / 128 {
                        let blk = &xqb[(kb * n + t) * 136..(kb * n + t + 1) * 136];
                        let d = f32::from_le_bytes(blk[0..4].try_into().unwrap()) as f64;
                        let sc = f16::from_bits(u16::from_le_bytes([
                            w8[r * stride + k + kb * 2],
                            w8[r * stride + k + kb * 2 + 1],
                        ]))
                        .to_f64();
                        for j in 0..128 {
                            let p = sc
                                * d
                                * (w8[r * stride + kb * 128 + j] as i8) as f64
                                * (blk[8 + j] as i8) as f64;
                            acc += p;
                            l1 += p.abs();
                        }
                    }
                    scale = scale.max(l1);
                    let (s, a) = (set[t * m + r], add[t * m + r]);
                    // An unwritten output stays NaN; fail on any non-finite
                    // element instead of letting a max() swallow it.
                    assert!(
                        s.is_finite() && a.is_finite(),
                        "m={m} k={k} n={n} t={t} r={r}"
                    );
                    abs = abs.max((s as f64 - acc).abs());
                    abs = abs.max((a as f64 - (acc + seed[t * m + r] as f64)).abs());
                }
            }
            let rel = abs / scale;
            eprintln!("w8a8 ragged m={m:<4} k={k:<5} n={n:<4} rel={rel:.3e}");
            assert!(rel < 1e-5, "w8a8 m={m} k={k} n={n} rel {rel:.3e}");
            worst = worst.max(rel);
            for tt in [xd, xt, ys, ya] {
                gpu.free_tensor(tt).ok();
            }
        }
        gpu.free_tensor(wd).ok();
    }
    eprintln!("w8a8 ragged: worst rel vs f64 = {worst:.3e}");
}

/// Crossover timing for [`Gpu::iu8g32_gemm_route`]: the W8A8 route
/// (activation int8 quant + IU8 WMMA) against the f16 twin (activation f16
/// convert + f16 WMMA), both as `Y += W X` on real 27B qt59 shapes, each
/// activation conversion charged per call as it is in a layer loop.
///
///   cargo test -p hipfire-runtime --release --test iu8g32_wmma_gemm_gpu \
///     iu8g32_route_crossover -- --ignored --nocapture
#[test]
#[ignore]
fn iu8g32_route_crossover() {
    use std::time::Instant;
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu8g32_w8a8_eligible(256) {
        eprintln!("SKIP — no IU8 W8A8 GEMM on {}.", gpu.arch);
        return;
    }
    let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);
    for &(m, k) in &[(5120usize, 6144usize), (5120, 17408)] {
        let w8 = w8a8_weights(&mut rng, m, k);
        let wd = gpu.upload_raw(&w8, &[w8.len()]).unwrap();
        for &n in &[1usize, 16, 32, 40, 48, 56, 63, 64, 128, 512] {
            let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
            let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
            let y = gpu.upload_f32(&vec![0.0f32; n * m], &[n * m]).unwrap();
            let mut time = |gpu: &mut Gpu, w8a8: bool| {
                let mut run = |gpu: &mut Gpu| {
                    if w8a8 {
                        let xq = gpu.ensure_int8_mmq_linear_x(&xd, n, k).unwrap();
                        gpu.gemm_iu8g32_w8a8_min(&wd, xq, &y, m, k, n, true)
                            .unwrap();
                    } else {
                        gpu.invalidate_fp16_cache();
                        gpu.gemm_iu8g32_residual_wmma(&wd, &xd, &y, m, k, n)
                            .unwrap();
                    }
                };
                for _ in 0..3 {
                    run(gpu);
                }
                gpu.hip.device_synchronize().unwrap();
                let reps = 10;
                let t0 = Instant::now();
                for _ in 0..reps {
                    run(gpu);
                }
                gpu.hip.device_synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e6 / reps as f64
            };
            // ABAB so a DPM ramp is not charged to one arm.
            let (a1, b1) = (time(&mut gpu, true), time(&mut gpu, false));
            let (a2, b2) = (time(&mut gpu, true), time(&mut gpu, false));
            let (w8a8, twin) = (a1.min(a2), b1.min(b2));
            eprintln!(
                "crossover m={m} k={k} n={n:<4} w8a8={w8a8:>9.1}us f16twin={twin:>9.1}us twin/w8a8={:.2}",
                twin / w8a8
            );
            gpu.free_tensor(xd).ok();
            gpu.free_tensor(y).ok();
        }
        gpu.free_tensor(wd).ok();
    }
}
