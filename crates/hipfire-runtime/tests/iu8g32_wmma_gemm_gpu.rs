// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the IU8G32 (qt59) batched residual GEMM
//! `gemm_iu8g32_residual_wmma`.
//!
//! 1. Bit-exact: on the qt59 relayout of a Q8_0 tensor it returns exactly the
//!    bits of `gemm_q8_0_residual_wmma` (same f16 WMMA arithmetic).
//! 2. vs an f64 reference of the dequantized weights (f16 X): max abs, and
//!    rel = max abs / max_row sum|w x| (the kernel rounds W to f16).
//! 3. Real tensors (`IU8_REAL=<.iu4p>` `Q8_REAL=<.mq4p>`): every qt59 tensor
//!    shape vs its twin, bit-exact.
//!
//! M in {1, 7, 129, 256}, K in {32, 96, 2592, 4096}, N in {1, 2, 15, 16, 17, 77, 512}.
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

fn q8_0_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; m * (k / 32) * 34];
    for blk in out.chunks_mut(34) {
        let sc = 0.001 + 0.02 * rng.f().abs();
        blk[0..2].copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
        for b in &mut blk[2..] {
            *b = (rng.next() % 255) as u8 ^ 0x80; // -127..=127 after cast, plus -128 occasionally
        }
    }
    out
}

/// Q8_0 `[scale f16][32 i8]` blocks -> qt59 row `[K i8][K/32 f16]`.
fn relayout(q8: &[u8], m: usize, k: usize) -> Vec<u8> {
    let g = k / 32;
    let mut out = vec![0u8; m * (k + 2 * g)];
    for r in 0..m {
        let src = &q8[r * g * 34..(r + 1) * g * 34];
        let dst = &mut out[r * (k + 2 * g)..(r + 1) * (k + 2 * g)];
        for b in 0..g {
            dst[b * 32..b * 32 + 32].copy_from_slice(&src[b * 34 + 2..b * 34 + 34]);
            dst[k + 2 * b..k + 2 * b + 2].copy_from_slice(&src[b * 34..b * 34 + 2]);
        }
    }
    out
}

fn dequant_q8(q8: &[u8], m: usize, k: usize) -> Vec<f64> {
    let g = k / 32;
    let mut w = vec![0f64; m * k];
    for r in 0..m {
        for b in 0..g {
            let o = (r * g + b) * 34;
            let sc = f16::from_bits(u16::from_le_bytes([q8[o], q8[o + 1]])).to_f64();
            for i in 0..32 {
                w[r * k + b * 32 + i] = sc * (q8[o + 2 + i] as i8) as f64;
            }
        }
    }
    w
}

fn bits_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn check(
    gpu: &mut Gpu,
    label: &str,
    q8: &[u8],
    iu: &[u8],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut Rng,
    reference: bool,
) -> f64 {
    let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
    let seed: Vec<f32> = (0..n * m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let wq = gpu.upload_raw(q8, &[q8.len()]).unwrap();
    let wi = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
    // The f16-X scratch is keyed by source pointer; a host upload into a
    // recycled allocation would otherwise reuse the previous case's X.
    gpu.invalidate_fp16_cache();
    let yq = gpu.upload_f32(&seed, &[n * m]).unwrap();
    let yi = gpu.upload_f32(&seed, &[n * m]).unwrap();
    gpu.gemm_q8_0_residual_wmma(&wq, &xd, &yq, m, k, n).unwrap();
    gpu.gemm_iu8g32_residual_wmma(&wi, &xd, &yi, m, k, n)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let a = gpu.download_f32(&yq).unwrap();
    let b = gpu.download_f32(&yi).unwrap();
    let diff = bits_diff(&a, &b);
    let mut rel = 0.0;
    let mut abs = 0.0;
    if reference {
        let wd = dequant_q8(q8, m, k);
        let xh: Vec<f64> = x.iter().map(|v| f16::from_f32(*v).to_f64()).collect();
        let mut sc = 1e-12f64;
        for t in 0..n {
            for r in 0..m {
                let wx: f64 = (0..k).map(|c| wd[r * k + c] * xh[t * k + c]).sum();
                // Normalise by sum|w x| (the conditioning of the dot), not
                // |w x|: a single cancelling output (M=N=1) would inflate rel.
                let l1: f64 = (0..k).map(|c| (wd[r * k + c] * xh[t * k + c]).abs()).sum();
                sc = sc.max(l1);
                let e = (b[t * m + r] as f64 - (wx + seed[t * m + r] as f64)).abs();
                if e > 1.0 && std::env::var("IU8_DBG").is_ok() {
                    eprintln!(
                        "  bad t={t} r={r} got={} want={} seed={}",
                        b[t * m + r],
                        wx + seed[t * m + r] as f64,
                        seed[t * m + r]
                    );
                }
                abs = f64::max(abs, e);
            }
        }
        rel = abs / sc;
    }
    eprintln!("{label:<30} m={m:<6} k={k:<6} n={n:<4} bitdiff_vs_q8_0_wmma={diff} abs={abs:.3e} rel={rel:.3e}");
    assert_eq!(
        diff, 0,
        "{label} m={m} k={k} n={n}: IU8G32 WMMA differs from Q8_0 WMMA"
    );
    if reference && std::env::var("IU8_DBG").is_err() {
        assert!(rel < 1e-2, "{label} rel {rel:.3e}");
    }
    for tt in [wq, wi, xd, yq, yi] {
        gpu.free_tensor(tt).ok();
    }
    rel
}

#[test]
#[ignore]
fn iu8g32_wmma_residual_is_bit_exact_to_q8_0_wmma() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu8g32_wmma_eligible(32) {
        eprintln!("SKIP — not gfx1100/gfx1151.");
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0f64;
    let mut cases = 0;
    for &m in &[1usize, 7, 129, 256] {
        for &k in &[32usize, 96, 2592, 4096] {
            let q8 = q8_0_random(&mut rng, m, k);
            let iu = relayout(&q8, m, k);
            for &n in &[1usize, 2, 15, 16, 17, 77, 512] {
                let reference = m * k * n <= 256 * 4096 * 77;
                worst = worst.max(check(
                    &mut gpu, "random", &q8, &iu, m, k, n, &mut rng, reference,
                ));
                cases += 1;
            }
        }
    }
    if let (Ok(iu_path), Ok(q8_path)) = (std::env::var("IU8_REAL"), std::env::var("Q8_REAL")) {
        use hipfire_runtime::hfq::HfqFile;
        let iu_f = HfqFile::open(std::path::Path::new(&iu_path)).unwrap();
        let q8_f = HfqFile::open(std::path::Path::new(&q8_path)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for t in iu_f.tensors().to_vec() {
            let (ti, ib) = iu_f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 59 || ti.shape.len() < 2 || t.name.contains("embed") {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen.insert((k, m)) {
                continue;
            }
            let (tq, qb) = q8_f.tensor_data(&t.name).unwrap();
            assert_eq!(
                tq.quant_type, 3,
                "{}: twin is not Q8_0 (qt {})",
                t.name, tq.quant_type
            );
            assert_eq!(
                relayout(qb, m, k),
                ib.to_vec(),
                "{}: qt59 is not the Q8_0 relayout",
                t.name
            );
            for &n in &[1usize, 77, 512] {
                check(
                    &mut gpu,
                    &format!("real:{}", &t.name[t.name.len().saturating_sub(24)..]),
                    qb,
                    ib,
                    m,
                    k,
                    n,
                    &mut rng,
                    n == 1,
                );
                cases += 1;
            }
        }
    }
    eprintln!("iu8g32 wmma: {cases} cases, worst rel vs f64 = {worst:.3e}");
}
