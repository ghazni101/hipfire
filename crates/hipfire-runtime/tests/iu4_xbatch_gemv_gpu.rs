// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the x-batched IU4G256 decode GEMV
//! (`gemv_iu4g256_xbatch`, `gemv_iu4g256_xbatch_residual`).
//!
//! Every column of the x-batch launch must match running the singleton
//! `gemv_iu4g256` / `gemv_iu4g256_residual` on that column. The files compile
//! separately, so the FMA contraction can differ: the contract is f32
//! rounding (max relative error < 1e-6 against the singleton's own scale),
//! not bit identity. Determinism (two runs are bit-identical) IS required.
//!
//!   cargo test -p hipfire-runtime --release --test iu4_xbatch_gemv_gpu -- --ignored --nocapture

use half::f16;
use rdna_compute::{DType, Gpu};

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
    fn u8(&mut self) -> u8 {
        self.next() as u8
    }
}

/// Planar qt 57 row block: `[g*128 codes][g*8 headers]`.
fn iu4_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let g = k / 256;
    let mut out = vec![0u8; m * g * 136];
    for row in out.chunks_mut(g * 136) {
        let (codes, hdr) = row.split_at_mut(g * 128);
        for c in codes.iter_mut() {
            *c = rng.u8();
        }
        for h in hdr.chunks_mut(8) {
            for half_slot in 0..2 {
                let sc = 0.002 + 0.03 * rng.f().abs();
                let zp = -8.0 * sc + 0.01 * rng.f();
                h[half_slot * 4..half_slot * 4 + 2]
                    .copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
                h[half_slot * 4 + 2..half_slot * 4 + 4]
                    .copy_from_slice(&f16::from_f32(zp).to_bits().to_le_bytes());
            }
        }
    }
    out
}

fn bits_differ(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Max relative error of `got` against `want`, in units of `want`'s largest
/// magnitude (the metric the IU4 GEMV gates use).
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    let scale = want
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs() as f64))
        .max(1e-12);
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max)
        / scale
}

fn check(gpu: &mut Gpu, label: &str, m: usize, k: usize, batch: usize, rng: &mut Rng) {
    let iu = iu4_random(rng, m, k);
    let w = gpu.upload_raw(&iu, &[iu.len()]).unwrap();
    let x: Vec<f32> = (0..batch * k).map(|_| rng.f() * 2.0).collect();
    let xd = gpu.upload_f32(&x, &[batch * k]).unwrap();

    // Plain: singleton per column vs one x-batch launch.
    let y_single = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    let y_batch = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    let y_again = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    for b in 0..batch {
        let xb = xd.sub_offset(b * k, k);
        let yb = y_single.sub_offset(b * m, m);
        gpu.gemv_iu4g256(&w, &xb, &yb, m, k).unwrap();
    }
    gpu.gemv_iu4g256_xbatch(&w, &xd, &y_batch, m, k, batch, false)
        .unwrap();
    gpu.gemv_iu4g256_xbatch(&w, &xd, &y_again, m, k, batch, false)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let single = gpu.download_f32(&y_single).unwrap();
    let batched = gpu.download_f32(&y_batch).unwrap();
    let again = gpu.download_f32(&y_again).unwrap();
    let rel = max_rel(&batched, &single);
    assert_eq!(
        bits_differ(&batched, &again),
        0,
        "{label}: x-batch plain is not deterministic"
    );
    assert!(
        single.iter().any(|v| *v != 0.0),
        "{label}: all-zero output is not a check"
    );
    eprintln!(
        "{label} plain   m={m} k={k} b={batch} bitdiff={} max_rel={rel:.3e}",
        bits_differ(&single, &batched)
    );
    assert!(rel < 1e-6, "{label}: x-batch plain rel {rel:.3e}");

    // Residual: same seeds, singleton per column vs one x-batch launch.
    let seed: Vec<f32> = (0..batch * m)
        .map(|i| 0.37 * ((i % 11) as f32) - 1.1)
        .collect();
    let r_single = gpu.upload_f32(&seed, &[batch * m]).unwrap();
    let r_batch = gpu.upload_f32(&seed, &[batch * m]).unwrap();
    for b in 0..batch {
        let xb = xd.sub_offset(b * k, k);
        let yb = r_single.sub_offset(b * m, m);
        gpu.gemv_iu4g256_residual(&w, &xb, &yb, m, k).unwrap();
    }
    gpu.gemv_iu4g256_xbatch(&w, &xd, &r_batch, m, k, batch, true)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let single = gpu.download_f32(&r_single).unwrap();
    let batched = gpu.download_f32(&r_batch).unwrap();
    let rel = max_rel(&batched, &single);
    eprintln!(
        "{label} residual m={m} k={k} b={batch} bitdiff={} max_rel={rel:.3e}",
        bits_differ(&single, &batched)
    );
    assert!(rel < 1e-6, "{label}: x-batch residual rel {rel:.3e}");
}

#[test]
#[ignore = "requires an AMD GPU"]
fn iu4_xbatch_matches_singleton_gemv() {
    let mut gpu = Gpu::init().expect("GPU required");
    let mut rng = Rng(0x9e3779b97f4a7c15);
    // Every G mod 4 tail, all admitted batch widths, decode shapes plus the
    // model's real projection shapes.
    for &(m, k) in &[
        (1, 256),
        (7, 512),
        (129, 768),
        (17, 2560),
        (256, 9216),
        (4097, 1024),
        (5120, 5120),
        (17408, 5120),
        (5120, 6144),
    ] {
        for batch in 1..=4 {
            check(&mut gpu, "rand", m, k, batch, &mut rng);
        }
    }
}
