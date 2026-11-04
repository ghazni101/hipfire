// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the x-batched IU8G128 (qt 59) decode GEMV
//! (`gemv_iu8g32_xbatch`, the multi-slot lm_head route).
//!
//! Every column of the x-batch launch must match running the singleton
//! `gemv_iu8g32` on that column. The files compile separately, so the FMA
//! contraction can differ: the contract is f32 rounding (max relative error
//! < 1e-6 against the singleton's own scale), not bit identity. Determinism
//! (two runs are bit-identical) IS required.
//!
//!   cargo test -p hipfire-runtime --release --test iu8g32_xbatch_gemv_gpu -- --ignored --nocapture

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
    fn i8(&mut self) -> i8 {
        (self.next() >> 33) as i8
    }
}

/// Compact qt 59 row: `[K signed codes][K/128 f16 scales]`.
fn iu8_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let row_bytes = k + k / 64;
    let mut out = vec![0u8; m * row_bytes];
    for row in out.chunks_mut(row_bytes) {
        let (codes, scales) = row.split_at_mut(k);
        for c in codes.iter_mut() {
            *c = rng.i8() as u8;
        }
        for s in scales.chunks_mut(2) {
            let sc = 0.002 + 0.03 * rng.f().abs();
            s.copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
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
/// magnitude (the metric the IU GEMV gates use).
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
    let iu = iu8_random(rng, m, k);
    let w = gpu.upload_raw(&iu, &[iu.len()]).unwrap();
    let x: Vec<f32> = (0..batch * k).map(|_| rng.f() * 2.0).collect();
    let xd = gpu.upload_f32(&x, &[batch * k]).unwrap();

    let y_single = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    let y_batch = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    let y_again = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
    for b in 0..batch {
        let xb = xd.sub_offset(b * k, k);
        let yb = y_single.sub_offset(b * m, m);
        gpu.gemv_iu8g32(&w, &xb, &yb, m, k).unwrap();
    }
    gpu.gemv_iu8g32_xbatch(&w, &xd, &y_batch, m, k, batch)
        .unwrap();
    gpu.gemv_iu8g32_xbatch(&w, &xd, &y_again, m, k, batch)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let single = gpu.download_f32(&y_single).unwrap();
    let batched = gpu.download_f32(&y_batch).unwrap();
    let again = gpu.download_f32(&y_again).unwrap();
    assert!(
        single.iter().all(|v| v.is_finite()),
        "{label} m={m} k={k} b={batch}: non-finite singleton output"
    );
    assert!(
        batched.iter().all(|v| v.is_finite()),
        "{label} m={m} k={k} b={batch}: non-finite x-batch output"
    );
    let rel = max_rel(&batched, &single);
    let diff = bits_differ(&batched, &again);
    let bit = bits_differ(&batched, &single);
    eprintln!(
        "{label:<10} m={m:<6} k={k:<5} b={batch}  max_rel={rel:.3e}  \
         nondeterministic_words={diff}  differing_vs_singleton={bit}"
    );
    assert_eq!(diff, 0, "{label}: x-batch is not deterministic");
    assert!(
        single.iter().any(|v| *v != 0.0),
        "{label}: all-zero output is not a check"
    );
    assert!(
        rel < 1e-6,
        "{label} m={m} k={k} b={batch}: max_rel {rel:.3e} exceeds 1e-6"
    );
    gpu.free_tensor(y_single).unwrap();
    gpu.free_tensor(y_batch).unwrap();
    gpu.free_tensor(y_again).unwrap();
    gpu.free_tensor(w).unwrap();
    gpu.free_tensor(xd).unwrap();
}

#[test]
#[ignore = "GPU required: run with --ignored on a gfx11/gfx12 host"]
fn iu8g32_head_shape_cost() {
    // The 27B IU4-Pro lm_head shape: one weight pass = 1.29 GB.
    let (m, k) = (248_320usize, 5120usize);
    let mut gpu = Gpu::init_with_device(0).expect("GPU 0");
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let w = gpu
        .upload_raw(&iu8_random(&mut rng, m, k), &[m * (k + k / 64)])
        .unwrap();
    let x: Vec<f32> = (0..4 * k).map(|_| rng.f()).collect();
    let xd = gpu.upload_f32(&x, &[4 * k]).unwrap();
    let y = gpu.alloc_tensor(&[4 * m], DType::F32).unwrap();
    fn time(gpu: &mut Gpu, label: &str, bytes: f64, mut f: impl FnMut(&mut Gpu)) {
        f(gpu);
        gpu.hip.device_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
        eprintln!("{label:<28} {ms:8.3} ms   {:.1} GB/s", bytes / (ms * 1e6));
    }
    let bytes = (m as f64) * ((k + k / 64) as f64);
    for b in 1..=4usize {
        let xb = xd.sub_offset(0, b * k);
        let yb = y.sub_offset(0, b * m);
        time(&mut gpu, &format!("xbatch B={b}"), bytes, |gpu| {
            gpu.gemv_iu8g32_xbatch(&w, &xb, &yb, m, k, b).unwrap()
        });
    }
    for b in 1..=4usize {
        time(
            &mut gpu,
            &format!("per-slot B={b}"),
            bytes * b as f64,
            |gpu| {
                for s in 0..b {
                    let xs = xd.sub_offset(s * k, k);
                    let ys = y.sub_offset(s * m, m);
                    gpu.gemv_iu8g32(&w, &xs, &ys, m, k).unwrap();
                }
            },
        );
    }
}

#[test]
#[ignore = "GPU required: run with --ignored on a gfx11/gfx12 host"]
fn iu8g32_xbatch_matches_singleton() {
    let mut gpu = Gpu::init_with_device(0).expect("GPU 0");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for &(m, k) in &[
        (1usize, 1024usize),
        (7, 1024),
        (129, 1024),
        (4097, 1024),
        (32768, 1024),
        (5120, 5120),
        (37, 6144),
    ] {
        for batch in 1..=4 {
            check(&mut gpu, "random", m, k, batch, &mut rng);
        }
    }

    // A compact row whose stride is not 16 B aligned (`K = 256` → 260 B) has no
    // valid `uint4` path: the launcher must refuse rather than misread.
    let w = gpu
        .upload_raw(&iu8_random(&mut rng, 4, 256), &[4 * 260])
        .unwrap();
    let x = gpu.upload_f32(&vec![0.0; 256], &[256]).unwrap();
    let y = gpu.alloc_tensor(&[4], DType::F32).unwrap();
    let err = gpu.gemv_iu8g32_xbatch(&w, &x, &y, 4, 256, 1).unwrap_err();
    assert!(
        err.message.contains("16 B-aligned"),
        "misaligned row must be refused, got {err:?}"
    );
}
