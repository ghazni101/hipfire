// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU parity for the IU quant family's decode readers: qt 57
//! (`IU4G256`), qt 58 (`IU8G256`), qt 59 (`IU8G32`), each with its fused
//! `y += W·x` twin, plus the qt 59 embedding gather.
//!
//! The CPU contracts are `hipfire_runtime::iu_quant`:
//! `gemv_iu4_dot8` (int4 activation, integer dot, per-128 `(scale, zero)`),
//! `gemv_iu8_dot` (byte plane, same affine grid, f32), and `gemv_q8` (planar
//! Q8 grid, natural basis). A wrong row stride, a scale read at the wrong
//! offset, a nibble-order flip or a sign error lands orders of magnitude
//! outside the tolerance; the kernels' warp reduction explains the last bits.
//!
//! The activation here is an arbitrary vector: the kernels do not rotate, the
//! model's rotation step is upstream, so layout/sign/scale are what this
//! proves.
//!
//! `#[ignore]`d: needs an RDNA GPU. Run:
//!
//!   cargo test -p hipfire-runtime --release --test iu_quant_gemv_gpu -- --ignored --nocapture

use half::f16;
use hipfire_runtime::iu_quant::{
    dequant_q8, gemv_iu4_dot8, gemv_iu8_dot, gemv_q8, repack_mq4v2, repack_q8f16, Layout,
    Q8_BLOCK_BYTES,
};
use rdna_compute::{DType, Gpu};

const TOL: f32 = 1e-4;

/// Synthetic MQ4V2 blob: `136 B = [s0, z0, s1, z1] f16 + 128 nibble bytes` per
/// 256 weights.
fn mq4v2_blob(m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let mut src = vec![0u8; m * gpr * 136];
    for row in 0..m {
        for g in 0..gpr {
            let base = (row * gpr + g) * 136;
            let s0 = 0.021 + 0.004 * ((row + g) % 5) as f32;
            let s1 = 0.017 + 0.003 * ((row + g) % 3) as f32;
            for (i, (scale, zero)) in [(s0, -8.0 * s0), (s1, -8.0 * s1)].iter().enumerate() {
                let at = base + i * 4;
                src[at..at + 2].copy_from_slice(&f16::from_f32(*scale).to_bits().to_le_bytes());
                src[at + 2..at + 4].copy_from_slice(&f16::from_f32(*zero).to_bits().to_le_bytes());
            }
            for i in 0..128 {
                let lo = ((row * 7 + g * 3 + i) % 16) as u8;
                let hi = ((row * 5 + g * 11 + i * 3) % 16) as u8;
                src[base + 8 + i] = lo | (hi << 4);
            }
        }
    }
    src
}

fn q8f16_blob(m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 32;
    let mut src = vec![0u8; m * gpr * Q8_BLOCK_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let base = (row * gpr + g) * Q8_BLOCK_BYTES;
            let scale = 0.011 + 0.003 * ((row + g) % 7) as f32;
            src[base..base + 2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for i in 0..32 {
                let code = (((row * 37 + g * 11 + i * 5) % 255) as i32) - 127;
                src[base + 2 + i] = code as i8 as u8;
            }
        }
    }
    src
}

fn activation(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i as f32) * 0.017).sin() * 1.7 + ((i * 7) as f32 * 0.003).cos())
        .collect()
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    max_abs / scale
}

/// Seed for the residual arm so `y += W·x` is not tested with zeroes.
fn seed(m: usize) -> Vec<f32> {
    (0..m).map(|i| 0.5 * i as f32 - 0.25).collect()
}

#[test]
#[ignore]
fn iu_quant_gemv_and_gather_match_the_cpu_contract() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let m = 5usize;
    // K must be a multiple of 256 for the FWHT-256 layouts (qt 57/58).
    for k in [256usize, 512, 3072] {
        let src = mq4v2_blob(m, k);
        let x = activation(k);

        // ── qt 57 IU4G256 ────────────────────────────────────────────────
        let iu4 = repack_mq4v2(&src, m, k, Layout::Iu4).expect("repack iu4");
        let want = gemv_iu4_dot8(&iu4, &x, m, k).expect("cpu iu4");
        let (w, xd) = (
            gpu.upload_raw(&iu4, &[iu4.len()]).expect("upload iu4"),
            gpu.upload_f32(&x, &[k]).expect("upload x"),
        );
        let yd = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
        gpu.gemv_iu4g256(&w, &xd, &yd, m, k).expect("iu4 gemv");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yd).expect("download y");
        let rel = max_rel(&got, &want);
        eprintln!("iu4 gemv      k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "iu4 gemv k={k}: rel {rel:.3e} > {TOL:.0e}");

        let s = seed(m);
        let yr = gpu.upload_f32(&s, &[m]).expect("upload seed");
        gpu.gemv_iu4g256_residual(&w, &xd, &yr, m, k)
            .expect("iu4 residual");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yr).expect("download yr");
        let want_r: Vec<f32> = want.iter().zip(&s).map(|(a, b)| a + b).collect();
        let rel = max_rel(&got, &want_r);
        eprintln!("iu4 residual  k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "iu4 residual k={k}: rel {rel:.3e} > {TOL:.0e}");
        for t in [w, xd, yd, yr] {
            gpu.free_tensor(t).ok();
        }

        // ── qt 58 IU8G256 ───────────────────────────────────────────────
        let iu8 = repack_mq4v2(&src, m, k, Layout::Iu8).expect("repack iu8");
        let want = gemv_iu8_dot(&iu8, &x, m, k).expect("cpu iu8");
        let (w, xd) = (
            gpu.upload_raw(&iu8, &[iu8.len()]).expect("upload iu8"),
            gpu.upload_f32(&x, &[k]).expect("upload x"),
        );
        let yd = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
        gpu.gemv_iu8g256(&w, &xd, &yd, m, k).expect("iu8 gemv");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yd).expect("download y");
        let rel = max_rel(&got, &want);
        eprintln!("iu8 gemv      k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "iu8 gemv k={k}: rel {rel:.3e} > {TOL:.0e}");

        let s = seed(m);
        let yr = gpu.upload_f32(&s, &[m]).expect("upload seed");
        gpu.gemv_iu8g256_residual(&w, &xd, &yr, m, k)
            .expect("iu8 residual");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yr).expect("download yr");
        let want_r: Vec<f32> = want.iter().zip(&s).map(|(a, b)| a + b).collect();
        let rel = max_rel(&got, &want_r);
        eprintln!("iu8 residual  k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "iu8 residual k={k}: rel {rel:.3e} > {TOL:.0e}");
        for t in [w, xd, yd, yr] {
            gpu.free_tensor(t).ok();
        }
    }

    // ── qt 59 IU8G32 + gather ────────────────────────────────────────────
    for k in [32usize, 256, 2688] {
        let src = q8f16_blob(m, k);
        let packed = repack_q8f16(&src, m, k).expect("repack");
        let x = activation(k);

        let w = gpu.upload_raw(&packed, &[packed.len()]).expect("upload w");
        let xd = gpu.upload_f32(&x, &[k]).expect("upload x");
        let yd = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
        gpu.gemv_iu8g32(&w, &xd, &yd, m, k).expect("gemv");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yd).expect("download y");
        let want = gemv_q8(&packed, &x, m, k).expect("cpu gemv");
        let rel = max_rel(&got, &want);
        eprintln!("iu8g32 gemv     k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "gemv k={k}: rel {rel:.3e} > {TOL:.0e}");

        let s = seed(m);
        let yr = gpu.upload_f32(&s, &[m]).expect("upload seed");
        gpu.gemv_iu8g32_residual(&w, &xd, &yr, m, k)
            .expect("residual");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yr).expect("download yr");
        let want_r: Vec<f32> = want.iter().zip(&s).map(|(a, b)| a + b).collect();
        let rel = max_rel(&got, &want_r);
        eprintln!("iu8g32 resid    k={k:<5} rel={rel:.3e}");
        assert!(rel <= TOL, "residual k={k}: rel {rel:.3e} > {TOL:.0e}");

        let tok = 3usize;
        let table = gpu
            .upload_raw(&packed, &[packed.len()])
            .expect("upload table");
        let out = gpu.alloc_tensor(&[k], DType::F32).expect("alloc row");
        gpu.embedding_lookup_iu8g32(&table, &out, tok as u32, k)
            .expect("gather");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&out).expect("download row");
        let deq = dequant_q8(&packed, m, k).expect("cpu dequant");
        let want = &deq[tok * k..(tok + 1) * k];
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("iu8g32 gather  k={k:<5} max_abs={max_abs:.3e}");
        assert!(max_abs <= 1e-5, "gather k={k}: max_abs {max_abs:.3e}");

        for t in [w, xd, yd, yr, table, out] {
            gpu.free_tensor(t).ok();
        }
    }
}
