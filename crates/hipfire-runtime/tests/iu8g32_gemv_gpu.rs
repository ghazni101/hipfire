// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU parity for the IU8G32 (qt 59) readers: the planar-Q8 decode GEMV
//! (`gemv_iu8g32`), its fused `y += W·x` twin, and the embedding gather.
//!
//! The CPU contract is `hipfire_runtime::iu_quant::gemv_q8` / `dequant_q8`:
//! `w = f32(scale) * f32(i8(code))` with the scales at the row tail and `x` in
//! the natural basis (qt 59 is not FWHT-rotated). The device kernel folds the
//! same product per 32-block and reduces with a warp tree, so this asserts a
//! tolerance, not bit identity — but a wrong row stride, a scale read at the
//! code position (i.e. the interleaved Q8F16 layout), or a sign error lands
//! orders of magnitude outside it.
//!
//! `#[ignore]`d: needs an RDNA GPU. Run:
//!
//!   cargo test -p hipfire-runtime --release --test iu8g32_gemv_gpu -- --ignored --nocapture

use half::f16;
use hipfire_runtime::iu_quant::{dequant_q8, gemv_q8, repack_q8f16, Q8_BLOCK_BYTES};
use rdna_compute::{DType, Gpu};

const TOL: f32 = 1e-4;

fn q8f16_blob(m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 32;
    let mut src = vec![0u8; m * gpr * Q8_BLOCK_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let base = (row * gpr + g) * Q8_BLOCK_BYTES;
            let scale = 0.011 + 0.003 * ((row + g) % 7) as f32;
            src[base..base + 2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for i in 0..32 {
                // Full int8 range: qt 59 is not a nibble grid.
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

#[test]
#[ignore]
fn iu8g32_gemv_and_gather_match_the_cpu_contract() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    for k in [32usize, 256, 2688] {
        let m = 5usize;
        let src = q8f16_blob(m, k);
        let packed = repack_q8f16(&src, m, k).expect("repack");
        let x = activation(k);

        let w = gpu.upload_raw(&packed, &[packed.len()]).expect("upload w");
        let xd = gpu.upload_f32(&x, &[k]).expect("upload x");
        let yd = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
        gpu.gemv_iu8g32(&w, &xd, &yd, m, k).expect("gemv");
        gpu.hip.device_synchronize().expect("sync");
        let gpu_y = gpu.download_f32(&yd).expect("download y");
        let cpu_y = gemv_q8(&packed, &x, m, k).expect("cpu gemv");
        let max_abs = gpu_y
            .iter()
            .zip(&cpu_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = cpu_y.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
        let rel = max_abs / scale;
        eprintln!("iu8g32 gemv m={m} k={k:<5} max_abs={max_abs:.3e} rel={rel:.3e}");
        assert!(rel <= TOL, "gemv k={k}: rel {rel:.3e} > {TOL:.0e}");

        // Residual: y += W·x, in place, one f32 add at the store. With a
        // non-zero seed it must equal set + seed.
        let seed: Vec<f32> = (0..m).map(|i| 0.5 * i as f32 - 0.25).collect();
        let yr = gpu.upload_f32(&seed, &[m]).expect("upload seed");
        gpu.gemv_iu8g32_residual(&w, &xd, &yr, m, k)
            .expect("residual");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&yr).expect("download yr");
        let max_abs = got
            .iter()
            .zip(&cpu_y)
            .zip(&seed)
            .map(|((g, c), s)| (g - (c + s)).abs())
            .fold(0.0f32, f32::max);
        let rel = max_abs / scale;
        eprintln!("iu8g32 residual m={m} k={k:<5} max_abs={max_abs:.3e} rel={rel:.3e}");
        assert!(rel <= TOL, "residual k={k}: rel {rel:.3e} > {TOL:.0e}");

        // Embedding gather: row `tok` must dequantize to the same values.
        let tok = 3usize;
        let dim = k;
        let table = gpu
            .upload_raw(&packed, &[packed.len()])
            .expect("upload table");
        let out = gpu.alloc_tensor(&[dim], DType::F32).expect("alloc row");
        gpu.embedding_lookup_iu8g32(&table, &out, tok as u32, dim)
            .expect("gather");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&out).expect("download row");
        let deq = dequant_q8(&packed, m, k).expect("cpu dequant");
        let want = &deq[tok * dim..(tok + 1) * dim];
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("iu8g32 gather tok={tok} dim={dim} max_abs={max_abs:.3e}");
        assert!(max_abs <= 1e-5, "gather dim={dim}: max_abs {max_abs:.3e}");

        for t in [w, xd, yd, yr, table, out] {
            gpu.free_tensor(t).ok();
        }
    }
}
