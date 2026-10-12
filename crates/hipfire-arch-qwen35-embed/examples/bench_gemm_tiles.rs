// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sweep every compiled LDS macro-tile for the attention-shaped GEMM.
//!
//! `gemm_f16_x_f16_wmma_lds_auto` picks its tile per arch from measurement at
//! *projection* shapes (large K). QK^T has K = head_dim = 256, where the K-loop
//! is only a handful of stages and the auto pick may not be the best choice.
//! This walks `Gpu::LDS_TILE_VARIANTS` and reports throughput per tile, so the
//! attention-as-GEMMs design can use the right one instead of the default.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example bench_gemm_tiles --features lab -- [n] [hd] [iters]

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let hd: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let iters: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(20);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    println!("gpu {} | [M={n}, K={hd}, B={n}]", gpu.arch);

    let mut s: u32 = 12345;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        half::f16::from_f32(((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5).to_bits()
    };
    let w_words: Vec<u16> = (0..n * hd).map(|_| rnd()).collect();
    let x_words: Vec<u16> = (0..n * hd).map(|_| rnd()).collect();
    let w = gpu.upload_f16_bits(&w_words, &[n, hd]).map_err(|e| format!("w: {e:?}"))?;
    let x = gpu.upload_f16_bits(&x_words, &[n, hd]).map_err(|e| format!("x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[n, n], DType::F32).map_err(|e| format!("y: {e:?}"))?;
    let flops = 2.0 * (n as f64) * (n as f64) * (hd as f64);

    let mut best: Option<(String, f64)> = None;
    // The default first, as the baseline.
    let mut results: Vec<(String, f64)> = Vec::new();
    {
        for _ in 0..3 {
            gpu.gemm_f16_x_f16_wmma_lds_auto(&w, &x, &y, None, n, hd, n)
                .map_err(|e| format!("auto: {e:?}"))?;
        }
        let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_f16_x_f16_wmma_lds_auto(&w, &x, &y, None, n, hd, n)
                .map_err(|e| format!("auto: {e:?}"))?;
        }
        let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "  {:<34} {ms:7.3} ms  {:6.2} TFLOP/s",
            "auto (default)",
            flops / (ms * 1e-3) / 1e12
        );
        results.push(("auto (default)".into(), ms));
    }

    for tile in rdna_compute::Gpu::LDS_TILE_VARIANTS {
        if tile.bm > n || tile.bn > n {
            continue;
        }
        let name = format!("{}", tile.entry());
        let mut ok = true;
        for _ in 0..2 {
            if gpu.gemm_f16_x_f16_wmma_lds_tiled(&w, &x, &y, None, n, hd, n, *tile).is_err() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_f16_x_f16_wmma_lds_tiled(&w, &x, &y, None, n, hd, n, *tile)
                .map_err(|e| format!("tiled: {e:?}"))?;
        }
        let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("  {name:<34} {ms:7.3} ms  {:6.2} TFLOP/s", flops / (ms * 1e-3) / 1e12);
        if best.as_ref().is_none_or(|(_, b)| ms < *b) {
            best = Some((name.clone(), ms));
        }
        results.push((name, ms));
    }

    if let Some((name, ms)) = best {
        println!("\n  best: {name}  {ms:.3} ms  (8 heads = {:.1} ms; today's attention is 66.7 ms/layer)", 8.0 * ms);
    }
    Ok(())
}
