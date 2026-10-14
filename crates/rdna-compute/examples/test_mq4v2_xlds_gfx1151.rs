// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Exact-gfx1151 MQ4G256V2 small-batch WMMA `*_gfx1151_xlds` parity + timing.
//!
//! For gate_up / qkvza / qkv / residual at the 27B shapes (plus odd shapes with
//! row counts that are not multiples of 64) and N in {1, 5, 16}: deterministic
//! random MQ4G256V2 weights (136 B groups, f16 `[s0,z0,s1,z1]` headers, random
//! nibble bytes) and F32 X are run through the base kernels (the public
//! dispatcher with `flags.gfx1151_xlds = false`) and through the direct
//! `*_gfx1151_xlds` launchers into separate outputs. Every output element is
//! compared by raw f32 bits (hard gate; exit code 1 on any mismatch). Both
//! variants are then timed (20 launches after warmup, device sync).
//! Any arch other than exact gfx1151 SKIPs cleanly (exit 0).

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{Gpu, GpuTensor};
use std::sync::Arc;
use std::time::Instant;

const GROUP_BYTES: usize = 136;
const REPS: usize = 20;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Random MQ4G256V2 weights: `m` rows x `k/256` groups of 136 B.
fn random_mq4v2(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for g in blob.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let s = half_from_f32(rng.range(0.002, 0.03));
            let z = half_from_f32(rng.range(-0.25, 0.05));
            g[h * 4..h * 4 + 2].copy_from_slice(&s.to_le_bytes());
            g[h * 4 + 2..h * 4 + 4].copy_from_slice(&z.to_le_bytes());
        }
        let mut i = 8;
        while i < GROUP_BYTES {
            let r = rng.next().to_le_bytes();
            let n = (GROUP_BYTES - i).min(8);
            g[i..i + n].copy_from_slice(&r[..n]);
            i += n;
        }
    }
    blob
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload f32")
}

fn compare(family: &str, shape: &str, n: usize, got: &[Vec<f32>], want: &[Vec<f32>]) -> bool {
    let mut mism = 0usize;
    let mut total = 0usize;
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.len(), w.len());
        total += g.len();
        mism += g
            .iter()
            .zip(w.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    let ok = mism == 0 && total > 0;
    println!(
        "{} {family} {shape} N={n} mismatches={mism} (of {total})",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn set_flag(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).gfx1151_xlds = on;
}

fn timed(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu), weight_bytes: usize) -> (f64, f64) {
    f(gpu);
    f(gpu);
    gpu.hip.device_synchronize().expect("sync warmup");
    let t0 = Instant::now();
    for _ in 0..REPS {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync timed");
    let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
    let gbs = weight_bytes as f64 / (us * 1e-6) / 1e9;
    (us, gbs)
}

fn report(family: &str, shape: &str, n: usize, base: (f64, f64), xlds: (f64, f64)) {
    println!(
        "  TIME {family} {shape} N={n}: base {:.1} us {:.1} GB/s | xlds {:.1} us {:.1} GB/s",
        base.0, base.1, xlds.0, xlds.1
    );
}

fn weight_bytes(rows: &[usize], k: usize) -> usize {
    rows.iter().map(|m| m * (k / 256) * GROUP_BYTES).sum()
}

/// Output tensors for `rows`, N batch rows each, filled with a NaN sentinel.
fn alloc_outs(gpu: &mut Gpu, rows: &[usize], n: usize) -> Vec<GpuTensor> {
    rows.iter()
        .map(|&m| upload_f32(gpu, &vec![f32::from_bits(0x7FC0_00FF); n * m]))
        .collect()
}

fn fetch(gpu: &Gpu, ts: &[GpuTensor]) -> Vec<Vec<f32>> {
    gpu.hip.device_synchronize().expect("sync");
    ts.iter()
        .map(|t| gpu.download_f32(t).expect("download"))
        .collect()
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let arch = gpu.arch.clone();
    if !(gpu.arch_caps.is_gfx1151() && arch == "gfx1151") {
        eprintln!("SKIP: arch {arch} is not exact gfx1151");
        return;
    }
    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some");
        return;
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let ns = [1usize, 5, 16];
    let mut all_ok = true;

    // gate_up
    let gate_up_cases: [(usize, usize, usize); 2] = [(17408, 17408, 5120), (1000, 1048, 2560)];
    for &(gm, um, k) in &gate_up_cases {
        let shape = format!("({gm},{um},K={k})");
        let a_gate = gpu
            .upload_raw(
                &random_mq4v2(&mut rng, gm, k),
                &[gm * (k / 256) * GROUP_BYTES],
            )
            .expect("up gate");
        let a_up = gpu
            .upload_raw(
                &random_mq4v2(&mut rng, um, k),
                &[um * (k / 256) * GROUP_BYTES],
            )
            .expect("up up");
        for &n in &ns {
            let xh: Vec<f32> = (0..n * k).map(|_| rng.range(-2.0, 2.0)).collect();
            let x = upload_f32(&mut gpu, &xh);
            let rows = [gm, um];
            let ref_y = alloc_outs(&mut gpu, &rows, n);
            let got_y = alloc_outs(&mut gpu, &rows, n);
            set_flag(&mut gpu, false);
            gpu.gemm_gate_up_mq4g256v2_wmma(&a_gate, &a_up, &x, &ref_y[0], &ref_y[1], gm, um, k, n)
                .expect("base gate_up");
            gpu.gemm_gate_up_mq4g256v2_wmma_gfx1151_xlds(
                &a_gate, &a_up, &x, &got_y[0], &got_y[1], gm, um, k, n,
            )
            .expect("xlds gate_up");
            let ok = compare(
                "gate_up",
                &shape,
                n,
                &fetch(&gpu, &got_y),
                &fetch(&gpu, &ref_y),
            );
            all_ok &= ok;
            let wb = weight_bytes(&rows, k);
            let base = timed(
                &mut gpu,
                |g| {
                    g.gemm_gate_up_mq4g256v2_wmma(
                        &a_gate, &a_up, &x, &ref_y[0], &ref_y[1], gm, um, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            let xl = timed(
                &mut gpu,
                |g| {
                    g.gemm_gate_up_mq4g256v2_wmma_gfx1151_xlds(
                        &a_gate, &a_up, &x, &got_y[0], &got_y[1], gm, um, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            report("gate_up", &shape, n, base, xl);
        }
    }

    // qkvza
    {
        let (qm, zm, bm, am, k) = (10240usize, 6144usize, 48usize, 48usize, 5120usize);
        let shape = format!("({qm},{zm},{bm},{am},K={k})");
        let rows = [qm, zm, bm, am];
        let w: Vec<GpuTensor> = rows
            .iter()
            .map(|&m| {
                gpu.upload_raw(
                    &random_mq4v2(&mut rng, m, k),
                    &[m * (k / 256) * GROUP_BYTES],
                )
                .expect("up w")
            })
            .collect();
        for &n in &ns {
            let xh: Vec<f32> = (0..n * k).map(|_| rng.range(-2.0, 2.0)).collect();
            let x = upload_f32(&mut gpu, &xh);
            let ref_y = alloc_outs(&mut gpu, &rows, n);
            let got_y = alloc_outs(&mut gpu, &rows, n);
            set_flag(&mut gpu, false);
            gpu.gemm_qkvza_mq4g256v2_wmma(
                &w[0], &w[1], &w[2], &w[3], &x, &ref_y[0], &ref_y[1], &ref_y[2], &ref_y[3], qm, zm,
                bm, am, k, n,
            )
            .expect("base qkvza");
            gpu.gemm_qkvza_mq4g256v2_wmma_gfx1151_xlds(
                &w[0], &w[1], &w[2], &w[3], &x, &got_y[0], &got_y[1], &got_y[2], &got_y[3], qm, zm,
                bm, am, k, n,
            )
            .expect("xlds qkvza");
            let ok = compare(
                "qkvza",
                &shape,
                n,
                &fetch(&gpu, &got_y),
                &fetch(&gpu, &ref_y),
            );
            all_ok &= ok;
            let wb = weight_bytes(&rows, k);
            let base = timed(
                &mut gpu,
                |g| {
                    g.gemm_qkvza_mq4g256v2_wmma(
                        &w[0], &w[1], &w[2], &w[3], &x, &ref_y[0], &ref_y[1], &ref_y[2], &ref_y[3],
                        qm, zm, bm, am, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            let xl = timed(
                &mut gpu,
                |g| {
                    g.gemm_qkvza_mq4g256v2_wmma_gfx1151_xlds(
                        &w[0], &w[1], &w[2], &w[3], &x, &got_y[0], &got_y[1], &got_y[2], &got_y[3],
                        qm, zm, bm, am, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            report("qkvza", &shape, n, base, xl);
        }
    }

    // qkv
    {
        let (qm, km, vm, k) = (12288usize, 1024usize, 1024usize, 5120usize);
        let shape = format!("({qm},{km},{vm},K={k})");
        let rows = [qm, km, vm];
        let w: Vec<GpuTensor> = rows
            .iter()
            .map(|&m| {
                gpu.upload_raw(
                    &random_mq4v2(&mut rng, m, k),
                    &[m * (k / 256) * GROUP_BYTES],
                )
                .expect("up w")
            })
            .collect();
        for &n in &ns {
            let xh: Vec<f32> = (0..n * k).map(|_| rng.range(-2.0, 2.0)).collect();
            let x = upload_f32(&mut gpu, &xh);
            let ref_y = alloc_outs(&mut gpu, &rows, n);
            let got_y = alloc_outs(&mut gpu, &rows, n);
            set_flag(&mut gpu, false);
            gpu.gemm_qkv_mq4g256v2_wmma(
                &w[0], &w[1], &w[2], &x, &ref_y[0], &ref_y[1], &ref_y[2], qm, km, vm, k, n,
            )
            .expect("base qkv");
            gpu.gemm_qkv_mq4g256v2_wmma_gfx1151_xlds(
                &w[0], &w[1], &w[2], &x, &got_y[0], &got_y[1], &got_y[2], qm, km, vm, k, n,
            )
            .expect("xlds qkv");
            let ok = compare("qkv", &shape, n, &fetch(&gpu, &got_y), &fetch(&gpu, &ref_y));
            all_ok &= ok;
            let wb = weight_bytes(&rows, k);
            let base = timed(
                &mut gpu,
                |g| {
                    g.gemm_qkv_mq4g256v2_wmma(
                        &w[0], &w[1], &w[2], &x, &ref_y[0], &ref_y[1], &ref_y[2], qm, km, vm, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            let xl = timed(
                &mut gpu,
                |g| {
                    g.gemm_qkv_mq4g256v2_wmma_gfx1151_xlds(
                        &w[0], &w[1], &w[2], &x, &got_y[0], &got_y[1], &got_y[2], qm, km, vm, k, n,
                    )
                    .unwrap()
                },
                wb,
            );
            report("qkv", &shape, n, base, xl);
        }
    }

    // residual (Y += W@X)
    let residual_cases: [(usize, usize); 4] =
        [(5120, 17408), (5120, 6144), (248320, 5120), (1000, 1536)];
    for &(m, k) in &residual_cases {
        let shape = format!("(M={m},K={k})");
        let a = gpu
            .upload_raw(
                &random_mq4v2(&mut rng, m, k),
                &[m * (k / 256) * GROUP_BYTES],
            )
            .expect("up w");
        for &n in &ns {
            let xh: Vec<f32> = (0..n * k).map(|_| rng.range(-2.0, 2.0)).collect();
            let x = upload_f32(&mut gpu, &xh);
            let yh: Vec<f32> = (0..n * m).map(|_| rng.range(-2.0, 2.0)).collect();
            let y_ref = upload_f32(&mut gpu, &yh);
            let y_got = upload_f32(&mut gpu, &yh);
            set_flag(&mut gpu, false);
            gpu.gemm_mq4g256v2_residual_wmma(&a, &x, &y_ref, m, k, n)
                .expect("base residual");
            gpu.gemm_mq4g256v2_residual_wmma_gfx1151_xlds(&a, &x, &y_got, m, k, n)
                .expect("xlds residual");
            let ok = compare(
                "residual",
                &shape,
                n,
                &fetch(&gpu, std::slice::from_ref(&y_got)),
                &fetch(&gpu, std::slice::from_ref(&y_ref)),
            );
            all_ok &= ok;
            let wb = weight_bytes(&[m], k);
            // Timing accumulates into scratch outputs (Y += W@X), not the checked ones.
            let y_t = upload_f32(&mut gpu, &yh);
            let base = timed(
                &mut gpu,
                |g| {
                    g.gemm_mq4g256v2_residual_wmma(&a, &x, &y_t, m, k, n)
                        .unwrap()
                },
                wb,
            );
            let xl = timed(
                &mut gpu,
                |g| {
                    g.gemm_mq4g256v2_residual_wmma_gfx1151_xlds(&a, &x, &y_t, m, k, n)
                        .unwrap()
                },
                wb,
            );
            report("residual", &shape, n, base, xl);
        }
    }

    if all_ok {
        println!("ALL PASS");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
