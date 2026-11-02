// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the IU4G256 W4A4 prefill route (`HIPFIRE_IU4_PREFILL_ACT`
//! default `a4`) and the small-M MW-LDS route (`HIPFIRE_IU4_SMALLM` default
//! `wmma`).
//!
//! 1. Bit-exact: on `repack_mq4v2(src)`, `gemm_iu4g256_w4a4_prequant` returns
//!    the bits of `gemm_mq4g256v2_mmq_{set,add}_prequant_iu4` on `src` over
//!    the same `block_i4_128` X (full lf16 tiles and guarded tails).
//! 2. Bit-exact: `gemm_iu4g256_mwl_residual` equals the MQ4V2 MW4-LDS
//!    residual on the twin.
//! 3. The routed `gemm_iu4g256` (set and add) vs an f64 reference of the CPU
//!    dequant weights: max rel reported (int4 / f16 activations: loose bound).
//! 4. Determinism: every routed call is run twice and must match bit for bit.
//!
//!   cargo test -p hipfire-runtime --release --test iu4_w4a4_gemm_gpu -- --ignored --nocapture

use half::f16;
use hipfire_runtime::iu_quant::{dequant, repack_mq4v2, Layout};
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

fn mq4v2_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let g = k / 256;
    let mut out = vec![0u8; m * g * 136];
    for blk in out.chunks_mut(136) {
        for h in 0..2 {
            let sc = 0.002 + 0.03 * rng.f().abs();
            let zp = -8.0 * sc + 0.01 * rng.f();
            blk[h * 4..h * 4 + 2].copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
            blk[h * 4 + 2..h * 4 + 4].copy_from_slice(&f16::from_f32(zp).to_bits().to_le_bytes());
        }
        for b in &mut blk[8..] {
            *b = rng.next() as u8;
        }
    }
    out
}

fn bits_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[derive(Default)]
struct Worst {
    a4_rel: f64,
    mwl_rel: f64,
    cases: usize,
}

#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    label: &str,
    mq: &[u8],
    iu: &[u8],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut Rng,
    w: &mut Worst,
) {
    let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
    let seed: Vec<f32> = (0..n * m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let wm = gpu.upload_raw(mq, &[mq.len()]).unwrap();
    let wi = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
    let wd = dequant(iu, m, k, Layout::Iu4).expect("dequant");
    let want: Vec<f64> = (0..n)
        .flat_map(|t| {
            let xr = &x[t * k..(t + 1) * k];
            let wd = &wd;
            (0..m).map(move |r| {
                (0..k)
                    .map(|c| wd[r * k + c] as f64 * xr[c] as f64)
                    .sum::<f64>()
            })
        })
        .collect();
    let sc = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
    for add in [false, true] {
        let up = |gpu: &mut Gpu| gpu.upload_f32(&seed, &[n * m]).unwrap();
        // 1. W4A4: IU4 planar twin vs MQ4V2 over the same block_i4_128 X.
        let (y_mq, y_iu) = (up(gpu), up(gpu));
        let xq4 = gpu.ensure_int4_mmq_x(&xd, n, k).unwrap();
        if add {
            gpu.gemm_mq4g256v2_mmq_add_prequant_iu4(&wm, xq4, &y_mq, m, k, n)
                .unwrap();
        } else {
            gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&wm, xq4, &y_mq, m, k, n)
                .unwrap();
        }
        gpu.gemm_iu4g256_w4a4_prequant(&wi, xq4, &y_iu, m, k, n, add)
            .unwrap();
        // 2. MW-LDS residual twin (Y += W X), seeded.
        let (z_mq, z_iu) = (up(gpu), up(gpu));
        gpu.invalidate_fp16_cache();
        gpu.gemm_mqv2_residual_wmma_gfx11_mw_lds(4, 4, &wm, &xd, &z_mq, m, k, n)
            .unwrap();
        gpu.gemm_iu4g256_mwl_residual(&wi, &xd, &z_iu, m, k, n)
            .unwrap();
        // 3/4. Routed call, twice.
        let (r1, r2) = (up(gpu), up(gpu));
        gpu.gemm_iu4g256(&wi, &xd, &r1, m, k, n, add).unwrap();
        gpu.gemm_iu4g256(&wi, &xd, &r2, m, k, n, add).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let d = |gpu: &mut Gpu, t: &rdna_compute::GpuTensor| gpu.download_f32(t).unwrap();
        let (a, b, za, zb, ra, rb) = (
            d(gpu, &y_mq),
            d(gpu, &y_iu),
            d(gpu, &z_mq),
            d(gpu, &z_iu),
            d(gpu, &r1),
            d(gpu, &r2),
        );
        for t in [y_mq, y_iu, z_mq, z_iu, r1, r2] {
            gpu.free_tensor(t).ok();
        }
        let wv: Vec<f64> = if add {
            want.iter().zip(&seed).map(|(w, s)| w + *s as f64).collect()
        } else {
            want.clone()
        };
        let wz: Vec<f64> = want.iter().zip(&seed).map(|(w, s)| w + *s as f64).collect();
        let rel = |g: &[f32], wv: &[f64]| {
            g.iter()
                .zip(wv)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max)
                / sc
        };
        let (ea4, emw, er) = (rel(&b, &wv), rel(&zb, &wz), rel(&ra, &wv));
        let (d4, dmw, ddet) = (bits_diff(&a, &b), bits_diff(&za, &zb), bits_diff(&ra, &rb));
        let route = if m < 128 { "mwl" } else { "a4" };
        if m < 128 {
            w.mwl_rel = w.mwl_rel.max(er);
        } else {
            w.a4_rel = w.a4_rel.max(er);
        }
        eprintln!(
            "{label:<26} m={m:<6} k={k:<6} n={n:<4} {} a4_bitdiff={d4} a4_rel={ea4:.3e} mwl_bitdiff={dmw} mwl_rel={emw:.3e} routed={route} routed_rel={er:.3e} det_diff={ddet}",
            if add { "add" } else { "set" }
        );
        assert_eq!(
            d4, 0,
            "{label} m={m} k={k} n={n} add={add}: IU4 W4A4 differs from MQ4V2 W4A4"
        );
        assert_eq!(
            dmw, 0,
            "{label} m={m} k={k} n={n}: IU4 MW-LDS differs from MQ4V2 MW-LDS"
        );
        assert_eq!(ddet, 0, "{label}: routed call not deterministic");
        assert!(ea4 < 0.30, "{label} a4 rel {ea4:.3e}");
        assert!(emw < 1e-2, "{label} mwl rel {emw:.3e}");
    }
    w.cases += 1;
    for t in [wm, wi, xd] {
        gpu.free_tensor(t).ok();
    }
}

#[test]
#[ignore]
fn iu4_w4a4_and_mwl_are_bit_exact_to_mq4v2() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu4g256_mmq_eligible(256, 64) {
        eprintln!("SKIP — IU4 MMQ not eligible on this arch / env.");
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut w = Worst::default();
    for &m in &[1usize, 7, 32, 127, 128, 129, 384] {
        for &k in &[256usize, 768, 1280, 5120] {
            let mq = mq4v2_random(&mut rng, m, k);
            let iu = repack_mq4v2(&mq, m, k, Layout::Iu4).unwrap();
            for &n in &[1usize, 2, 63, 64, 65, 128, 130, 512] {
                check(&mut gpu, "random", &mq, &iu, m, k, n, &mut rng, &mut w);
            }
        }
    }
    if let (Ok(iu_path), Ok(mq_path)) = (std::env::var("IU4_REAL"), std::env::var("MQ_REAL")) {
        use hipfire_runtime::hfq::HfqFile;
        let iu_f = HfqFile::open(std::path::Path::new(&iu_path)).unwrap();
        let mq_f = HfqFile::open(std::path::Path::new(&mq_path)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut done = 0;
        for t in iu_f.tensors().to_vec() {
            let (ti, ib) = iu_f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 57 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen.insert((k, m)) || m > 20_000 {
                continue;
            }
            let (_, mb) = mq_f.tensor_data(&t.name).unwrap();
            let label: String = format!(
                "real:{}",
                t.name
                    .chars()
                    .rev()
                    .take(21)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            );
            for &n in &[1usize, 77, 512] {
                check(&mut gpu, &label, mb, ib, m, k, n, &mut rng, &mut w);
            }
            done += 1;
            if done >= 8 {
                break;
            }
        }
    }
    eprintln!(
        "PASS cases={} worst_a4_routed_rel={:.3e} worst_mwl_routed_rel={:.3e}",
        w.cases, w.a4_rel, w.mwl_rel
    );
}
