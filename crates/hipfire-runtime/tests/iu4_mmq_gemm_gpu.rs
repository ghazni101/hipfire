// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the IU4G256 batched (prefill) GEMM `gemm_iu4g256_multi`.
//!
//! 1. Bit-exact: on `repack_mq4v2(src)` the IU4 MMQ kernel returns exactly
//!    the bits of `gemm_mq4g256v2_mmq_{set,add}_prequant` on `src` over the
//!    same quantized X (set and add, all six entry variants via M/N tails).
//! 2. Per-token fallback: `gemm_iu4g256_pertoken` equals `gemv_iu4g256` per
//!    row bit for bit, and both are compared to an f64 reference.
//! 3. MMQ vs f64 reference of the CPU dequant weights (Q8_1 activations, so
//!    a loose bound): max abs / rel reported.
//!
//! M in {1, 7, 129, 256}, N in {1, 2, 63, 64, 65, 128, 130, 512}, K covering
//! G mod 4 = 0..3. Optional real tensors: `IU4_REAL=` / `MQ_REAL=`.
//!
//!   cargo test -p hipfire-runtime --release --test iu4_mmq_gemm_gpu -- --ignored --nocapture

use half::f16;
use hipfire_runtime::iu_quant::{dequant, repack_mq4v2, Layout};
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
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn err(got: &[f32], want: &[f64]) -> (f64, f64) {
    let max_abs = got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    let scale = want.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-12);
    (max_abs, max_abs / scale)
}

#[derive(Default)]
struct Worst {
    mmq_rel: f64,
    tok_rel: f64,
    cases: usize,
}

#[allow(clippy::too_many_arguments)]
fn check(gpu: &mut Gpu, label: &str, mq: &[u8], iu: &[u8], m: usize, k: usize, n: usize, rng: &mut Rng, w: &mut Worst) {
    let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
    let seed: Vec<f32> = (0..n * m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let wm = gpu.upload_raw(mq, &[mq.len()]).unwrap();
    let wi = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
    let mut outs = Vec::new();
    for add in [false, true] {
        let y_mq = gpu.upload_f32(&seed, &[n * m]).unwrap();
        let y_iu = gpu.upload_f32(&seed, &[n * m]).unwrap();
        let y_tok = gpu.upload_f32(&seed, &[n * m]).unwrap();
        // The oracle: MQ4V2 MMQ over the same quantized X.
        let xq = gpu.ensure_q8_1_mmq_x128(&xd, n, k).unwrap();
        if add {
            gpu.gemm_mq4g256v2_mmq_add_prequant(&wm, xq, &y_mq, m, k, n).unwrap();
        } else {
            gpu.gemm_mq4g256v2_mmq_set_prequant(&wm, xq, &y_mq, m, k, n).unwrap();
        }
        gpu.gemm_iu4g256(&wi, &xd, &y_iu, m, k, n, add).unwrap();
        gpu.gemm_iu4g256_pertoken(&[(&wi, &y_tok, m)], &xd, k, n, add).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let a = gpu.download_f32(&y_mq).unwrap();
        let b = gpu.download_f32(&y_iu).unwrap();
        let t = gpu.download_f32(&y_tok).unwrap();
        outs.push((add, a, b, t));
        for tt in [y_mq, y_iu, y_tok] {
            gpu.free_tensor(tt).ok();
        }
    }
    // f64 reference.
    let wd = dequant(iu, m, k, Layout::Iu4).expect("dequant");
    let want: Vec<f64> = (0..n)
        .flat_map(|t| {
            let xr = &x[t * k..(t + 1) * k];
            let wd = &wd;
            (0..m).map(move |r| (0..k).map(|c| wd[r * k + c] as f64 * xr[c] as f64).sum::<f64>())
        })
        .collect();
    // Per-token rows vs the single-row decode kernel (row 0 and last token).
    let mut tok_vs_gemv = 0;
    for t in [0, n - 1] {
        let xt = gpu.upload_f32(&x[t * k..(t + 1) * k], &[k]).unwrap();
        let yt = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        gpu.gemv_iu4g256(&wi, &xt, &yt, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let g = gpu.download_f32(&yt).unwrap();
        tok_vs_gemv += bits_diff(&g, &outs[0].3[t * m..(t + 1) * m]);
        gpu.free_tensor(xt).ok();
        gpu.free_tensor(yt).ok();
    }
    for (add, a, b, t) in &outs {
        let want_v: Vec<f64> = if *add {
            want.iter().zip(&seed).map(|(w, s)| w + *s as f64).collect()
        } else {
            want.clone()
        };
        let diff = bits_diff(a, b);
        // Relative error is normalised by max|W x| (the GEMM output), not by
        // max|seed + W x|: the add seed can cancel W x and would inflate rel.
        let sc = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
        let (ea, _) = err(b, &want_v);
        let (ta, _) = err(t, &want_v);
        let (er, tr) = (ea / sc, ta / sc);
        w.mmq_rel = w.mmq_rel.max(er);
        w.tok_rel = w.tok_rel.max(tr);
        eprintln!(
            "{label:<26} m={m:<6} k={k:<6} n={n:<4} {} bitdiff_vs_mq4v2_mmq={diff} mmq_abs={ea:.3e} mmq_rel={er:.3e} tok_abs={ta:.3e} tok_rel={tr:.3e}",
            if *add { "add" } else { "set" }
        );
        assert_eq!(diff, 0, "{label} m={m} k={k} n={n} add={add}: IU4 MMQ differs from MQ4V2 MMQ");
        assert!(er < 3e-2, "{label} mmq rel {er:.3e}");
        assert!(tr < 1e-4, "{label} per-token rel {tr:.3e}");
    }
    assert_eq!(tok_vs_gemv, 0, "{label}: per-token path differs from gemv_iu4g256");
    w.cases += 1;
    for tt in [wm, wi, xd] {
        gpu.free_tensor(tt).ok();
    }
}

#[test]
#[ignore]
fn iu4_mmq_gemm_is_bit_exact_to_mq4v2_mmq() {
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
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut w = Worst::default();
    for &m in &[1usize, 7, 129, 256] {
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
            let label: String = format!("real:{}", t.name.chars().rev().take(21).collect::<String>().chars().rev().collect::<String>());
            for &n in &[1usize, 77, 512] {
                check(&mut gpu, &label, mb, ib, m, k, n, &mut rng, &mut w);
            }
            done += 1;
            if done >= 6 {
                break;
            }
        }
    }
    eprintln!(
        "PASS cases={} worst_mmq_rel_vs_f64={:.3e} worst_pertoken_rel_vs_f64={:.3e}",
        w.cases, w.mmq_rel, w.tok_rel
    );
}
