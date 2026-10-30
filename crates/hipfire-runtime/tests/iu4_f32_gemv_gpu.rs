// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the f32-activation IU4G256 decode kernels
//! (`gemv_iu4g256`, `gemv_iu4g256_residual`, `fused_iu4g256_rows`).
//!
//! 1. Bit-exact: on `repack_mq4v2(src)` the IU4 kernels must return exactly
//!    the bits of `gemv_mq4g256v2` / `gemv_hfq4g256_residual_mq4v2` on `src`.
//! 2. Reference: max abs / rel error against an f64 dot of the CPU
//!    `iu_quant::dequant` weights.
//! 3. Fused rows: each fused output equals the plain kernel on its matrix.
//!
//! Random headers and codes (xorshift), M in {1, 7, 129}, K covering every
//! tail (G mod 4 = 0..3) up to 17408. Optional real tensors: set
//! `IU4_REAL=<model.iu4p>` and `MQ_REAL=<twin.mq4p>`.
//!
//!   cargo test -p hipfire-runtime --release --test iu4_f32_gemv_gpu -- --ignored --nocapture

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

fn reference(iu4: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let w = dequant(iu4, m, k, Layout::Iu4).expect("dequant");
    (0..m)
        .map(|r| (0..k).map(|c| w[r * k + c] as f64 * x[c] as f64).sum())
        .collect()
}

fn err(got: &[f32], want: &[f64]) -> (f64, f64) {
    let max_abs = got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    let scale = want.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-12);
    (max_abs, max_abs / scale)
}

fn bits_equal(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

struct Case<'a> {
    label: &'a str,
    mq: &'a [u8],
    iu: &'a [u8],
    x: &'a [f32],
    m: usize,
    k: usize,
}

fn check(gpu: &mut Gpu, c: &Case<'_>, worst_rel: &mut f64) {
    let (m, k) = (c.m, c.k);
    let wm = gpu.upload_raw(c.mq, &[c.mq.len()]).unwrap();
    let wi = gpu.upload_raw(c.iu, &[c.iu.len()]).unwrap();
    let xd = gpu.upload_f32(c.x, &[k]).unwrap();
    let y_mq = gpu.alloc_tensor(&[m], DType::F32).unwrap();
    let y_iu = gpu.alloc_tensor(&[m], DType::F32).unwrap();
    gpu.gemv_mq4g256v2(&wm, &xd, &y_mq, m, k).unwrap();
    gpu.gemv_iu4g256(&wi, &xd, &y_iu, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let a = gpu.download_f32(&y_mq).unwrap();
    let b = gpu.download_f32(&y_iu).unwrap();
    let diff = bits_equal(&a, &b);
    let want = reference(c.iu, c.x, m, k);
    let (ea, er) = err(&b, &want);
    *worst_rel = worst_rel.max(er);
    eprintln!(
        "{:<28} m={m:<6} k={k:<6} plain   bitdiff_vs_mq4v2={diff} max_abs={ea:.3e} max_rel={er:.3e}",
        c.label
    );
    assert_eq!(diff, 0, "{} plain: {diff} rows differ from gemv_mq4g256v2", c.label);
    assert!(er < 1e-4, "{} plain rel {er:.3e}", c.label);

    // Residual: seed both outputs with the same values.
    let seed: Vec<f32> = (0..m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let r_mq = gpu.upload_f32(&seed, &[m]).unwrap();
    let r_iu = gpu.upload_f32(&seed, &[m]).unwrap();
    gpu.gemv_hfq4g256_residual_mq4v2(&wm, &xd, &r_mq, m, k).unwrap();
    gpu.gemv_iu4g256_residual(&wi, &xd, &r_iu, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let a = gpu.download_f32(&r_mq).unwrap();
    let b = gpu.download_f32(&r_iu).unwrap();
    let diff = bits_equal(&a, &b);
    let want_r: Vec<f64> = want.iter().zip(&seed).map(|(w, s)| w + *s as f64).collect();
    let (ea, er) = err(&b, &want_r);
    eprintln!(
        "{:<28} m={m:<6} k={k:<6} resid   bitdiff_vs_mq4v2={diff} max_abs={ea:.3e} max_rel={er:.3e}",
        c.label
    );
    assert_eq!(diff, 0, "{} residual differs from the mq4v2 residual", c.label);

    // Fused rows: split M into up to 3 parts plus the full matrix again.
    let plain = gpu.download_f32(&y_iu).unwrap();
    let parts: Vec<usize> = if m >= 3 { vec![m / 3, m / 3, m - 2 * (m / 3)] } else { vec![m] };
    let mut off = 0;
    let mut views = Vec::new();
    let row_b = (k / 256) * 136;
    for &p in &parts {
        let w = gpu.upload_raw(&c.iu[off * row_b..(off + p) * row_b], &[p * row_b]).unwrap();
        let y = gpu.alloc_tensor(&[p], DType::F32).unwrap();
        views.push((w, y, p, off));
        off += p;
    }
    let y_all = gpu.alloc_tensor(&[m], DType::F32).unwrap();
    {
        let mut mats: Vec<(&rdna_compute::GpuTensor, &rdna_compute::GpuTensor, usize)> =
            views.iter().map(|(w, y, p, _)| (w, y, *p)).collect();
        mats.push((&wi, &y_all, m));
        gpu.fused_iu4g256_rows(&mats, &xd, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let mut fused_diff = bits_equal(&gpu.download_f32(&y_all).unwrap(), &plain);
    for (_, y, p, o) in &views {
        fused_diff += bits_equal(&gpu.download_f32(y).unwrap(), &plain[*o..o + p]);
    }
    eprintln!("{:<28} m={m:<6} k={k:<6} fused   bitdiff_vs_plain={fused_diff}", c.label);
    assert_eq!(fused_diff, 0, "{} fused rows differ from plain", c.label);

    for (w, y, _, _) in views {
        gpu.free_tensor(w).ok();
        gpu.free_tensor(y).ok();
    }
    for t in [wm, wi, xd, y_mq, y_iu, r_mq, r_iu, y_all] {
        gpu.free_tensor(t).ok();
    }
}

#[test]
#[ignore]
fn iu4_f32_gemv_is_bit_exact_to_mq4v2_and_matches_reference() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0.0f64;
    for &m in &[1usize, 7, 129] {
        for &k in &[256usize, 512, 768, 1024, 1280, 2048, 2304, 5120, 6144, 17408] {
            let mq = mq4v2_random(&mut rng, m, k);
            let iu = repack_mq4v2(&mq, m, k, Layout::Iu4).unwrap();
            let x: Vec<f32> = (0..k).map(|_| rng.f() * 3.0).collect();
            check(&mut gpu, &Case { label: "random", mq: &mq, iu: &iu, x: &x, m, k }, &mut worst);
        }
    }

    if let (Ok(iu_path), Ok(mq_path)) = (std::env::var("IU4_REAL"), std::env::var("MQ_REAL")) {
        use hipfire_runtime::hfq::HfqFile;
        let iu_f = HfqFile::open(std::path::Path::new(&iu_path)).unwrap();
        let mq_f = HfqFile::open(std::path::Path::new(&mq_path)).unwrap();
        let mut done = 0;
        let mut seen_k = std::collections::BTreeSet::new();
        for t in iu_f.tensors().to_vec() {
            let (ti, ib) = iu_f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 57 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen_k.insert((k, m)) || m > 300_000 {
                continue;
            }
            let (_, mb) = mq_f.tensor_data(&t.name).unwrap();
            let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
            let label = format!("real:{}", t.name.chars().rev().take(26).collect::<String>().chars().rev().collect::<String>());
            check(&mut gpu, &Case { label: &label, mq: mb, iu: ib, x: &x, m, k }, &mut worst);
            done += 1;
            if done >= 8 {
                break;
            }
        }
        eprintln!("real tensors checked: {done}");
    }
    eprintln!("WORST max_rel vs f64 dequant reference = {worst:.3e}");
}
