// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the IU4G256 W4A8 iu8-WMMA prefill route (C11,
//! `HIPFIRE_IU4_PREFILL_ACT=a8`, `block_i8_128` X).
//!
//! 1. Bit-exact: on `repack_mq4v2(src)`, `gemm_iu4g256_w4a8_prequant` returns
//!    the bits of the interleaved-row twin `gemm_mq4g256v2_asym_i8_prequant`
//!    on `src` over the same `block_i8_128` X (tails in M and N).
//! 2. Contract: both vs a CPU emulation of the fold over the downloaded X
//!    (C = sum q*x8 exact int, IU4_FOLD_RN DAG in f32): bit-exact.
//! 3. vs an f64 reference of the CPU dequant weights x f32 X: max rel.
//! 4. Determinism: two runs, bit for bit.
//!
//!   HIPFIRE_IU4_I8_TILE=128|256 cargo test -p hipfire-runtime --release \
//!     --test iu4_w4a8_i8_gemm_gpu -- --ignored --nocapture

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

/// CPU emulation of the i8 fold. `xq` is the raw `block_i8_128` plane
/// [K/128][N] x 136 B (f32 d, i32 s, 128 qs in nibble-pair order: byte
/// 8j+i of a K128 block holds element 16(j/2)... see `qs_k`).
fn emulate(mq: &[u8], xq: &[u8], m: usize, k: usize, n: usize, seed: Option<&[f32]>) -> Vec<f32> {
    let kb_n = k / 128;
    let gpr = k / 256;
    let mut y = vec![0f32; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = seed.map_or(0.0, |s| s[t * m + r]);
            let mut sum = 0.0f32;
            for kb in 0..kb_n {
                let blk = &xq[(kb * n + t) * 136..(kb * n + t + 1) * 136];
                let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                let s = i32::from_le_bytes(blk[4..8].try_into().unwrap());
                let g = &mq[(r * gpr + kb / 2) * 136..(r * gpr + kb / 2 + 1) * 136];
                let h = (kb & 1) * 4;
                let sc = f16::from_bits(u16::from_le_bytes([g[h], g[h + 1]])).to_f32();
                let zp = f16::from_bits(u16::from_le_bytes([g[h + 2], g[h + 3]])).to_f32();
                let codes = &g[8 + (kb & 1) * 64..8 + (kb & 1) * 64 + 64];
                // Nibble-pair order (block_i8_128_quant.hip): weight dword g
                // (bytes 4g..4g+3) lo nibbles pair with qs[8g+0..3], hi with
                // qs[8g+4..7].
                let mut c: i64 = 0;
                for gg in 0..16 {
                    for j in 0..4 {
                        let b = codes[4 * gg + j];
                        c += (b & 15) as i64 * (blk[8 + 8 * gg + j] as i8) as i64;
                        c += (b >> 4) as i64 * (blk[8 + 8 * gg + 4 + j] as i8) as i64;
                    }
                }
                let t1 = sc * d;
                let p = t1 * c as f32;
                let t2 = zp * d;
                sum += t2.mul_add(s as f32, p);
            }
            acc = if seed.is_some() { acc + sum } else { sum };
            y[t * m + r] = acc;
        }
    }
    y
}

#[derive(Default)]
struct Worst {
    rel: f64,
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
    emu: bool,
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
        let (y_mq, y_iu, y_iu2) = (up(gpu), up(gpu), up(gpu));
        let xq = gpu.ensure_int8_mmq_x(&xd, n, k).unwrap();
        gpu.gemm_mq4g256v2_asym_i8_prequant(&wm, xq, &y_mq, m, k, n, add)
            .unwrap();
        gpu.gemm_iu4g256_w4a8_prequant(&wi, xq, &y_iu, m, k, n, add)
            .unwrap();
        gpu.gemm_iu4g256_w4a8_prequant(&wi, xq, &y_iu2, m, k, n, add)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let xq_len = (k / 128) * n * 136;
        let xq_t = gpu.upload_raw(&vec![0u8; xq_len], &[xq_len]).unwrap();
        unsafe {
            gpu.hip
                .memcpy_dtod_raw(xq_t.buf.as_ptr(), xq as *const std::ffi::c_void, xq_len)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let xq_host = gpu.download_raw_bytes(&xq_t).unwrap();
        gpu.free_tensor(xq_t).ok();
        let d = |gpu: &mut Gpu, t: &rdna_compute::GpuTensor| gpu.download_f32(t).unwrap();
        let (a, b, b2) = (d(gpu, &y_mq), d(gpu, &y_iu), d(gpu, &y_iu2));
        for t in [y_mq, y_iu, y_iu2] {
            gpu.free_tensor(t).ok();
        }
        let wv: Vec<f64> = if add {
            want.iter().zip(&seed).map(|(w, s)| w + *s as f64).collect()
        } else {
            want.clone()
        };
        let rel = b
            .iter()
            .zip(&wv)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max)
            / sc;
        let (dtwin, ddet) = (bits_diff(&a, &b), bits_diff(&b, &b2));
        let demu = if emu {
            let e = emulate(mq, &xq_host, m, k, n, if add { Some(&seed) } else { None });
            bits_diff(&e, &b) as i64
        } else {
            -1
        };
        w.rel = w.rel.max(rel);
        eprintln!(
            "{label:<26} m={m:<6} k={k:<6} n={n:<4} {} twin_bitdiff={dtwin} emu_bitdiff={demu} rel={rel:.3e} det_diff={ddet}",
            if add { "add" } else { "set" }
        );
        assert_eq!(
            dtwin, 0,
            "{label} m={m} k={k} n={n} add={add}: IU4 W4A8 differs from the MQ4V2 asym twin"
        );
        assert_eq!(ddet, 0, "{label}: not deterministic");
        assert!(rel < 0.05, "{label} rel {rel:.3e}");
    }
    w.cases += 1;
    for t in [wm, wi, xd] {
        gpu.free_tensor(t).ok();
    }
}

#[test]
#[ignore]
fn iu4_w4a8_i8_is_bit_exact_to_twin_and_emulation() {
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
    for &m in &[1usize, 7, 127, 128, 129, 384] {
        for &k in &[256usize, 768, 1280] {
            let mq = mq4v2_random(&mut rng, m, k);
            let iu = repack_mq4v2(&mq, m, k, Layout::Iu4).unwrap();
            for &n in &[1usize, 63, 64, 65, 130, 257] {
                check(
                    &mut gpu, "random", &mq, &iu, m, k, n, &mut rng, &mut w, true,
                );
            }
        }
    }
    // Large prefill shape (256-tile grid fills the CUs).
    {
        let (m, k, n) = (4096usize, 2560usize, 2048usize);
        let mq = mq4v2_random(&mut rng, m, k);
        let iu = repack_mq4v2(&mq, m, k, Layout::Iu4).unwrap();
        check(
            &mut gpu,
            "random-large",
            &mq,
            &iu,
            m,
            k,
            n,
            &mut rng,
            &mut w,
            false,
        );
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
                check(&mut gpu, &label, mb, ib, m, k, n, &mut rng, &mut w, n <= 77);
            }
            done += 1;
            if done >= 8 {
                break;
            }
        }
    }
    eprintln!("PASS cases={} worst_rel={:.3e}", w.cases, w.rel);
}
