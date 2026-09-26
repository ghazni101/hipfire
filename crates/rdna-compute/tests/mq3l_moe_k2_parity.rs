// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerical parity for the K2-Horizon MQ3-Lloyd (qt=20) MoE additions:
//!
//!   * `gemv_mq3g256_lloyd_moe_gate_up_indexed_batched` — wraps the
//!     pre-existing `gemv_mq3g256_lloyd_moe_gate_up_indexed_batched_k4.hip`.
//!   * `gemv_mq3g256_lloyd_moe_down_indexed_batched_expanded` — wraps the
//!     new `gemv_mq3g256_lloyd_moe_down_indexed_batched_expanded.hip`, which
//!     writes raw per-expert rows (no topk weight, no residual fold) so the
//!     caller owns the nonlinearity + combine.
//!
//! Reference: CPU dequant of the documented MQ3G256Lloyd group layout —
//! 112 B per 256 weights = 16 B (8 × fp16 codebook, ascending) + 96 B
//! (32 chunks × 3 bytes, 8 × 3-bit LE indices per chunk).
//!
//! `#[ignore]`d: needs an RDNA wave32 GPU with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test mq3l_moe_k2_parity -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor};

const TOL: f32 = 2e-4;

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], DType::Raw).expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu.alloc_tensor(&[data.len()], DType::F32).expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn alloc_f32(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.alloc_tensor(&[n], DType::F32).expect("alloc f32")
}
fn download_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    let mut out = vec![0f32; t.shape.iter().product::<usize>()];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}
fn f16(v: f32) -> u16 {
    // half-precision encode (round-to-nearest-even)
    let b = v.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7FFFFF;
    if exp == 255 {
        return sign as u16 | if man == 0 { 0x7C00 } else { 0x7E00 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign as u16 | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        let m = man | 0x800000;
        let shift = (14 - e) as u32;
        let mut half = m >> shift;
        if (m >> (shift - 1)) & 1 == 1 {
            half += 1;
        }
        return (sign | half) as u16;
    }
    let mut half = ((e as u32) << 10) | (man >> 13);
    if man & 0x1FFF == 0x1000 {
        // tie: keep even
    } else if man & 0x1000 != 0 {
        half += 1;
    }
    (sign | half) as u16
}
fn f16d(h: u16) -> f32 {
    let sign = (h & 0x8000) as u32;
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    let bits = if e == 0 {
        let mut ee = -14i32;
        let mut mm = m as u32;
        if mm == 0 {
            sign << 16
        } else {
            while mm & 0x400 == 0 {
                mm <<= 1;
                ee -= 1;
            }
            mm &= 0x3FF;
            (sign << 16) | (((ee + 127) as u32) << 23) | (mm << 13)
        }
    } else if e == 31 {
        (sign << 16) | 0x7F800000 | ((m as u32) << 13)
    } else {
        (sign << 16) | (((e as u32 - 15 + 127) << 23)) | ((m as u32) << 13)
    };
    f32::from_bits(bits)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Synthesize an MQ3G256Lloyd packed tensor: m rows × k cols, 112 B/group.
/// Codebook entries are 8 distinct sorted f32s encoded fp16; indices nonzero.
fn synth_mq3l(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 112;
    let mut out = vec![0u8; m * row_bytes];
    let mut rng = Rng(seed);
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 112;
            // 8 ascending codebook entries around a row-varying center.
            let center = ((rng.next() % 1000) as f32 - 500.0) * 1e-3;
            for e in 0..8 {
                let v = center + (e as f32 - 3.5) * 0.03;
                out[off + e * 2..off + e * 2 + 2].copy_from_slice(&f16(v).to_le_bytes());
            }
            for ch in 0..32 {
                // every chunk carries a nonzero index field
                let pk = (rng.next() | 0x0100_0000) & 0x00FF_FFFF;
                out[off + 16 + ch * 3..off + 16 + ch * 3 + 3]
                    .copy_from_slice(&pk.to_le_bytes()[..3]);
            }
        }
    }
    out
}

/// CPU dequant one row → k floats (codebook[((pk >> 3i) & 7)]).
fn deq_mq3l_row(packed: &[u8], row: usize, k: usize) -> Vec<f32> {
    let groups = k / 256;
    let rb = groups * 112;
    let mut w = vec![0f32; k];
    for g in 0..groups {
        let off = row * rb + g * 112;
        let mut cb = [0f32; 8];
        for e in 0..8 {
            cb[e] = f16d(u16::from_le_bytes([packed[off + e * 2], packed[off + e * 2 + 1]]));
        }
        for ch in 0..32 {
            let d = &packed[off + 16 + ch * 3..off + 16 + ch * 3 + 3];
            let pk = (d[0] as u32) | ((d[1] as u32) << 8) | ((d[2] as u32) << 16);
            for i in 0..8 {
                w[g * 256 + ch * 8 + i] = cb[((pk >> (3 * i)) & 7) as usize];
            }
        }
    }
    w
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Expanded-down parity: batch=2 rows, top-3 of 4 experts, K=1024 (4 groups,
/// no tail) + K=2816 (11 groups, 3-tail) to cover the tail path.
#[test]
#[ignore]
fn mq3l_down_expanded_parity() {
    let mut gpu = Gpu::init().expect("gpu init");
    for &k in &[1024usize, 2816] {
        let (m, n_exp, k_top, n) = (64usize, 4usize, 3usize, 2usize);
        // Per-expert packed weights.
        let mut expert_ptrs_host = Vec::new();
        let mut experts_gpu = Vec::new();
        let mut refs = Vec::new();
        for e in 0..n_exp {
            let packed = synth_mq3l(m, k, 0x1234 + e as u64);
            let t = upload_u8(&mut gpu, &packed);
            expert_ptrs_host.push(t.buf.as_ptr() as u64);
            experts_gpu.push(t);
            refs.push(packed);
        }
        let ptrs = upload_u64(&mut gpu, &expert_ptrs_host);
        // topk indices/weights: N×K_TOP
        let mut topk = Vec::new();
        for b in 0..n {
            for kk in 0..k_top {
                topk.push(((b * k_top + kk) % n_exp) as i32);
            }
        }
        let topk_gpu = upload_i32(&mut gpu, &topk);
        // rot_batch [n × k_top × k]
        let mut rot = vec![0f32; n * k_top * k];
        let mut rng = Rng(0xBEEF);
        for v in rot.iter_mut() {
            *v = ((rng.next() % 2000) as f32 - 1000.0) * 1e-4;
        }
        let rot_gpu = upload_f32(&mut gpu, &rot);
        let out = alloc_f32(&mut gpu, n * k_top * m);
        gpu.gemv_mq3g256_lloyd_moe_down_indexed_batched_expanded(
            &ptrs, &topk_gpu, &rot_gpu, &out, m, k, k_top, n,
        )
        .expect("expanded gemv");
        gpu.hip.device_synchronize().expect("sync");
        let got = download_f32(&gpu, &out);

        for b in 0..n {
            for kk in 0..k_top {
                let e = topk[b * k_top + kk] as usize;
                let w = deq_mq3l_row(&refs[e], 0, k);
                let x = &rot[(b * k_top + kk) * k..(b * k_top + kk) * k + k];
                // CPU reference: row 0 of the expert only (row r = dot(row_r, x)).
                // Compare a few rows: recompute each row's dequant.
                for row in 0..m.min(16) {
                    let wr = deq_mq3l_row(&refs[e], row, k);
                    let exp: f32 = wr.iter().zip(x).map(|(a, b)| a * b).sum();
                    let idx = (b * k_top + kk) * m + row;
                    let err = (got[idx] - exp).abs();
                    assert!(
                        err < TOL || err < exp.abs() * 1e-3,
                        "down-expanded K={k} b{b} k{kk} row{row}: got {} exp {} (err {err})",
                        got[idx],
                        exp
                    );
                }
                let _ = w;
            }
        }
        for t in experts_gpu {
            gpu.free_tensor(t).ok();
        }
    }
}

/// Gate_up parity: y_gate/y_up split at M/2, K=1024 + K=2816.
#[test]
#[ignore]
fn mq3l_gate_up_batched_parity() {
    let mut gpu = Gpu::init().expect("gpu init");
    for &k in &[1024usize, 2816] {
        let (mi, n_exp, k_top, n) = (64usize, 4usize, 3usize, 2usize);
        let m = 2 * mi;
        let mut expert_ptrs_host = Vec::new();
        let mut experts_gpu = Vec::new();
        let mut refs = Vec::new();
        for e in 0..n_exp {
            let packed = synth_mq3l(m, k, 0x5678 + e as u64);
            let t = upload_u8(&mut gpu, &packed);
            expert_ptrs_host.push(t.buf.as_ptr() as u64);
            experts_gpu.push(t);
            refs.push(packed);
        }
        let ptrs = upload_u64(&mut gpu, &expert_ptrs_host);
        let mut topk = Vec::new();
        for b in 0..n {
            for kk in 0..k_top {
                topk.push(((b * k_top + kk + 1) % n_exp) as i32);
            }
        }
        let topk_gpu = upload_i32(&mut gpu, &topk);
        // x is per-token [n × k] (not per-krank).
        let mut x = vec![0f32; n * k];
        let mut rng = Rng(0xCAFE);
        for v in x.iter_mut() {
            *v = ((rng.next() % 2000) as f32 - 1000.0) * 1e-4;
        }
        let x_gpu = upload_f32(&mut gpu, &x);
        let y_gate = alloc_f32(&mut gpu, n * k_top * mi);
        let y_up = alloc_f32(&mut gpu, n * k_top * mi);
        gpu.gemv_mq3g256_lloyd_moe_gate_up_indexed_batched(
            &ptrs, &topk_gpu, &x_gpu, &y_gate, &y_up, m, k, k_top, n,
        )
        .expect("gate_up gemv");
        gpu.hip.device_synchronize().expect("sync");
        let got_g = download_f32(&gpu, &y_gate);
        let got_u = download_f32(&gpu, &y_up);

        for b in 0..n {
            for kk in 0..k_top {
                let e = topk[b * k_top + kk] as usize;
                let xrow = &x[b * k..b * k + k];
                for row in 0..16usize {
                    // gate half: row → y_gate[(b*K_TOP+kk)*mi + row]
                    let wg = deq_mq3l_row(&refs[e], row, k);
                    let exp_g: f32 = wg.iter().zip(xrow).map(|(a, b)| a * b).sum();
                    let gi = (b * k_top + kk) * mi + row;
                    assert!(
                        (got_g[gi] - exp_g).abs() < TOL
                            || (got_g[gi] - exp_g).abs() < exp_g.abs() * 1e-3,
                        "gate K={k} b{b} k{kk} row{row}: got {} exp {}",
                        got_g[gi],
                        exp_g
                    );
                    // up half: row+mi → y_up
                    let wu = deq_mq3l_row(&refs[e], row + mi, k);
                    let exp_u: f32 = wu.iter().zip(xrow).map(|(a, b)| a * b).sum();
                    let ui = (b * k_top + kk) * mi + row;
                    assert!(
                        (got_u[ui] - exp_u).abs() < TOL
                            || (got_u[ui] - exp_u).abs() < exp_u.abs() * 1e-3,
                        "up K={k} b{b} k{kk} row{row}: got {} exp {}",
                        got_u[ui],
                        exp_u
                    );
                }
            }
        }
        for t in experts_gpu {
            gpu.free_tensor(t).ok();
        }
    }
}
