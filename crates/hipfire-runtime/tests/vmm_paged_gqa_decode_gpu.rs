// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the paged-pool GQA-shared Q8_0 decode tile
//! (`attention_flash_q8_0_tile_gqa_paged_gfx1100`).
//!
//! The GQA tile reads each K/V page once per kv head and serves six query
//! heads from it; the batched VMM tile reads the page once per query head.
//! Both write the same `[rows, n_heads, max_tiles, 2 + head_dim]` partials and
//! run the same Q8 reduce, so their outputs must agree to f32 rounding (their
//! per-tile reduction orders differ). Determinism is exact.
//!
//!   cargo test -p hipfire-runtime --release --test vmm_paged_gqa_decode_gpu -- --ignored --nocapture

use rdna_compute::kv_slots::{VmmPagedKvSlotDesc, VMM_PAGED_DESC_BYTES};
use rdna_compute::{DType, Gpu};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn i8(&mut self) -> i8 {
        (self.next() >> 33) as i8
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Q8_0 page arena: `[page][128 positions][n_kv_heads][8 blocks][34 B]`.
fn q8_arena(rng: &mut Rng, pages: usize, n_kv_heads: usize, head_dim: usize) -> Vec<u8> {
    let stride = n_kv_heads * (head_dim / 32) * 34;
    let mut out = vec![0u8; pages * 128 * stride];
    let scale = half::f16::from_f32(0.02).to_le_bytes();
    for page in out.chunks_mut(128 * stride) {
        for pos in page.chunks_mut(stride) {
            for seg in pos.chunks_mut(head_dim / 32 * 34) {
                for blk in seg.chunks_mut(34) {
                    blk[..2].copy_from_slice(&scale);
                    for c in &mut blk[2..] {
                        *c = rng.i8() as u8;
                    }
                }
            }
        }
    }
    out
}

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

fn bits_differ(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
#[ignore = "GPU required: run with --ignored on a gfx1100 host"]
fn paged_gqa_matches_batched_vmm_decode() {
    let mut gpu = Gpu::init().expect("GPU required");
    if gpu.arch != "gfx1100" {
        eprintln!("skip: paged GQA tile is gfx1100-only (arch {})", gpu.arch);
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (24usize, 4usize, 256usize);
    let tile = gpu.attn_tile_size();
    if tile != 128 {
        eprintln!("skip: batched reference tile {tile} != 128");
        return;
    }
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    for tokens in [129usize, 300, 1024] {
        let pages = tokens.div_ceil(128);
        let arena = q8_arena(&mut rng, pages, n_kv_heads, head_dim);
        let stride = n_kv_heads * (head_dim / 32) * 34;
        let k = gpu.upload_raw(&arena, &[arena.len()]).unwrap();
        let v = gpu.upload_raw(&arena, &[arena.len()]).unwrap();
        let block_table: Vec<i32> = (0..pages as i32).collect();
        let bt = gpu
            .upload_raw(&i32_bytes(&block_table), &[block_table.len() * 4])
            .unwrap();
        let desc = VmmPagedKvSlotDesc {
            block_table: bt.buf.as_ptr() as u64,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len: tokens as i32,
            page_tokens: 128,
            granule_bytes: 0,
            pages_per_granule: 0,
            _pad: 0,
        };
        let descs = gpu
            .upload_raw(
                VmmPagedKvSlotDesc::as_bytes(&[desc]),
                &[VMM_PAGED_DESC_BYTES],
            )
            .unwrap();
        let positions = gpu
            .upload_raw(&i32_bytes(&[(tokens - 1) as i32]), &[4])
            .unwrap();
        let row_slot = gpu.upload_raw(&i32_bytes(&[0]), &[4]).unwrap();
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
            .collect();
        let qd = gpu.upload_f32(&q, &[n_heads * head_dim]).unwrap();
        let max_tiles = tokens.div_ceil(tile);
        let per_row = n_heads * max_tiles * (2 + head_dim);
        let out_ref = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32).unwrap();
        let out_gqa = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32).unwrap();
        let out_again = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32).unwrap();
        let p_ref = gpu.alloc_tensor(&[per_row], DType::F32).unwrap();
        let p_gqa = gpu.alloc_tensor(&[per_row], DType::F32).unwrap();
        let p_again = gpu.alloc_tensor(&[per_row], DType::F32).unwrap();

        gpu.attention_flash_q8_0_batched_masked_slots_vmm_paged(
            &qd,
            &k,
            &v,
            &out_ref,
            &positions,
            n_heads,
            n_kv_heads,
            head_dim,
            tokens,
            tokens,
            1,
            &p_ref,
            None,
            0,
            0,
            Some(&descs),
            Some(&row_slot),
        )
        .unwrap();
        gpu.attention_flash_q8_0_gqa_paged_slots(
            &qd, &k, &v, &out_gqa, &positions, n_heads, n_kv_heads, head_dim, tokens, tokens, 1,
            &p_gqa, &descs, &row_slot,
        )
        .unwrap();
        gpu.attention_flash_q8_0_gqa_paged_slots(
            &qd, &k, &v, &out_again, &positions, n_heads, n_kv_heads, head_dim, tokens, tokens, 1,
            &p_again, &descs, &row_slot,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let want = gpu.download_f32(&out_ref).unwrap();
        let got = gpu.download_f32(&out_gqa).unwrap();
        let again = gpu.download_f32(&out_again).unwrap();
        let got_p = gpu.download_f32(&p_gqa).unwrap();
        let again_p = gpu.download_f32(&p_again).unwrap();
        let _ = stride;

        assert!(
            want.iter().all(|v| v.is_finite()) && want.iter().any(|v| *v != 0.0),
            "tokens={tokens}: batched reference is degenerate"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "tokens={tokens}: GQA output is not finite"
        );
        assert_eq!(
            bits_differ(&got, &again),
            0,
            "tokens={tokens}: GQA output is not deterministic"
        );
        assert_eq!(
            bits_differ(&got_p, &again_p),
            0,
            "tokens={tokens}: GQA partials are not deterministic"
        );
        let rel = max_rel(&got, &want);
        eprintln!("tokens={tokens} max_rel={rel:.3e}");
        assert!(
            rel < 1e-5,
            "tokens={tokens}: GQA vs batched VMM max_rel {rel:.3e} exceeds 1e-5"
        );

        for t in [
            k, v, bt, descs, positions, row_slot, qd, out_ref, out_gqa, out_again, p_ref, p_gqa,
            p_again,
        ] {
            gpu.free_tensor(t).unwrap();
        }
    }
}
