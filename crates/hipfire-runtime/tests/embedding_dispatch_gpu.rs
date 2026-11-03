// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Routing check for the shared embedding helpers
//! `llama::embedding_lookup_dispatch` / `embedding_lookup_batched_dispatch`:
//! every `EmbeddingFormat` arm must reach the kernel of its own format.
//!
//! The reference is the format-named `Gpu` method called directly (written out
//! below, not taken from the helper), so a misrouted arm — a Q8_0 table read by
//! the HFQ4 gather, an IU8G32 arm left out — shows up as a byte mismatch or an
//! error. The helper and the reference read the same table, so agreement is
//! bit-exact. The F32 arm is also checked against the host table, and every
//! batched row against the single-row gather of the same token. Kernel
//! numerics are covered by the per-format parity tests (`iu_quant_gemv_gpu`,
//! ...); this test only proves the format→kernel mapping.
//!
//! `direct_single` matches `EmbeddingFormat` exhaustively, so a new format
//! fails to compile here until it has a reference arm.
//!
//! `#[ignore]`d: needs an RDNA GPU. Ggml tables run on gfx1201 only; elsewhere
//! the helper must return the ggml path's own refusal. Run:
//!
//!   cargo test -p hipfire-runtime --release --test embedding_dispatch_gpu -- --ignored --nocapture

use hip_bridge::HipResult;
use hipfire_runtime::llama::{
    embedding_lookup_batched_dispatch, embedding_lookup_dispatch, EmbeddingFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const DIM: usize = 256;
const VOCAB: usize = 8;

const FORMATS: [EmbeddingFormat; 7] = [
    EmbeddingFormat::F32,
    EmbeddingFormat::Q4K,
    EmbeddingFormat::HFQ4G256,
    EmbeddingFormat::HFQ4G128,
    EmbeddingFormat::Q8_0,
    EmbeddingFormat::IU8G32,
    EmbeddingFormat::Ggml(DType::GgmlQ8_0),
];

/// Table bytes, `4 * DIM` per row (enough for every format, F32 included).
/// Every byte is below 0x3C, so any f16/bf16/f32 scale read from it has a
/// biased exponent below the all-ones pattern: finite, so outputs compare
/// bit-exactly.
fn table_bytes() -> Vec<u8> {
    (0..VOCAB * DIM * 4)
        .map(|i| ((i * 37 + i / 7) % 0x3C) as u8)
        .collect()
}

fn direct_single(
    gpu: &mut Gpu,
    fmt: EmbeddingFormat,
    table: &GpuTensor,
    out: &GpuTensor,
    tok: u32,
) -> HipResult<()> {
    match fmt {
        EmbeddingFormat::F32 => gpu.embedding_lookup(table, out, tok, DIM),
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(table, out, tok, DIM),
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(table, out, tok, DIM),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(table, out, tok, DIM),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(table, out, tok, DIM),
        EmbeddingFormat::IU8G32 => gpu.embedding_lookup_iu8g32(table, out, tok, DIM),
        EmbeddingFormat::Ggml(dtype) => gpu.ggml_embedding_lookup(dtype, table, out, tok, DIM),
    }
}

/// `None`: the format has no batched kernel (the helper must refuse it).
fn direct_batched(
    gpu: &mut Gpu,
    fmt: EmbeddingFormat,
    table: &GpuTensor,
    out: &GpuTensor,
    ids: &GpuTensor,
    n: usize,
) -> Option<HipResult<()>> {
    Some(match fmt {
        EmbeddingFormat::F32 => gpu.embedding_lookup_f32_batched(table, out, ids, n, DIM),
        EmbeddingFormat::Q4K => return None,
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(table, out, ids, n, DIM),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128_batched(table, out, ids, n, DIM),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(table, out, ids, n, DIM),
        EmbeddingFormat::IU8G32 => gpu.embedding_lookup_iu8g32_batched(table, out, ids, n, DIM),
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup_batched(dtype, table, out, ids, n, DIM)
        }
    })
}

fn nan_rows(gpu: &mut Gpu, rows: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::NAN; rows * DIM], &[rows * DIM])
        .expect("alloc rows")
}

fn read(gpu: &mut Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(t).expect("download")
}

fn assert_written_and_equal(what: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().chain(want).all(|v| v.is_finite()),
        "{what}: non-finite (unwritten) output"
    );
    assert!(
        got.iter().any(|&v| v != 0.0),
        "{what}: all-zero output carries no routing signal"
    );
    let same = got
        .iter()
        .zip(want)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        same,
        "{what}: helper output differs from the format's own kernel"
    );
}

#[test]
#[ignore]
fn embedding_helpers_route_every_format_to_its_kernel() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let bytes = table_bytes();
    let toks = [3u32, 0, 7, 3, 1];
    let tok_bytes: Vec<u8> = toks.iter().flat_map(|t| t.to_le_bytes()).collect();
    let ids = gpu
        .upload_raw(&tok_bytes, &[toks.len()])
        .expect("upload ids");

    for fmt in FORMATS {
        let mut table = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .expect("upload table");
        if let EmbeddingFormat::Ggml(dtype) = fmt {
            table.dtype = dtype;
        }

        // Single row.
        let want = nan_rows(&mut gpu, 1);
        let got = nan_rows(&mut gpu, 1);
        let r_want = direct_single(&mut gpu, fmt, &table, &want, toks[0]);
        let r_got = embedding_lookup_dispatch(&mut gpu, fmt, &table, &got, toks[0], DIM);
        match (r_want, r_got) {
            (Ok(()), Ok(())) => {
                let (w, g) = (read(&mut gpu, &want), read(&mut gpu, &got));
                assert_written_and_equal(&format!("{fmt:?} single"), &g, &w);
                if fmt == EmbeddingFormat::F32 {
                    let row = toks[0] as usize * DIM * 4;
                    let host: Vec<f32> = bytes[row..row + DIM * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    assert_written_and_equal("F32 single vs host", &g, &host);
                }
            }
            // Ggml off gfx1201: the helper must surface the ggml path's refusal.
            (Err(we), Err(ge)) if matches!(fmt, EmbeddingFormat::Ggml(_)) => {
                assert_eq!(
                    format!("{we:?}"),
                    format!("{ge:?}"),
                    "{fmt:?} single refusal"
                );
                eprintln!(
                    "{fmt:?}: refused on {} (ggml is gfx1201-only): {ge:?}",
                    gpu.arch
                );
                gpu.free_tensor(table).ok();
                continue;
            }
            (w, g) => panic!("{fmt:?} single: reference {w:?} vs helper {g:?}"),
        }

        // Batched rows.
        let n = toks.len();
        let got = nan_rows(&mut gpu, n);
        let r_got = embedding_lookup_batched_dispatch(&mut gpu, fmt, &table, &got, &ids, n, DIM);
        let want = nan_rows(&mut gpu, n);
        match direct_batched(&mut gpu, fmt, &table, &want, &ids, n) {
            None => {
                assert!(
                    r_got.is_err(),
                    "{fmt:?} batched: no kernel, helper must refuse"
                );
                eprintln!("{fmt:?}: single ok, batched refused (no batched kernel)");
            }
            Some(r_want) => {
                r_want.expect("reference batched");
                r_got.expect("helper batched");
                let (w, g) = (read(&mut gpu, &want), read(&mut gpu, &got));
                assert_written_and_equal(&format!("{fmt:?} batched"), &g, &w);
                // Each batched row equals the single-row gather of its token.
                let one = nan_rows(&mut gpu, 1);
                for (i, &t) in toks.iter().enumerate() {
                    embedding_lookup_dispatch(&mut gpu, fmt, &table, &one, t, DIM)
                        .expect("single row");
                    let row = read(&mut gpu, &one);
                    assert_written_and_equal(
                        &format!("{fmt:?} batched row {i} vs single"),
                        &g[i * DIM..(i + 1) * DIM],
                        &row,
                    );
                }
                eprintln!("{fmt:?}: single + batched routed to the format's kernels");
            }
        }
        gpu.free_tensor(table).ok();
    }
}
