// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight streaming for the EmbeddingGemma-2 checkpoint.
//!
//! Layout on the GPU (the qwen35-embed contract):
//!
//! - **Projection weights f16** `[out, in]` row-major — the WMMA GEMM's
//!   weight operand. The checkpoint ships BF16; every raw or compressed
//!   source decodes through the same host-f32 step, so an `.hfq` of any
//!   tier serves the identical f16 GEMM path as the dir (an `.hfq` saves
//!   disk, not VRAM).
//! - **Norm vectors f32**, used DIRECTLY — this family's RMSNorm has no
//!   zero-centered `+1` (HF `EmbeddingGemma2RMSNorm` multiplies by
//!   `self.weight` as stored), so unlike the qwen35-embed loader nothing is
//!   offset here. `v_norm` is scale-less and has no checkpoint tensor; a
//!   ones vector is uploaded per distinct head_dim.
//! - **`layer_scalar` f32 [1]** per layer (a checkpoint buffer, not a
//!   constant — applied to the whole residual stream at each layer's end).
//! - **PLE**: `ple.per_layer_model_projection` f16 `[layers·ple, hidden]`,
//!   `ple.per_layer_projection_norm` f32 `[ple]`; per-layer block
//!   `ple_block.*` (gate/projection f16, post-norm f32).
//! - **Embedding table host f32** `[vocab, hidden]` (~537 MB at
//!   262 144×512): gathered per encode, spliced never (text-only).
//! - **`embedding_projection` f16** `[768, hidden]` — the trunk's own head.

use crate::config::{EmbedConfig, LayerType};
use crate::names::{self, TensorBytes};
use half::f16;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use rdna_compute::{Gpu, GpuTensor};

pub struct LayerWeights {
    pub input_ln: GpuTensor,  // f32 [hidden]
    pub post_attn_ln: GpuTensor,
    pub pre_ff_ln: GpuTensor,
    pub post_ff_ln: GpuTensor,
    /// The layer's output scalar (host f32 — one value, applied via
    /// `scale_f32`; a whole GPU tensor would be pure overhead).
    pub layer_scalar: f32,
    pub q_proj: GpuTensor,       // f16 [heads*hd, hidden]
    pub k_proj: GpuTensor,       // f16 [kv*hd, hidden]
    pub v_proj: GpuTensor,
    pub o_proj: GpuTensor, // f16 [hidden, heads*hd]
    pub q_norm: GpuTensor, // f32 [hd]
    pub k_norm: GpuTensor, // f32 [hd]
    /// Ones f32 [hd] — v_norm is scale-less.
    pub v_ones: GpuTensor,
    pub gate_proj: GpuTensor, // f16 [intermediate, hidden]
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor, // f16 [hidden, intermediate]
    pub ple_gate: GpuTensor,  // f16 [ple, hidden]
    pub ple_proj: GpuTensor,  // f16 [hidden, ple]
    pub ple_post_norm: GpuTensor, // f32 [hidden]
}

pub struct EmbedWeights {
    /// Host embedding table `[vocab, hidden]` f32 (gathered per encode).
    pub embed_table: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: GpuTensor, // f32 [hidden]
    /// Model-level PLE projection `[layers*ple, hidden]` f16.
    pub ple_projection: GpuTensor,
    /// PLE slice norm f32 `[ple]`.
    pub ple_norm: GpuTensor,
    /// Final head `[embedding_dim, hidden]` f16.
    pub embedding_projection: GpuTensor,
}

// ─── dtype decode (same two-convention contract as qwen35-embed) ────────────

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn decode_f32(t: &TensorBytes<'_>, name: &str) -> Result<Vec<f32>, String> {
    let n = t.numel();
    if t.quant_type != 0xFF {
        let v =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(t.quant_type, &t.bytes, n);
        if v.len() != n {
            return Err(format!(
                "tensor `{name}`: dequantized {} values for {n} elements (qt={})",
                v.len(),
                t.quant_type
            ));
        }
        return Ok(v);
    }
    let bytes = &t.bytes;
    let need_bytes = |w: usize| -> Result<(), String> {
        if bytes.len() < n * w {
            return Err(format!("tensor `{name}`: {} bytes < {}", bytes.len(), n * w));
        }
        Ok(())
    };
    match t.dtype.as_str() {
        "F32" => {
            need_bytes(4)?;
            Ok(bytes[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F16" => {
            need_bytes(2)?;
            Ok(bytes[..n * 2]
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        "BF16" => {
            need_bytes(2)?;
            Ok(bytes[..n * 2]
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        other => Err(format!(
            "tensor `{name}` has dtype {other} — expected F32/BF16/F16"
        )),
    }
}

fn decode_f16(t: &TensorBytes<'_>, name: &str) -> Result<Vec<u16>, String> {
    if t.quant_type == 0xFF && t.dtype == "F16" {
        let n = t.numel();
        if t.bytes.len() < n * 2 {
            return Err(format!("tensor `{name}`: {} bytes < {}", t.bytes.len(), n * 2));
        }
        return Ok(t.bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect());
    }
    if t.quant_type == 1 {
        let n = t.numel();
        if t.bytes.len() < n * 2 {
            return Err(format!("tensor `{name}`: {} bytes < {}", t.bytes.len(), n * 2));
        }
        return Ok(t.bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect());
    }
    Ok(decode_f32(t, name)?
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect())
}

// ─── upload helpers ─────────────────────────────────────────────────────────

fn upload_lin_f16(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, String> {
    let t = names::read(src, name)?;
    let words = decode_f16(&t, name)?;
    names::release(src, name);
    if words.len() != rows * cols {
        return Err(format!(
            "embeddinggemma2: tensor `{name}` has {} elems, expected {rows}×{cols}",
            words.len()
        ));
    }
    gpu.upload_f16_bits(&words, &[rows, cols])
        .map_err(|e| format!("embeddinggemma2: upload `{name}`: {e:?}"))
}

fn upload_vec_f32(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    n: usize,
) -> Result<GpuTensor, String> {
    let t = names::read(src, name)?;
    let v = decode_f32(&t, name)?;
    names::release(src, name);
    if v.len() != n {
        return Err(format!(
            "embeddinggemma2: tensor `{name}` has {} elems, expected {n}",
            v.len()
        ));
    }
    gpu.upload_f32(&v, &[n])
        .map_err(|e| format!("embeddinggemma2: upload `{name}`: {e:?}"))
}

fn upload_ones(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.upload_f32(&vec![1.0f32; n], &[n])
        .map_err(|e| format!("embeddinggemma2: upload ones: {e:?}"))
}

// ─── main loader ────────────────────────────────────────────────────────────

/// Stream the text backbone to the GPU. The vision/audio towers are not
/// addressed (see [`crate`]).
pub fn load_weights(gpu: &mut Gpu, src: &ModelSource, cfg: &EmbedConfig) -> Result<EmbedWeights, String> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let t = names::read(src, "embed_tokens.weight")?;
    let embed_table = decode_f32(&t, "embed_tokens.weight")?;
    names::release(src, "embed_tokens.weight");
    if embed_table.len() != cfg.vocab_size * h {
        return Err(format!(
            "embeddinggemma2: embed_tokens is {} elems, expected {}×{h}",
            embed_table.len(),
            cfg.vocab_size
        ));
    }
    drop(t);

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("layers.{li}");
        let hd = cfg.attn(li).head_dim;
        let qd = cfg.q_dim(li);
        let kvd = cfg.kv_dim(li);
        layers.push(LayerWeights {
            input_ln: upload_vec_f32(gpu, src, &format!("{p}.input_layernorm.weight"), h)?,
            post_attn_ln: upload_vec_f32(
                gpu,
                src,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
            )?,
            pre_ff_ln: upload_vec_f32(
                gpu,
                src,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                h,
            )?,
            post_ff_ln: upload_vec_f32(
                gpu,
                src,
                &format!("{p}.post_feedforward_layernorm.weight"),
                h,
            )?,
            layer_scalar: {
                let t = names::read(src, &format!("{p}.layer_scalar"))?;
                let v = decode_f32(&t, &format!("{p}.layer_scalar"))?;
                names::release(src, &format!("{p}.layer_scalar"));
                if v.len() != 1 {
                    return Err(format!(
                        "embeddinggemma2: tensor `{p}.layer_scalar` has {} elems, expected 1",
                        v.len()
                    ));
                }
                v[0]
            },
            q_proj: upload_lin_f16(gpu, src, &format!("{p}.self_attn.q_proj.weight"), qd, h)?,
            k_proj: upload_lin_f16(gpu, src, &format!("{p}.self_attn.k_proj.weight"), kvd, h)?,
            v_proj: upload_lin_f16(gpu, src, &format!("{p}.self_attn.v_proj.weight"), kvd, h)?,
            o_proj: upload_lin_f16(gpu, src, &format!("{p}.self_attn.o_proj.weight"), h, qd)?,
            q_norm: upload_vec_f32(gpu, src, &format!("{p}.self_attn.q_norm.weight"), hd)?,
            k_norm: upload_vec_f32(gpu, src, &format!("{p}.self_attn.k_norm.weight"), hd)?,
            v_ones: upload_ones(gpu, hd)?,
            gate_proj: upload_lin_f16(gpu, src, &format!("{p}.mlp.gate_proj.weight"), inter, h)?,
            up_proj: upload_lin_f16(gpu, src, &format!("{p}.mlp.up_proj.weight"), inter, h)?,
            down_proj: upload_lin_f16(gpu, src, &format!("{p}.mlp.down_proj.weight"), h, inter)?,
            ple_gate: upload_lin_f16(
                gpu,
                src,
                &format!("{p}.ple_block.per_layer_input_gate.weight"),
                h,
                h,
            )?,
            ple_proj: upload_lin_f16(
                gpu,
                src,
                &format!("{p}.ple_block.per_layer_projection.weight"),
                h,
                h,
            )?,
            ple_post_norm: upload_vec_f32(
                gpu,
                src,
                &format!("{p}.ple_block.post_per_layer_input_norm.weight"),
                h,
            )?,
        });
    }

    let final_norm = upload_vec_f32(gpu, src, "norm.weight", h)?;
    let ple_projection = upload_lin_f16(
        gpu,
        src,
        "ple.per_layer_model_projection.weight",
        cfg.num_layers * cfg.ple_dim,
        h,
    )?;
    let ple_norm = upload_vec_f32(gpu, src, "ple.per_layer_projection_norm.weight", cfg.ple_dim)?;
    let embedding_projection =
        upload_lin_f16(gpu, src, "embedding_projection.weight", cfg.embedding_dim, h)?;

    Ok(EmbedWeights {
        embed_table,
        layers,
        final_norm,
        ple_projection,
        ple_norm,
        embedding_projection,
    })
}

impl EmbedWeights {
    /// Free every GPU allocation. The embedding table is host memory.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let EmbedWeights {
            embed_table: _,
            layers,
            final_norm,
            ple_projection,
            ple_norm,
            embedding_projection,
        } = self;
        for l in layers {
            for t in [
                l.input_ln,
                l.post_attn_ln,
                l.pre_ff_ln,
                l.post_ff_ln,
                l.q_proj,
                l.k_proj,
                l.v_proj,
                l.o_proj,
                l.q_norm,
                l.k_norm,
                l.v_ones,
                l.gate_proj,
                l.up_proj,
                l.down_proj,
                l.ple_gate,
                l.ple_proj,
                l.ple_post_norm,
            ] {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(final_norm);
        let _ = gpu.free_tensor(ple_projection);
        let _ = gpu.free_tensor(ple_norm);
        let _ = gpu.free_tensor(embedding_projection);
    }

    /// Gather `ids` rows of the host embedding table into `out`
    /// (row-major `[ids.len(), hidden]`).
    pub fn gather_hidden(&self, ids: &[u32], hidden: usize, out: &mut Vec<f32>) -> Result<(), String> {
        out.clear();
        out.reserve(ids.len() * hidden);
        for &id in ids {
            let row = id as usize * hidden;
            if row + hidden > self.embed_table.len() {
                return Err(format!("token id {id} out of embedding range"));
            }
            out.extend_from_slice(&self.embed_table[row..row + hidden]);
        }
        Ok(())
    }

    /// `true` when the checkpoint carries the model-level PLE tensors
    /// (`ple.per_layer_model_projection`), used by the loader to require
    /// them exactly when the config declares a PLE width.
    pub fn source_has_ple(src: &ModelSource) -> bool {
        names::read(src, "ple.per_layer_model_projection.weight").is_ok()
    }
}

/// Layer-count sanity used by the loader's error messages.
pub fn layer_type_summary(cfg: &EmbedConfig) -> String {
    let sliding = cfg
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::Sliding)
        .count();
    format!(
        "{} sliding / {} full",
        sliding,
        cfg.num_layers - sliding
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_widen_is_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn gather_hidden_refuses_out_of_range() {
        let null = || rdna_compute::GpuTensor::null_for_test();
        let w = EmbedWeights {
            embed_table: vec![0.0f32; 6 * 2],
            layers: Vec::new(),
            final_norm: null(),
            ple_projection: null(),
            ple_norm: null(),
            embedding_projection: null(),
        };
        let mut out = Vec::new();
        assert!(w.gather_hidden(&[5], 2, &mut out).is_ok());
        assert_eq!(out.len(), 2);
        assert!(w.gather_hidden(&[6], 2, &mut out).is_err());
    }
}
