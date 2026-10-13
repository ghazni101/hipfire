// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-name resolution and source access for the EmbeddingGemma-2
//! checkpoint (same two-source contract as the qwen35-embed crate; see that
//! module's docs for the HFQ native-methods / owned-fallback rationale).
//!
//! The checkpoint stores the text backbone under `language_model.*`
//! (`EmbeddingGemma2Model` is the bare composite). The vision/audio towers
//! (`vision_tower.*`, `audio_tower.*`, `embed_vision.*`, `embed_audio.*`)
//! are deliberately NOT addressed anywhere in this crate: this arch serves
//! the text embedding path and refuses multimodal input at assembly.

use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use std::borrow::Cow;

/// One tensor's storage descriptor + bytes (borrowed from a live mmap or
/// owned when the source must copy).
pub struct TensorBytes<'a> {
    /// HFQ quant-type byte; `0xFF` for a safetensors dir (dtype-driven).
    pub quant_type: u8,
    /// Safetensors dtype string; empty for an `.hfq`.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Cow<'a, [u8]>,
}

impl TensorBytes<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Candidate source names for a backbone tensor given its bare
/// `language_model.`-stripped name, most-specific first.
pub fn tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("language_model.{name}"),
        format!("model.language_model.{name}"),
        name.to_string(),
    ]
}

/// Read a tensor by bare name from either source kind.
///
/// The caller must decode the returned bytes BEFORE calling [`release`]
/// (the release madvise drops the mmap's resident pages).
pub fn read<'a>(src: &'a ModelSource, name: &str) -> Result<TensorBytes<'a>, String> {
    let candidates = tensor_candidates(name);
    match src {
        ModelSource::Dir(d) => {
            for cand in &candidates {
                if let Some((info, bytes)) = d.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
            }
        }
        ModelSource::Hfq(h) => {
            for cand in &candidates {
                if let Some((info, bytes)) = h.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
                if let Some((info, vec)) = h.tensor_data_vec(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Owned(vec),
                    });
                }
            }
        }
    }
    Err(format!(
        "embeddinggemma2: missing tensor `{name}` (tried {candidates:?})"
    ))
}

/// Release the pages a [`read`] pulled in, once the caller has consumed the
/// bytes. Best-effort and advisory.
pub fn release(src: &ModelSource, name: &str) {
    let source: &dyn hipfire_runtime::model_source::ModelSource = match src {
        ModelSource::Dir(d) => d,
        ModelSource::Hfq(h) => h,
    };
    for cand in tensor_candidates(name) {
        if source.tensor_info(&cand).is_some() {
            source.release_tensor_pages(&cand);
            return;
        }
    }
}

/// Bare backbone names every checkpoint must provide (layer index elided).
pub fn required_text_tensors() -> Vec<String> {
    let mut out = vec![
        "embed_tokens.weight".to_string(),
        "norm.weight".to_string(),
        "embedding_projection.weight".to_string(),
        "ple.per_layer_model_projection.weight".to_string(),
        "ple.per_layer_projection_norm.weight".to_string(),
    ];
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "layer_scalar",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "ple_block.per_layer_input_gate.weight",
        "ple_block.per_layer_projection.weight",
        "ple_block.post_per_layer_input_norm.weight",
    ] {
        out.push(format!("layers.N.{n}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_prefer_bare_composite_prefix() {
        assert_eq!(
            tensor_candidates("layers.0.self_attn.q_proj.weight"),
            vec![
                "language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "layers.0.self_attn.q_proj.weight".to_string(),
            ]
        );
        assert_eq!(
            tensor_candidates("ple.per_layer_model_projection.weight")[0],
            "language_model.ple.per_layer_model_projection.weight"
        );
    }
}
