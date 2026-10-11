// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-name resolution for the qwen35-embed checkpoint.
//!
//! The pplx-embed-v2 release stores the text backbone under
//! `language_model.*` and the tower under `visual.*` (no `model.` wrapper —
//! `Qwen3_5Model` is the bare composite, unlike `Qwen3_5ForCausalLM` whose
//! tensors live under `model.*`). Exports that do carry the `model.` wrapper
//! (`Qwen3_5ForConditionalGeneration`-style re-saves) are accepted too, so
//! [`tensor_candidates`] tries both prefixes in order.

/// Checkpoint-relative name of the sentence-transformers Dense weight.
pub const DENSE_TENSOR: &str = "linear.weight";
/// Dir-relative path of the Dense weight's safetensors shard.
pub const DENSE_SHARD: &str = "1_Dense/model.safetensors";

/// Candidate source names for a backbone tensor given its bare
/// `language_model.`-stripped name (`embed_tokens.weight`,
/// `layers.0.self_attn.q_proj.weight`, …), most-specific first.
pub fn tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("language_model.{name}"),
        format!("model.language_model.{name}"),
        name.to_string(),
    ]
}

/// Candidate source names for a vision-tower tensor given its bare
/// `visual.`-stripped name (`blocks.0.attn.qkv.weight`, …).
pub fn vision_tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("visual.{name}"),
        format!("model.visual.{name}"),
        name.to_string(),
    ]
}

/// Look `name` up in a source, trying every candidate prefix.
pub fn resolve<'a>(
    src: &'a dyn hipfire_runtime::model_source::ModelSource,
    name: &str,
) -> Option<(&'a hipfire_runtime::model_source::TensorInfo, &'a [u8])> {
    for candidate in tensor_candidates(name) {
        if let Some(hit) = src.tensor_data(&candidate) {
            return Some(hit);
        }
    }
    None
}

/// Look a vision tensor up, trying every candidate prefix.
pub fn resolve_vision<'a>(
    src: &'a dyn hipfire_runtime::model_source::ModelSource,
    name: &str,
) -> Option<(&'a hipfire_runtime::model_source::TensorInfo, &'a [u8])> {
    for candidate in vision_tensor_candidates(name) {
        if let Some(hit) = src.tensor_data(&candidate) {
            return Some(hit);
        }
    }
    None
}

/// Bare backbone names every checkpoint must provide (layer index elided).
pub fn required_text_tensors() -> Vec<String> {
    let mut out = vec![
        "embed_tokens.weight".to_string(),
        "norm.weight".to_string(),
    ];
    // Both layer kinds share the MLP + norms; the mixer names differ.
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        out.push(format!("layers.N.{n}"));
    }
    for n in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
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
            vision_tensor_candidates("patch_embed.proj.weight")[0],
            "visual.patch_embed.proj.weight"
        );
    }
}
