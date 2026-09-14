// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon config parsing (`K2HorizonConfig`).
//!
//! Parses the `config.json` from the K2-Horizon model family (e.g.
//! `IFM/K2-Horizon-MoVA-36B-A4B`). The config is a Qwen3-MoE derivative
//! with additional fields for MoVA attention, sigmoid routing, grouped
//! RMSNorm, and a dense-layer prefix.
//!
//! All fields are pulled from the model's `config.json` verified against
//! `~/models/RAW/IFM/K2-Horizon-MoVA-36B-A4B/source/config.json`.

use serde::Deserialize;

// ─── Config ─────────────────────────────────────────────────────────────

/// Whether a layer uses standard attention + dense MLP or MoVA attention +
/// sigmoid-routed MoE FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Standard MHA (q/k/v/o) + dense SwiGLU MLP.
    Dense,
    /// MoVA attention (q/k/o + v_experts + gate) + sigmoid-routed MoE FFN.
    Moe,
}

/// Parsed K2-Horizon configuration.
///
/// Not all fields are used by every forward path, but every field present
/// in the upstream `config.json` is captured here so the loader and
/// forward have a single source of truth.
#[derive(Debug, Clone)]
pub struct K2HorizonConfig {
    // ── Core dimensions ──
    pub dim: usize,        // hidden_size = 2560
    pub n_layers: usize,   // num_hidden_layers = 48
    pub vocab_size: usize, // 250624
    pub norm_eps: f32,     // rms_norm_eps = 1e-6
    pub eos_token: u32,    // first eos_token_id (1)

    // ── Attention ──
    pub n_heads: usize,       // num_attention_heads = 32
    pub n_kv_heads: usize,    // num_key_value_heads = 8
    pub head_dim: usize,      // 128
    pub rope_theta: f32,      // 10_000_000
    pub rope_head_dim: usize, // 128 (== head_dim → full rotary)
    pub attention_bias: bool, // false

    // ── Grouped RMSNorm ──
    pub layernorm_num_groups: usize, // 2

    // ── Dense FFN (layers 0–2) ──
    pub intermediate_size: usize, // 6144

    // ── MoE FFN (layers 3–47) ──
    pub num_experts: usize,           // 100
    pub num_experts_per_tok: usize,   // 8
    pub moe_intermediate_size: usize, // 768
    pub num_shared_experts: usize,    // 1
    pub norm_topk_prob: bool,         // true

    // ── MoE router ──
    pub router_score_func: String,  // "sigmoid"
    pub router_scaling_factor: f32, // 2.5
    pub moe_gate_bias: bool,        // true

    // ── MoVA attention ──
    pub mova_num_experts: usize,         // 64
    pub mova_num_experts_per_tok: usize, // 4
    pub attention_gate_func: String,     // "softplus"

    // ── Layer topology ──
    pub mlp_only_layers: Vec<usize>, // [0, 1, 2]
    pub layer_kinds: Vec<LayerKind>, // derived from mlp_only_layers

    // ── Misc ──
    pub max_position_embeddings: usize, // 524288
    pub tie_word_embeddings: bool,      // false
}

// ─── Raw serde struct (mirrors config.json) ─────────────────────────────

#[derive(Deserialize)]
struct RawK2HorizonConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    #[serde(default = "default_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    rope_head_dim: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default = "default_intermediate")]
    intermediate_size: usize,
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    num_shared_experts: usize,
    #[serde(default = "default_norm_topk")]
    norm_topk_prob: bool,
    #[serde(default = "default_router_score")]
    router_score_func: String,
    #[serde(default = "default_router_scaling")]
    router_scaling_factor: f32,
    #[serde(default)]
    moe_gate_bias: bool,
    #[serde(default)]
    mova_num_experts: usize,
    #[serde(default)]
    mova_num_experts_per_tok: usize,
    #[serde(default = "default_gate_func")]
    attention_gate_func: String,
    #[serde(default)]
    mlp_only_layers: Option<Vec<usize>>,
    #[serde(default)]
    layernorm_num_groups: Option<usize>,
    #[serde(default)]
    decoder_sparse_step: Option<usize>,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    rope_parameters: Option<RawRope>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct RawRope {
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_type: Option<String>,
}

fn default_norm_eps() -> f32 {
    1e-6
}
fn default_intermediate() -> usize {
    0
}
fn default_norm_topk() -> bool {
    true
}
fn default_router_score() -> String {
    // K2-Horizon is sigmoid-routed; a config that omits the field gets the
    // arch default, not a value the validator then rejects.
    "sigmoid".to_string()
}
fn default_router_scaling() -> f32 {
    1.0
}
fn default_gate_func() -> String {
    "softplus".to_string()
}
fn default_max_pos() -> usize {
    32768
}

/// Resolve a scalar-or-array `eos_token_id` to a single token, using the
/// first element of an array (uniform with qwen35). Absent/null/unexpected
/// → default.
fn first_token_or(v: Option<&serde_json::Value>, default: u32) -> u32 {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32).unwrap_or(default),
        Some(serde_json::Value::Array(a)) => a
            .first()
            .and_then(|e| e.as_u64())
            .map(|x| x as u32)
            .unwrap_or(default),
        _ => default,
    }
}

// ─── Parsing ────────────────────────────────────────────────────────────

/// Parse a `K2HorizonConfig` from the outer `config` JSON node (the inner
/// blob under the metadata_json `config` key, or a raw `config.json`).
pub fn config_from_json(config_json: &str) -> Result<K2HorizonConfig, String> {
    let value: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| format!("k2_horizon: parsing config JSON: {e}"))?;
    config_from_value(&value)
}

/// Parse from a pre-parsed JSON value (the `config` object, or a raw config.json).
fn config_from_value(config: &serde_json::Value) -> Result<K2HorizonConfig, String> {
    let raw: RawK2HorizonConfig = serde_json::from_value(config.clone())
        .map_err(|e| format!("k2_horizon: parsing config fields: {e}"))?;
    config_from_raw(raw)
}

/// Build a `K2HorizonConfig` from the parsed raw serde struct.
fn config_from_raw(raw: RawK2HorizonConfig) -> Result<K2HorizonConfig, String> {
    let dim = raw.hidden_size;
    let n_heads = raw.num_attention_heads;
    let n_kv_heads = raw.num_key_value_heads.unwrap_or(n_heads);
    let head_dim = raw.head_dim.unwrap_or(dim / n_heads.max(1));
    let rope_head_dim = raw.rope_head_dim.unwrap_or(head_dim);

    // ── Fail-closed validation ──
    // Every check below guards a silent-corruption or panic path: kernel
    // contracts (topk shared arrays, indexed-GEMV group strides), the
    // contiguous-dense-prefix assumption in the layer dispatch, and config
    // fields the forward path does not honor. A variant checkpoint that
    // deviates must fail at load, not serve wrong output.
    if n_heads == 0 {
        return Err("k2_horizon: num_attention_heads must be > 0".into());
    }
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(format!(
            "k2_horizon: num_attention_heads {n_heads} not divisible by num_key_value_heads {n_kv_heads}"
        ));
    }
    if head_dim == 0 || rope_head_dim != head_dim {
        return Err(format!(
            "k2_horizon: rope_head_dim {rope_head_dim} != head_dim {head_dim} — partial rotary unsupported"
        ));
    }
    let n_groups = raw.layernorm_num_groups.unwrap_or(2);
    if n_groups == 0 || dim % n_groups != 0 {
        return Err(format!(
            "k2_horizon: hidden_size {dim} not divisible by layernorm_num_groups {n_groups}"
        ));
    }
    // grouped_rmsnorm_f32 launches block = min(256, chunk_len) into a
    // power-of-two LDS tree — a non-pow2 chunk_len < 256 silently drops
    // tail partials. The launcher rounds down to pow2, so any chunk_len
    // works, but keep the divisibility check above as the real contract.
    if raw.attention_bias {
        return Err("k2_horizon: attention_bias=true unsupported (no bias on q/k/o)".into());
    }
    if raw.router_score_func != "sigmoid" {
        return Err(format!(
            "k2_horizon: router_score_func {:?} unsupported — forward is sigmoid-only",
            raw.router_score_func
        ));
    }
    if !raw.norm_topk_prob {
        return Err(
            "k2_horizon: norm_topk_prob=false unsupported — topk kernel always normalizes".into(),
        );
    }
    if raw.attention_gate_func != "softplus" {
        return Err(format!(
            "k2_horizon: attention_gate_func {:?} unsupported — softplus gate only",
            raw.attention_gate_func
        ));
    }
    if raw.num_experts > 0 && raw.num_shared_experts != 1 {
        return Err(format!(
            "k2_horizon: num_shared_experts {} unsupported — exactly 1 shared expert",
            raw.num_shared_experts
        ));
    }
    if raw.tie_word_embeddings {
        return Err("k2_horizon: tie_word_embeddings=true unsupported (lm_head required)".into());
    }
    if let Some(step) = raw.decoder_sparse_step {
        if step != 1 {
            return Err(format!(
                "k2_horizon: decoder_sparse_step {step} unsupported — every post-prefix layer must be MoE"
            ));
        }
    }
    // Kernel contracts: the bias-aware topk kernel uses static
    // __shared__[1024] and a 32-lane argmax; the indexed GEMVs stride by
    // 256-wide groups.
    if raw.num_experts > 1024 || raw.mova_num_experts > 1024 {
        return Err(format!(
            "k2_horizon: expert count exceeds kernel limit 1024 (moe={}, mova={})",
            raw.num_experts, raw.mova_num_experts
        ));
    }
    // topk bounds only apply when the corresponding expert set exists —
    // a config with num_experts=0 has no MoE layers to route.
    if raw.num_experts > 0
        && (raw.num_experts_per_tok == 0
            || raw.num_experts_per_tok > raw.num_experts
            || raw.num_experts_per_tok > 32)
    {
        return Err(format!(
            "k2_horizon: num_experts_per_tok {} out of range (1..=min(32, num_experts))",
            raw.num_experts_per_tok
        ));
    }
    if raw.mova_num_experts > 0
        && (raw.mova_num_experts_per_tok == 0
            || raw.mova_num_experts_per_tok > raw.mova_num_experts
            || raw.mova_num_experts_per_tok > 32)
    {
        return Err(format!(
            "k2_horizon: mova_num_experts_per_tok {} out of range (1..=min(32, mova_num_experts))",
            raw.mova_num_experts_per_tok
        ));
    }
    if dim % 256 != 0 || raw.moe_intermediate_size % 256 != 0 {
        return Err(format!(
            "k2_horizon: hidden {dim} and moe_intermediate {} must be multiples of 256 (indexed GEMV group stride)",
            raw.moe_intermediate_size
        ));
    }

    let mlp_only_layers = raw.mlp_only_layers.unwrap_or_default();
    // The layer dispatch assumes dense layers form a contiguous prefix
    // (run_decode_body computes global_layer = dense_layers.len() + l).
    // A non-prefix mlp_only_layers would silently bind wrong weights and
    // wrong KV slots — reject it rather than misroute.
    for (i, &l) in mlp_only_layers.iter().enumerate() {
        if l != i {
            return Err(format!(
                "k2_horizon: mlp_only_layers {mlp_only_layers:?} is not a contiguous prefix — unsupported topology"
            ));
        }
    }
    if mlp_only_layers.len() > raw.num_hidden_layers {
        return Err("k2_horizon: mlp_only_layers longer than num_hidden_layers".into());
    }

    let rope_theta = raw
        .rope_parameters
        .as_ref()
        .and_then(|r| r.rope_theta)
        .unwrap_or(10_000_000.0);

    // Derive per-layer kind: layers in mlp_only_layers are Dense, rest are Moe.
    let layer_kinds: Vec<LayerKind> = (0..raw.num_hidden_layers)
        .map(|i| {
            if mlp_only_layers.contains(&i) {
                LayerKind::Dense
            } else {
                LayerKind::Moe
            }
        })
        .collect();

    let config = K2HorizonConfig {
        dim,
        n_layers: raw.num_hidden_layers,
        vocab_size: raw.vocab_size,
        norm_eps: raw.rms_norm_eps,
        eos_token: first_token_or(raw.eos_token_id.as_ref(), 1),
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
        rope_head_dim,
        attention_bias: raw.attention_bias,
        layernorm_num_groups: n_groups,
        intermediate_size: raw.intermediate_size,
        num_experts: raw.num_experts,
        num_experts_per_tok: raw.num_experts_per_tok,
        moe_intermediate_size: raw.moe_intermediate_size,
        num_shared_experts: raw.num_shared_experts,
        norm_topk_prob: raw.norm_topk_prob,
        router_score_func: raw.router_score_func,
        router_scaling_factor: raw.router_scaling_factor,
        moe_gate_bias: raw.moe_gate_bias,
        mova_num_experts: raw.mova_num_experts,
        mova_num_experts_per_tok: raw.mova_num_experts_per_tok,
        attention_gate_func: raw.attention_gate_func,
        mlp_only_layers,
        layer_kinds,
        max_position_embeddings: raw.max_position_embeddings,
        tie_word_embeddings: raw.tie_word_embeddings,
    };

    Ok(config)
}

/// Parse a `K2HorizonConfig` from an HFQ file's metadata JSON.
/// The HFQ metadata wraps the source `config.json` under the `config` key.
pub fn config_from_hfq(hfq: &hipfire_runtime::hfq::HfqFile) -> Result<K2HorizonConfig, String> {
    let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|e| format!("k2_horizon: metadata_json not valid JSON: {e}"))?;
    let inner = wrapper
        .get("config")
        .ok_or_else(|| "k2_horizon: metadata_json missing `config` wrapper".to_string())?;
    let raw: RawK2HorizonConfig = serde_json::from_value(inner.clone())
        .map_err(|e| format!("k2_horizon: parsing config fields: {e}"))?;
    config_from_raw(raw)
}

/// Parse a `K2HorizonConfig` from a safetensors directory's config.json.
pub fn config_from_safetensors_dir(config_json: &str) -> Result<K2HorizonConfig, String> {
    config_from_json(config_json)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The real K2-Horizon-MoVA-36B-A4B config.json (minified for the test).
    /// Every field verified against
    /// `~/models/RAW/IFM/K2-Horizon-MoVA-36B-A4B/source/config.json`.
    const REAL_CONFIG: &str = r#"{
        "architectures": ["K2HorizonForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_gate_func": "softplus",
        "auto_map": {
            "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
            "AutoModel": "modeling_k2_horizon.K2HorizonModel",
            "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM"
        },
        "bos_token_id": 0,
        "decoder_sparse_step": 1,
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "layernorm_num_groups": 2,
        "max_position_embeddings": 524288,
        "mlp_only_layers": [0, 1, 2],
        "model_type": "k2_horizon",
        "moe_gate_bias": true,
        "moe_intermediate_size": 768,
        "mova_num_experts": 64,
        "mova_num_experts_per_tok": 4,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 100,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_shared_experts": 1,
        "output_router_logits": false,
        "pad_token_id": null,
        "query_key_norm": false,
        "rms_norm_eps": 1e-06,
        "rope_head_dim": 128,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "rope_type": "default"
        },
        "router_aux_loss_coef": 0.001,
        "router_scaling_factor": 2.5,
        "router_score_func": "sigmoid",
        "sliding_window": null,
        "tie_word_embeddings": false,
        "transformers_version": "5.13.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 250624
    }"#;

    #[test]
    fn parses_real_config() {
        let cfg = config_from_json(REAL_CONFIG).expect("parse real config");
        assert_eq!(cfg.dim, 2560);
        assert_eq!(cfg.n_layers, 48);
        assert_eq!(cfg.vocab_size, 250624);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.rope_head_dim, 128);
        assert!((cfg.rope_theta - 10_000_000.0).abs() < 1.0);
        assert_eq!(cfg.attention_bias, false);
        assert_eq!(cfg.layernorm_num_groups, 2);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.num_experts, 100);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 768);
        assert_eq!(cfg.num_shared_experts, 1);
        assert_eq!(cfg.norm_topk_prob, true);
        assert_eq!(cfg.router_score_func, "sigmoid");
        assert!((cfg.router_scaling_factor - 2.5).abs() < 1e-6);
        assert_eq!(cfg.moe_gate_bias, true);
        assert_eq!(cfg.mova_num_experts, 64);
        assert_eq!(cfg.mova_num_experts_per_tok, 4);
        assert_eq!(cfg.attention_gate_func, "softplus");
        assert_eq!(cfg.mlp_only_layers, vec![0, 1, 2]);
        assert_eq!(cfg.max_position_embeddings, 524288);
        assert_eq!(cfg.tie_word_embeddings, false);
        assert_eq!(cfg.eos_token, 1);
    }

    #[test]
    fn layer_kinds_derived_correctly() {
        let cfg = config_from_json(REAL_CONFIG).expect("parse real config");
        assert_eq!(cfg.layer_kinds.len(), 48);
        // Layers 0–2 are Dense
        for i in 0..=2 {
            assert_eq!(
                cfg.layer_kinds[i],
                LayerKind::Dense,
                "layer {i} should be Dense"
            );
        }
        // Layers 3–47 are Moe
        for i in 3..48 {
            assert_eq!(
                cfg.layer_kinds[i],
                LayerKind::Moe,
                "layer {i} should be Moe"
            );
        }
    }

    #[test]
    fn eos_token_array_first_element() {
        let json = r#"{"hidden_size": 2560, "num_hidden_layers": 48, "vocab_size": 250624, "num_attention_heads": 32, "eos_token_id": [1, 250019]}"#;
        let cfg = config_from_json(json).expect("parse");
        assert_eq!(cfg.eos_token, 1);
    }

    #[test]
    fn defaults_when_fields_absent() {
        let json = r#"{"hidden_size": 2560, "num_hidden_layers": 48, "vocab_size": 250624, "num_attention_heads": 32}"#;
        let cfg = config_from_json(json).expect("parse minimal config");
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.mova_num_experts, 0);
        assert_eq!(cfg.mlp_only_layers, Vec::<usize>::new());
        assert_eq!(cfg.layer_kinds.len(), 48);
        // All layers are Moe when mlp_only_layers is empty and num_experts > 0
        // (but num_experts=0 here, so all layers are still classified as Moe
        // — the dense/Moe split is purely from mlp_only_layers).
        assert_eq!(cfg.layer_kinds[0], LayerKind::Moe);
        assert!((cfg.norm_eps - 1e-6).abs() < 1e-10);
        assert!((cfg.rope_theta - 10_000_000.0).abs() < 1.0);
    }

    #[test]
    fn hfq_wrapper_config_key_extracted() {
        // HFQ metadata wraps the source config.json under a `config` key.
        // A bare config (no wrapper) should still parse via config_from_json,
        // but config_from_hfq requires the wrapper.
        let wrapped = format!(r#"{{"config":{REAL_CONFIG}}}"#);
        let value: serde_json::Value = serde_json::from_str(&wrapped).unwrap();
        let inner = value.get("config").unwrap();
        let raw: RawK2HorizonConfig = serde_json::from_value(inner.clone()).unwrap();
        let cfg = config_from_raw(raw).expect("parse wrapped config");
        assert_eq!(cfg.dim, 2560);
        assert_eq!(cfg.n_layers, 48);
    }
}
