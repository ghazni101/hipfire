// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Config for the EmbeddingGemma-2 text encoder (`embedding_gemma2_text`).
//!
//! Parsed from the composite `config.json`'s `text_config` plus the
//! sentence-transformers sidecars in the model dir (`prompts` for the two
//! kinds; no length keys ship with this release, so both default to the
//! model's 8 192-token context window). `per_layer_config` in the
//! serialized config is a map of zero-padded layer-index strings
//! (`"05"`, `"11"`, …) to `{head_dim, num_key_value_heads}` overrides —
//! on this release the four full-attention layers widen to head_dim 512
//! with a single KV head while every other layer keeps the base 256/2.

/// Which sentence-transformers prompt/truncation profile an encode uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Query,
    Document,
}

/// Layer attention kinds, parsed from `text_config.layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// Per-layer attention geometry: resolved once per layer at parse (base
/// values + that layer's `per_layer_config` override).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerAttn {
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

/// The parsed config.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub layer_types: Vec<LayerType>,
    /// Resolved per layer, same length as `layer_types`.
    pub layer_attn: Vec<LayerAttn>,
    pub n_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// Inclusive radius: a sliding layer attends to |i-j| <= window.
    pub sliding_window: usize,
    pub sliding_rope_theta: f32,
    pub full_rope_theta: f32,
    /// PLE width (`hidden_size_per_layer_input`); 0 disables the PLE paths.
    pub ple_dim: usize,
    /// Final projection width (`embedding_dim`).
    pub embedding_dim: usize,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,

    // Multimodal placeholder ids (recognized so they can be refused with a
    // pointing error rather than embedded as text).
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,

    // ── sentence-transformers profile ──────────────────────────────────
    /// Task-instruction prefix prepended to the raw text per kind.
    pub query_prompt: String,
    pub document_prompt: String,
    /// Max token count (prompt + bos + eos included) per kind.
    pub query_length: usize,
    pub document_length: usize,
}

/// This release's defaults: the model card's prompt prefixes and the 8 192
/// shared context window. A checkpoint carrying its own values in
/// `config_sentence_transformers.json` / `sentence_bert_config.json` wins.
pub const DEFAULT_QUERY_PROMPT: &str = "task: search result | query: ";
pub const DEFAULT_DOCUMENT_PROMPT: &str = "title: none | text: ";
pub const DEFAULT_QUERY_LENGTH: usize = 8192;
pub const DEFAULT_DOCUMENT_LENGTH: usize = 8192;

fn u64_of(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn f64_of(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

/// Unwrap a checkpoint-metadata blob to the composite config object (both
/// source kinds wrap the checkpoint config under `"config"`).
pub fn config_from_metadata(meta: &serde_json::Value) -> &serde_json::Value {
    meta.get("config").unwrap_or(meta)
}

fn parse_layer_types(text: &serde_json::Value, num_layers: usize) -> Result<Vec<LayerType>, String> {
    let raw = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            "text_config.layer_types is required (the builder emits it; a \
             hand-written config must spell every layer out)"
                .to_string()
        })?;
    if raw.len() != num_layers {
        return Err(format!(
            "text_config.layer_types has {} entries for num_hidden_layers {num_layers}",
            raw.len()
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(i, v)| match v.as_str().unwrap_or("") {
            "sliding_attention" => Ok(LayerType::Sliding),
            "full_attention" => Ok(LayerType::Full),
            other => Err(format!(
                "layer_types[{i}] = {other:?}: embeddinggemma2 supports \
                 sliding_attention / full_attention"
            )),
        })
        .collect()
}

/// Parse a `per_layer_config` key as a layer index. HF serializes the keys
/// zero-padded (`"05"`); accept plain digits too.
fn layer_index_key(key: &str) -> Option<usize> {
    key.parse::<usize>().ok().or_else(|| {
        let t = key.trim_start_matches('0');
        if t.is_empty() {
            Some(0)
        } else {
            t.parse::<usize>().ok()
        }
    })
}

/// Parse the composite `config.json` (a `serde_json::Value` of its contents).
pub fn parse_config(config: &serde_json::Value) -> Result<EmbedConfig, String> {
    let text = config.get("text_config").ok_or(
        "config.json has no text_config (embeddinggemma2 is a composite model)",
    )?;

    let hidden_size = u64_of(text, "hidden_size").ok_or("text_config.hidden_size")? as usize;
    let num_layers =
        u64_of(text, "num_hidden_layers").ok_or("text_config.num_hidden_layers")? as usize;
    let n_heads =
        u64_of(text, "num_attention_heads").ok_or("num_attention_heads")? as usize;
    let base_head_dim = u64_of(text, "head_dim").unwrap_or(0) as usize;
    let base_kv = u64_of(text, "num_key_value_heads").unwrap_or(0) as usize;
    if n_heads == 0 || base_kv == 0 {
        return Err(format!(
            "num_attention_heads={n_heads} / num_key_value_heads={base_kv}: both must be > 0"
        ));
    }
    let head_dim = if base_head_dim == 0 {
        hidden_size / n_heads
    } else {
        base_head_dim
    };
    let intermediate_size =
        u64_of(text, "intermediate_size").ok_or("intermediate_size")? as usize;
    let vocab_size = u64_of(text, "vocab_size").ok_or("vocab_size")? as usize;
    let rms_norm_eps = f64_of(text, "rms_norm_eps").unwrap_or(1e-6) as f32;
    let sliding_window = u64_of(text, "sliding_window").unwrap_or(512) as usize;

    let act = text
        .get("hidden_activation")
        .or_else(|| text.get("hidden_act"))
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh");
    if act != "gelu_pytorch_tanh" {
        return Err(format!("hidden_activation {act:?}: only gelu_pytorch_tanh is supported"));
    }
    if text.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
        return Err("attention_bias=true is not supported (bias-free projections)".into());
    }
    // This arch has no `is_causal` key at all — attention is bidirectional by
    // construction. A future checkpoint that declares causal attention would
    // be silently wrong here, so refuse it.
    if text.get("is_causal").and_then(|v| v.as_bool()) == Some(true) {
        return Err("is_causal=true is not the bidirectional encoder this arch implements".into());
    }

    // rope_parameters nests per-layer-type theta.
    let rope = text.get("rope_parameters").cloned().unwrap_or_default();
    let rope_theta_of = |ty: &str, default: f64| -> f64 {
        rope.get(ty)
            .and_then(|r| f64_of(r, "rope_theta"))
            .unwrap_or(default)
    };
    let sliding_rope_theta = rope_theta_of("sliding_attention", 10_000.0) as f32;
    let full_rope_theta = rope_theta_of("full_attention", 1_000_000.0) as f32;

    let layer_types = parse_layer_types(text, num_layers)?;

    // Resolve per-layer geometry: base (head_dim, kv) overridden by
    // per_layer_config entries keyed by layer index.
    let mut layer_attn = vec![
        LayerAttn {
            head_dim,
            n_kv_heads: base_kv,
        };
        num_layers
    ];
    if let Some(plc) = text.get("per_layer_config").and_then(|v| v.as_object()) {
        for (key, ov) in plc {
            let Some(li) = layer_index_key(key) else {
                // Per-type entries ("sliding_attention") only shape building
                // upstream; the serialized per-layer values are authoritative.
                continue;
            };
            if li >= num_layers {
                return Err(format!(
                    "per_layer_config[{key}] = layer {li} >= num_hidden_layers {num_layers}"
                ));
            }
            let a = &mut layer_attn[li];
            if let Some(hd) = u64_of(ov, "head_dim") {
                a.head_dim = hd as usize;
            }
            if let Some(kv) = u64_of(ov, "num_key_value_heads") {
                a.n_kv_heads = kv as usize;
            }
        }
    }
    for (li, a) in layer_attn.iter().enumerate() {
        if a.head_dim == 0 || a.head_dim % 2 != 0 || a.head_dim % 64 != 0 {
            return Err(format!(
                "layer {li}: head_dim {} must be an even multiple of 64 \
                 (rope pairs + the WMMA GEMM's K tile)",
                a.head_dim
            ));
        }
        if a.n_kv_heads == 0 || n_heads % a.n_kv_heads != 0 {
            return Err(format!(
                "layer {li}: {n_heads} heads not divisible by {} kv heads",
                a.n_kv_heads
            ));
        }
    }
    if hidden_size % 16 != 0 || intermediate_size % 16 != 0 {
        return Err(format!(
            "hidden_size {hidden_size} / intermediate_size {intermediate_size} must be \
             multiples of 16 (GEMM contract)"
        ));
    }

    let ple_dim = u64_of(text, "hidden_size_per_layer_input").unwrap_or(0) as usize;
    if ple_dim != 0 && ple_dim % 16 != 0 {
        return Err(format!("hidden_size_per_layer_input {ple_dim} must be a multiple of 16"));
    }
    let embedding_dim = u64_of(text, "embedding_dim").ok_or("text_config.embedding_dim")? as usize;
    if embedding_dim % 16 != 0 {
        return Err(format!("embedding_dim {embedding_dim} must be a multiple of 16"));
    }

    let bos_token_id = u64_of(text, "bos_token_id").unwrap_or(2) as u32;
    let eos_token_id = u64_of(text, "eos_token_id").unwrap_or(1) as u32;
    let pad_token_id = u64_of(text, "pad_token_id").unwrap_or(0) as u32;
    let image_token_id = u64_of(config, "image_token_id").unwrap_or(258_880) as u32;
    let video_token_id = u64_of(config, "video_token_id").unwrap_or(258_884) as u32;
    let audio_token_id = u64_of(config, "audio_token_id").unwrap_or(258_881) as u32;
    for (name, id) in [
        ("bos", bos_token_id),
        ("eos", eos_token_id),
        ("image", image_token_id),
        ("video", video_token_id),
        ("audio", audio_token_id),
    ] {
        if id as usize >= vocab_size {
            return Err(format!("{name}_token_id {id} >= vocab_size {vocab_size}"));
        }
    }

    Ok(EmbedConfig {
        hidden_size,
        num_layers,
        layer_types,
        layer_attn,
        n_heads,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        sliding_window,
        sliding_rope_theta,
        full_rope_theta,
        ple_dim,
        embedding_dim,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        image_token_id,
        video_token_id,
        audio_token_id,
        query_prompt: DEFAULT_QUERY_PROMPT.to_string(),
        document_prompt: DEFAULT_DOCUMENT_PROMPT.to_string(),
        query_length: DEFAULT_QUERY_LENGTH,
        document_length: DEFAULT_DOCUMENT_LENGTH,
    })
}

impl EmbedConfig {
    /// Overlay `sentence_bert_config.json` / the folded HFQ sidecar (same key
    /// schema). Applied before [`Self::apply_st_config`], whose keys win.
    pub fn apply_sbert_config(&mut self, sb: &serde_json::Value) {
        self.apply_st_config(sb);
    }

    /// Overlay `config_sentence_transformers.json` (or the `sentence_transformers`
    /// object the quantizer folds into an `.hfq`): the two kinds' prompt
    /// prefixes and, when a release ships them, its own length limits.
    pub fn apply_st_config(&mut self, st: &serde_json::Value) {
        if let Some(prompts) = st.get("prompts").and_then(|v| v.as_object()) {
            if let Some(q) = prompts.get("query").and_then(|v| v.as_str()) {
                self.query_prompt = q.to_string();
            }
            if let Some(d) = prompts.get("document").and_then(|v| v.as_str()) {
                self.document_prompt = d.to_string();
            }
        }
        if let Some(q) = u64_of(st, "query_length") {
            self.query_length = q as usize;
        }
        if let Some(d) = u64_of(st, "document_length") {
            self.document_length = d as usize;
        }
    }

    pub fn kind_prompt(&self, kind: InputKind) -> &str {
        match kind {
            InputKind::Query => &self.query_prompt,
            InputKind::Document => &self.document_prompt,
        }
    }

    pub fn kind_length(&self, kind: InputKind) -> usize {
        match kind {
            InputKind::Query => self.query_length,
            InputKind::Document => self.document_length,
        }
    }

    /// Attention geometry of one layer.
    pub fn attn(&self, layer: usize) -> LayerAttn {
        self.layer_attn[layer]
    }

    /// Whether a sliding layer's window actually restricts anything at this
    /// length: |i-j| <= window is vacuous once n-1 <= window.
    pub fn window_bites(&self, layer: usize, n: usize) -> bool {
        self.layer_types[layer] == LayerType::Sliding && n > self.sliding_window + 1
    }

    pub fn rope_theta(&self, layer: usize) -> f32 {
        match self.layer_types[layer] {
            LayerType::Sliding => self.sliding_rope_theta,
            LayerType::Full => self.full_rope_theta,
        }
    }

    pub fn q_dim(&self, layer: usize) -> usize {
        self.n_heads * self.layer_attn[layer].head_dim
    }

    pub fn kv_dim(&self, layer: usize) -> usize {
        self.layer_attn[layer].n_kv_heads * self.layer_attn[layer].head_dim
    }

    /// The embedding scale applied to `embed_tokens` output: √hidden (HF's
    /// `TextScaledWordEmbedding`).
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    /// PLE projection scale: hidden^-0.5 (HF's `per_layer_model_projection_scale`).
    pub fn ple_scale(&self) -> f32 {
        (self.hidden_size as f32).recip().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composite config.json of google/embeddinggemma-2, trimmed to the
    /// fields the parser reads (values verbatim).
    const EG2_CONFIG: &str = r#"{
      "architectures": ["EmbeddingGemma2Model"],
      "model_type": "embedding_gemma2",
      "text_config": {
        "attention_bias": false,
        "bos_token_id": 2,
        "embedding_dim": 768,
        "eos_token_id": 1,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 512,
        "hidden_size_per_layer_input": 512,
        "intermediate_size": 2048,
        "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "max_position_embeddings": 262144,
        "model_type": "embedding_gemma2_text",
        "num_attention_heads": 4,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "per_layer_config": {
          "05": {"head_dim": 512, "num_key_value_heads": 1},
          "11": {"head_dim": 512, "num_key_value_heads": 1},
          "17": {"head_dim": 512, "num_key_value_heads": 1},
          "23": {"head_dim": 512, "num_key_value_heads": 1}
        },
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "sliding_window": 512,
        "vocab_size": 262144
      }
    }"#;

    fn parse() -> EmbedConfig {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        parse_config(&v).unwrap()
    }

    #[test]
    fn parses_the_release_config() {
        let cfg = parse();
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.layer_types[5], LayerType::Full);
        assert_eq!(cfg.layer_types[23], LayerType::Full);
        assert_eq!(cfg.layer_types[0], LayerType::Sliding);
        // Per-layer overrides: full layers widen to hd 512 / 1 kv head, the
        // rest keep the 256/2 base.
        assert_eq!(cfg.attn(0), LayerAttn { head_dim: 256, n_kv_heads: 2 });
        assert_eq!(cfg.attn(5), LayerAttn { head_dim: 512, n_kv_heads: 1 });
        assert_eq!(cfg.attn(23), LayerAttn { head_dim: 512, n_kv_heads: 1 });
        assert_eq!(cfg.q_dim(0), 1024);
        assert_eq!(cfg.kv_dim(0), 512);
        assert_eq!(cfg.q_dim(5), 2048);
        assert_eq!(cfg.kv_dim(5), 512);
        assert_eq!(cfg.rope_theta(0), 10_000.0);
        assert_eq!(cfg.rope_theta(5), 1_000_000.0);
        assert_eq!(cfg.ple_dim, 512);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.bos_token_id, 2);
        assert_eq!(cfg.eos_token_id, 1);
        assert!((cfg.embed_scale() - (512.0f32).sqrt()).abs() < 1e-6);
        // Window is a radius: it only bites past window+1 rows.
        assert!(!cfg.window_bites(0, 513));
        assert!(cfg.window_bites(0, 514));
        assert!(!cfg.window_bites(5, 8192)); // full layer never masked
        // Sidecar defaults before overlays.
        assert_eq!(cfg.query_prompt, "task: search result | query: ");
        assert_eq!(cfg.document_prompt, "title: none | text: ");
        assert_eq!(cfg.query_length, 8192);
    }

    #[test]
    fn sidecar_overlay_applies() {
        let mut cfg = parse();
        let st: serde_json::Value = serde_json::from_str(
            r#"{"prompts":{"query":"task: clustering | query: ",
                "document":"title: t | text: "},
                "query_length":128,"document_length":512}"#,
        )
        .unwrap();
        cfg.apply_st_config(&st);
        assert_eq!(cfg.query_prompt, "task: clustering | query: ");
        assert_eq!(cfg.document_prompt, "title: t | text: ");
        assert_eq!(cfg.query_length, 128);
        assert_eq!(cfg.document_length, 512);
    }

    #[test]
    fn rejects_bad_layer_types_and_geometry() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let mut bad = v.clone();
        bad["text_config"]["layer_types"][0] = serde_json::json!("linear_attention");
        let err = parse_config(&bad).unwrap_err();
        assert!(err.contains("linear_attention"), "{err}");

        // head_dim must be an even multiple of 64 (the GEMM K tile + rope pairs).
        let mut bad2 = v.clone();
        bad2["text_config"]["per_layer_config"]["05"]["head_dim"] = serde_json::json!(200);
        let err = parse_config(&bad2).unwrap_err();
        assert!(err.contains("head_dim"), "{err}");

        // kv heads must divide the head count.
        let mut bad3 = v.clone();
        bad3["text_config"]["per_layer_config"]["05"]["num_key_value_heads"] = serde_json::json!(3);
        let err = parse_config(&bad3).unwrap_err();
        assert!(err.contains("divisible"), "{err}");

        // A per-layer override past the layer count is refused, not ignored.
        let mut bad4 = v.clone();
        bad4["text_config"]["per_layer_config"]["99"] =
            serde_json::json!({"head_dim": 512});
        let err = parse_config(&bad4).unwrap_err();
        assert!(err.contains("layer 99"), "{err}");

        // Non-tanh activation, attention bias, causal: all refused.
        let mut bad5 = v.clone();
        bad5["text_config"]["hidden_activation"] = serde_json::json!("silu");
        assert!(parse_config(&bad5).unwrap_err().contains("gelu_pytorch_tanh"));
        let mut bad6 = v.clone();
        bad6["text_config"]["attention_bias"] = serde_json::json!(true);
        assert!(parse_config(&bad6).unwrap_err().contains("attention_bias"));
        let mut bad7 = v.clone();
        bad7["text_config"]["is_causal"] = serde_json::json!(true);
        assert!(parse_config(&bad7).unwrap_err().contains("is_causal"));
    }

    #[test]
    fn config_from_metadata_unwraps_both_shapes() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let wrapped = serde_json::json!({"architecture": "embedding_gemma2", "config": v.clone()});
        assert_eq!(parse_config(config_from_metadata(&wrapped)).unwrap().num_layers, 24);
        assert_eq!(parse_config(config_from_metadata(&v)).unwrap().num_layers, 24);
    }
}
