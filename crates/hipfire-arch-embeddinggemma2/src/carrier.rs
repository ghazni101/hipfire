// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_embed_gemma_bundle` — the entry point the loader's
//! `EmbeddingGemma2Carrier` calls, plus the encode-side request assembly.
//!
//! Config comes from the source's `metadata_json` (dir: `config.json`
//! contents; HFQ: embedded metadata) with the sentence-transformers
//! sidecars overlaid from the model dir (dir) or the quantizer's folded
//! `sentence_transformers` object (HFQ). The tokenizer loads from
//! `tokenizer.json` / HFQ metadata and is switched to the exact
//! merge-rank SP-BPE path this vocabulary requires (see
//! `Tokenizer::enable_sp_bpe_merges`); BOS/EOS are pinned from the model
//! config, whose ids (2/1) the generic vocab heuristics cannot infer.

use crate::bundle::EmbedGemmaBundle;
use crate::config::{parse_config, EmbedConfig, InputKind};
use crate::weights::{layer_type_summary, load_weights};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::tokenizer::Tokenizer;

fn overlay_sidecars(cfg: &mut EmbedConfig, dir: &std::path::Path) {
    let read_json = |rel: &str| -> Option<serde_json::Value> {
        let p = dir.join(rel);
        let s = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&s).ok()
    };
    if let Some(sb) = read_json("sentence_bert_config.json") {
        cfg.apply_sbert_config(&sb);
    }
    if let Some(st) = read_json("config_sentence_transformers.json") {
        cfg.apply_st_config(&st);
    }
}

/// Apply the `sentence_transformers` object the quantizer folds into an
/// `.hfq`'s metadata (same key schema [`overlay_sidecars`] reads).
fn apply_folded_sidecars(cfg: &mut EmbedConfig, folded: &serde_json::Value) {
    cfg.apply_st_config(folded);
}

/// Load + fix up the tokenizer for either source kind.
fn load_tokenizer(st: &dyn ModelSourceTrait, cfg: &EmbedConfig) -> Result<Tokenizer, String> {
    let mut tok = match st.tokenizer_json_path() {
        Some(p) => Tokenizer::from_tokenizer_json(&p)
            .map_err(|e| format!("embeddinggemma2: tokenizer: {e}"))?
            .ok_or("embeddinggemma2: tokenizer.json present but unparseable")?,
        None => Tokenizer::from_hfq_metadata(st.metadata_json())
            .map_err(|e| format!("embeddinggemma2: tokenizer from metadata: {e}"))?,
    };
    // This vocabulary is a ▁-style BPE model, and the generic SentencePiece
    // greedy-longest path is not merge-rank BPE: on ordinary text it takes a
    // longer literal piece than the merge ranks form, so it diverges from HF
    // `tokenizers` (measured 8/29 sampled prompts, and 0/9 of this arch's
    // parity fixtures). Switch to the exact merge-rank path with byte
    // fallback. Both source kinds (dir and `.hfq`) reach this through here —
    // nothing else enables it — and the guard below fails the load outright
    // rather than let an arch-18 encode fall back to greedy.
    tok.enable_sp_bpe_merges();
    // A Gemma-shaped tokenizer.json with an EMPTY merge list would silently
    // keep the greedy SentencePiece path — the exact HF divergence this arch
    // exists to avoid — so fail the load instead.
    if !tok.sp_bpe_merges_enabled() {
        return Err(
            "embeddinggemma2: this tokenizer did not engage the merge-rank SP-BPE \
             path (no merge rules in tokenizer.json?) — the greedy fallback \
             diverges from HF on this vocabulary, refusing to load"
                .into(),
        );
    }
    // The generic bos/eos heuristics (`<|endoftext|>` / `<s>`) do not know
    // Gemma's <bos>=2 / <eos>=1; the model config's ids are authoritative.
    tok.bos_id = cfg.bos_token_id;
    tok.eos_id = cfg.eos_token_id;
    // The HF post-processor template adds <bos> … <eos>; the runtime's
    // generic add_bos would double it because assembly below adds both
    // explicitly.
    tok.add_bos = false;
    // The invariant that matters: no token id the tokenizer can EMIT may
    // exceed the embedding table (`vocab_size()` itself over-reports — the
    // runtime pads its vocab vector past the last real id).
    if let Some(max_id) = tok.max_token_id() {
        if max_id as usize >= cfg.vocab_size {
            return Err(format!(
                "embeddinggemma2: tokenizer can emit id {max_id} >= config vocab {} — \
                 that token would have no embedding row",
                cfg.vocab_size
            ));
        }
    }
    Ok(tok)
}

pub fn load_embed_gemma_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx<'_>,
) -> Result<EmbedGemmaBundle, String> {
    load_embed_gemma_from_source(ctx.gpu, src)
}

/// The loader body without [`LoadCtx`] (the slot engine calls this
/// directly; it owns its own GPU and has no load context).
pub fn load_embed_gemma_from_source(
    gpu: &mut rdna_compute::Gpu,
    src: ModelSource,
) -> Result<EmbedGemmaBundle, String> {
    let st: &dyn ModelSourceTrait = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    let meta: serde_json::Value = serde_json::from_str(st.metadata_json())
        .map_err(|e| format!("embeddinggemma2: config JSON: {e}"))?;
    let mut config = parse_config(crate::config::config_from_metadata(&meta))?;

    if src.is_dir() {
        overlay_sidecars(&mut config, st.path());
    } else if let Some(folded) = meta.get("sentence_transformers") {
        apply_folded_sidecars(&mut config, folded);
    }

    let tokenizer = load_tokenizer(st, &config)?;

    // PLE tensors are required exactly when the config declares a PLE width
    // — a checkpoint/config mismatch is a load error, not a silent
    // zero-filled PLE.
    let has_ple = crate::weights::EmbedWeights::source_has_ple(&src);
    if (config.ple_dim > 0) != has_ple {
        return Err(format!(
            "embeddinggemma2: config declares hidden_size_per_layer_input={} but the \
             checkpoint {} the ple tensors",
            config.ple_dim,
            if has_ple { "has" } else { "lacks" }
        ));
    }

    let weights = load_weights(gpu, &src, &config)?;

    eprintln!(
        "  embeddinggemma2: {} layers ({}), hidden {}, embed {}→{}, window {}, \
         query/doc max {}/{} tokens, text-only (vision/audio towers not loaded)",
        config.num_layers,
        layer_type_summary(&config),
        config.hidden_size,
        config.hidden_size,
        config.embedding_dim,
        config.sliding_window,
        config.query_length,
        config.document_length,
    );

    if std::env::var("HIPFIRE_EMBED_DEBUG").ok().as_deref() == Some("1") {
        let (f, t) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        eprintln!(
            "  [embed-debug] after load: VRAM {} MB free / {} MB",
            f / (1024 * 1024),
            t / (1024 * 1024)
        );
    }
    Ok(EmbedGemmaBundle {
        config,
        weights,
        tokenizer,
    })
}

/// The encode-side request assembly shared by the CLI, daemon verb and HTTP
/// layer: `[bos] prompt+text [eos]`, the body truncated so the whole
/// sequence fits the kind's limit.
pub struct AssembledInput {
    pub ids: Vec<u32>,
}

/// Tokenize + truncate one input.
///
/// Sequence shape (mirrors the checkpoint's HF post-processor +
/// sentence-transformers prompt): `[bos] [prompt ∘ text] [eos]`, truncation
/// cutting **body tokens only** from the end so the frame survives — the
/// same truncation shape the HF tokenizer applies at `max_length`
/// (special-token-aware). An input whose *prompt alone* already exceeds the
/// limit is an error rather than a mid-prompt cut.
pub fn assemble_input(
    bundle: &EmbedGemmaBundle,
    kind: InputKind,
    text: &str,
) -> Result<AssembledInput, String> {
    let cfg = &bundle.config;
    let limit = cfg.kind_length(kind);
    let framed = format!("{}{}", cfg.kind_prompt(kind), text);
    let body = bundle.tokenizer.encode(&framed);
    if body.is_empty() {
        return Err(
            "embeddinggemma2: input tokenized to zero tokens (empty text after the \
             task prompt?)"
                .into(),
        );
    }
    let bos = cfg.bos_token_id;
    let eos = cfg.eos_token_id;
    // The frame (bos + eos) is never truncated away.
    let room = limit.saturating_sub(2);
    // A prompt that alone overflows the limit is a configuration mismatch —
    // the encode would be nothing but a truncated prompt. Checked on the
    // prompt's OWN tokenization (the framed body check below cannot see it:
    // truncation makes any body fit).
    let prompt_ids = bundle.tokenizer.encode(cfg.kind_prompt(kind));
    if prompt_ids.len() > room {
        return Err(format!(
            "embeddinggemma2: kind prompt tokenizes to {} tokens, leaving no room \
             for text under the {} limit (check the checkpoint's prompt/length \
             sidecars)",
            prompt_ids.len(),
            limit
        ));
    }
    let ids = if body.len() > room {
        let mut ids = Vec::with_capacity(limit);
        ids.push(bos);
        ids.extend_from_slice(&body[..room]);
        ids.push(eos);
        ids
    } else {
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(bos);
        ids.extend_from_slice(&body);
        ids.push(eos);
        ids
    };
    if ids.len() < 3 {
        return Err(format!(
            "embeddinggemma2: input framed to {} tokens under the {} limit — \
             nothing left to pool",
            ids.len(),
            limit
        ));
    }
    Ok(AssembledInput { ids })
}

/// Slot-mode admission preflight for an arch-18 checkpoint: geometry, arch
/// and tokenizer, read without touching the GPU.
///
/// `Ok(None)` = not an arch-18 source (the caller falls through).
pub fn preflight_slot(
    model_path: &std::path::Path,
) -> Result<Option<hipfire_runtime::serve::SlotModelPreflight>, String> {
    use hipfire_runtime::serve::SlotModelPreflight;

    let Some(path) = model_path.to_str() else {
        return Ok(None);
    };
    let src = match ModelSource::from_path(path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let st: &dyn ModelSourceTrait = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    if st.arch_id() != crate::ARCH_ID {
        return Ok(None);
    }
    let meta: serde_json::Value = serde_json::from_str(st.metadata_json())
        .map_err(|e| format!("embeddinggemma2: config JSON: {e}"))?;
    let mut config = parse_config(crate::config::config_from_metadata(&meta))?;
    if src.is_dir() {
        overlay_sidecars(&mut config, st.path());
    } else if let Some(folded) = meta.get("sentence_transformers") {
        apply_folded_sidecars(&mut config, folded);
    }
    let tokenizer = load_tokenizer(st, &config)?;
    Ok(Some(SlotModelPreflight {
        arch_id: crate::ARCH_ID,
        arch_str: "embedding_gemma2".to_string(),
        dim: config.hidden_size,
        layers: config.num_layers,
        vocab: config.vocab_size,
        tokenizer,
        is_vl: false,
        vl_path: None,
    }))
}
