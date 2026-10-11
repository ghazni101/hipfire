// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_qwen35_embed_bundle` — the entry point the loader's
//! `Qwen35EmbedCarrier` calls (the `load_minimax_bundle` shape).
//!
//! Config comes from the source's `metadata_json` (dir: config.json
//! contents; HFQ: embedded metadata — the quantizer preserves the composite
//! JSON), with the sentence-transformers sidecars overlaid from the model
//! dir when present. The tokenizer loads from `tokenizer.json` (dir) or the
//! HFQ metadata. Marker ids are resolved by encoding the prompt strings and
//! requiring a single token each (`"[Q] "` / `"[D] "` are added tokens).

use crate::bundle::{EmbedBundle, MarkerIds};
use crate::config::{parse_config, EmbedConfig, InputKind};
use crate::post::{skiplist_token_ids, VocabPieces};
use crate::state::GdnEncodeState;
use crate::weights::load_weights;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::tokenizer::Tokenizer;

struct BundleVocab<'a>(&'a Tokenizer);

impl VocabPieces for BundleVocab<'_> {
    fn pieces(&self) -> Vec<(u32, String)> {
        self.0
            .vocab()
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p.clone()))
            .collect()
    }
}

fn overlay_sidecars(cfg: &mut EmbedConfig, dir: &std::path::Path) {
    let read_json = |rel: &str| -> Option<serde_json::Value> {
        let p = dir.join(rel);
        let s = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&s).ok()
    };
    if let Some(st) = read_json("config_sentence_transformers.json") {
        cfg.apply_st_config(&st);
    }
    if let Some(mask) = read_json("2_MultiVectorMask/config.json") {
        cfg.apply_mask_config(&mask);
    }
    if let Some(dense) = read_json("1_Dense/config.json") {
        cfg.apply_dense_config(&dense);
    }
}

/// Resolve one marker prompt to a single token id.
fn marker_id(tok: &Tokenizer, prompt: &str, which: &str) -> Result<u32, String> {
    let ids = tok.encode(prompt);
    if ids.len() == 1 {
        return Ok(ids[0]);
    }
    // Some exports ship the marker without the trailing space as the added
    // token and encode the space separately; accept [marker, space-piece]
    // only when the first id's piece starts the prompt.
    if ids.len() == 2 {
        let piece = tok
            .vocab()
            .get(ids[0] as usize)
            .cloned()
            .unwrap_or_default();
        if prompt.starts_with(&piece) {
            return Ok(ids[0]);
        }
    }
    Err(format!(
        "qwen35-embed: {which} marker {prompt:?} tokenized to {} pieces \
         (expected one added token)",
        ids.len()
    ))
}

pub fn load_qwen35_embed_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx<'_>,
) -> Result<EmbedBundle, String> {
    let st: &dyn ModelSourceTrait = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    let meta: serde_json::Value = serde_json::from_str(st.metadata_json())
        .map_err(|e| format!("qwen35-embed: config JSON: {e}"))?;
    let mut config = parse_config(crate::config::config_from_metadata(&meta))?;

    // Sidecar overlays live next to the checkpoint (dir sources; HFQ
    // re-saves fold them in via the defaults).
    if src.is_dir() {
        overlay_sidecars(&mut config, st.path());
    }

    // Tokenizer: dir file, else HFQ metadata.
    let tokenizer = match st.tokenizer_json_path() {
        Some(p) => Tokenizer::from_tokenizer_json(&p)
            .map_err(|e| format!("qwen35-embed: tokenizer: {e}"))?
            .ok_or("qwen35-embed: tokenizer.json present but unparseable")?,
        None => Tokenizer::from_hfq_metadata(st.metadata_json())
            .map_err(|e| format!("qwen35-embed: tokenizer from metadata: {e}"))?,
    };
    // The embedding table may be LARGER than the tokenizer's vocab (reserved
    // / padded rows): the config `vocab_size` (248,320 on the 0.6B) exceeds
    // the tokenizer.json entry count (248,144), and that is normal. The
    // direction that matters is the other one — a tokenizer that can emit an
    // id past the embedding table would read out of range, so refuse that.
    if tokenizer.vocab_size() > config.vocab_size {
        return Err(format!(
            "qwen35-embed: tokenizer vocab {} exceeds config vocab {} — some token              ids would have no embedding row",
            tokenizer.vocab_size(),
            config.vocab_size
        ));
    }

    let markers = MarkerIds {
        query: marker_id(&tokenizer, &config.query_prompt, "query")?,
        document: marker_id(&tokenizer, &config.document_prompt, "document")?,
    };

    let skiplist_ids = skiplist_token_ids(&BundleVocab(&tokenizer), &config.skiplist_words);

    let weights = load_weights(ctx.gpu, &src, &config)?;
    let state = GdnEncodeState::new(ctx.gpu, &config)?;

    eprintln!(
        "  qwen35-embed: {} layers ({} GDN / {} full-attn, bidirectional), dense {}→{}, \
         vision {}, query/doc max {}/{} tokens",
        config.num_layers,
        config
            .layer_types
            .iter()
            .filter(|t| **t == crate::config::LayerType::LinearAttention)
            .count(),
        config
            .layer_types
            .iter()
            .filter(|t| **t == crate::config::LayerType::FullAttention)
            .count(),
        config.hidden_size,
        config.dense_out_features,
        if weights.vision.is_some() { "on" } else { "off" },
        config.query_length,
        config.document_length,
    );

    Ok(EmbedBundle {
        config,
        weights,
        state,
        tokenizer,
        markers,
        skiplist_ids,
    })
}

/// The encode-side request assembly shared by the CLI, daemon verb and HTTP
/// layer: marker + (optional) vision block + text, truncated to the kind's
/// limit with the marker and vision block always kept whole.
pub struct AssembledInput {
    pub ids: Vec<u32>,
    /// `[n][3]` mRoPE positions.
    pub pos3: Vec<[i32; 3]>,
    /// Merged image embeds (empty for text-only).
    pub image_embeds: Vec<f32>,
    pub spans: Vec<crate::weights::ImageSpan>,
}

/// Tokenize + splice + truncate one input.
///
/// Sequence shape (mirrors the checkpoint's chat template + ST prompt):
/// `[marker] [vision_start image_pad×n_merged vision_end]? [text]`.
/// Truncation cuts **text tokens only**, from the end; a vision block that
/// alone overflows the kind's limit is an error (the splice would break).
pub fn assemble_input(
    bundle: &EmbedBundle,
    gpu: &mut rdna_compute::Gpu,
    kind: InputKind,
    text: &str,
    image_bytes: Option<&[u8]>,
) -> Result<AssembledInput, String> {
    let cfg = &bundle.config;
    let limit = cfg.kind_length(kind);
    let mut ids = vec![bundle.marker_id(kind)];
    let mut spans = Vec::new();
    let mut image_embeds = Vec::new();

    if let Some(bytes) = image_bytes {
        let vis_w = bundle
            .weights
            .vision
            .as_ref()
            .ok_or("qwen35-embed: image input but this checkpoint has no vision tower")?;
        let (merged, mh, mw) = crate::vision::encode_image(gpu, vis_w, cfg, bytes)?;
        let n_merged = mh * mw;
        // Vision delimiters + pads.
        ids.push(cfg.vision_start_token_id);
        let span_start = ids.len();
        for _ in 0..n_merged {
            ids.push(cfg.image_token_id);
        }
        ids.push(cfg.vision_end_token_id);
        spans.push(crate::weights::ImageSpan {
            start: span_start,
            len: n_merged,
            embed_offset: 0,
            grid_h: mh,
            grid_w: mw,
        });
        image_embeds = merged;
        if ids.len() + 1 > limit {
            return Err(format!(
                "qwen35-embed: image expands to {} tokens, over the {} limit for this kind",
                ids.len() + 1,
                limit
            ));
        }
    }

    if !text.is_empty() {
        let text_ids = bundle.tokenizer.encode(text);
        let room = limit.saturating_sub(ids.len());
        ids.extend(text_ids.into_iter().take(room));
    }
    if ids.len() > limit {
        return Err(format!(
            "qwen35-embed: input is {} tokens, over the {} limit",
            ids.len(),
            limit
        ));
    }

    let pos3 = build_positions(ids.len(), &spans);
    Ok(AssembledInput {
        ids,
        pos3,
        image_embeds,
        spans,
    })
}

/// Build positions: text rows `[p,p,p]`; each image span gets HF
/// `get_rope_index` phases (t fixed at the span's base, h/w the merged
/// grid) and advances the cursor by `max(grid_h, grid_w)`.
pub fn build_positions(
    n_tokens: usize,
    spans: &[crate::weights::ImageSpan],
) -> Vec<[i32; 3]> {
    let sp: Vec<hipfire_arch_qwen35_vl::mrope::ImageSpan> = spans
        .iter()
        .map(|s| hipfire_arch_qwen35_vl::mrope::ImageSpan {
            start: s.start,
            len: s.len,
            grid_h: s.grid_h,
            grid_w: s.grid_w,
        })
        .collect();
    let merge = 2;
    hipfire_arch_qwen35_vl::mrope::build_mrope_positions(n_tokens, &sp, merge).positions
}
