// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The loaded-model bundle: what the loader boxes as `Box<dyn ArchModel>`
//! and what the embed request path downcasts back to.

use crate::config::{EmbedConfig, InputKind};
use crate::post::VocabPieces;
use crate::state::GdnEncodeState;
use crate::weights::EmbedWeights;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

pub use crate::config::InputKind as Kind;

/// Marker token ids resolved from the tokenizer at load (`"[Q] "` /
/// `"[D] "` are single added tokens).
#[derive(Clone, Copy, Debug)]
pub struct MarkerIds {
    pub query: u32,
    pub document: u32,
}

pub struct EmbedBundle {
    pub config: EmbedConfig,
    pub weights: EmbedWeights,
    pub state: GdnEncodeState,
    pub tokenizer: Tokenizer,
    pub markers: MarkerIds,
    /// Token ids whose piece is a skiplist word (see [`crate::post`]).
    pub skiplist_ids: std::collections::HashSet<u32>,
}

/// `Tokenizer` vocab access for [`crate::post::skiplist_token_ids`].
struct TokVocab<'a>(&'a Tokenizer);

impl VocabPieces for TokVocab<'_> {
    fn pieces(&self) -> Vec<(u32, String)> {
        self.0
            .vocab()
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p.clone()))
            .collect()
    }
}

impl EmbedBundle {
    /// Encode one token id for the given kind's marker, or `None` when the
    /// tokenizer lacks it (the encode paths then refuse with a pointing
    /// error rather than silently dropping the marker).
    pub fn marker_id(&self, kind: InputKind) -> u32 {
        match kind {
            InputKind::Query => self.markers.query,
            InputKind::Document => self.markers.document,
        }
    }

    pub fn skiplist_ids_of(&self) -> &std::collections::HashSet<u32> {
        &self.skiplist_ids
    }
}

impl ArchModel for EmbedBundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "qwen35-embed"
    }

    /// No runtime KV cache: the encoder is one-shot per request.
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let bundle = *self;
        bundle.weights.free_gpu(gpu);
        bundle.state.free_gpu(gpu);
    }
}
