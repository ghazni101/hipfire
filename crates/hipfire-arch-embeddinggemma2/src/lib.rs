// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-embeddinggemma2: Google's **EmbeddingGemma 2** single-vector
//! embedding architecture (`arch_id` 18, `google/embeddinggemma-2`).
//!
//! What the model is: a 24-layer Gemma-4-family **bidirectional encoder**
//! (dim 512, gated gelu-tanh FFN 2048, sandwich RMSNorm, per-layer embeddings)
//! with a 768-d `embedding_projection` head. Every layer runs non-causal
//! attention — the "sliding" layers see a symmetric ±512 window, the 5:1
//! "full" layers see the whole sequence. Sliding layers carry 4 Q heads of
//! 256 with 2 KV heads; full layers widen to head_dim 512 with 1 KV head
//! (MQA), per the checkpoint's `per_layer_config`. RoPE is half-split over
//! the whole head dim with θ = 10 000 (sliding) / 1 000 000 (full), Q/K get
//! learned per-head RMSNorm, V a scale-less one, and attention runs with
//! scaling 1.0 (the QK norms replace the usual 1/√d). The per-layer
//! embedding (PLE) signal is *projection-only*: a single
//! `[layers·512, 512]` linear on the (embed-scaled) token embeddings,
//! reshaped per layer, RMSNorm'd, and mixed into every layer's residual
//! stream after the MLP by a gated bottleneck block. After the final norm
//! the trunk projects every token to 768 dims; sentence-transformers then
//! mean-pools the tokens and L2-normalizes (Matryoshka truncation to
//! 512/256/128 with re-normalization is supported per request).
//!
//! What this crate owns:
//! - [`config`]: parse `text_config` (model_type `embedding_gemma2_text`) +
//!   the sentence-transformers prompt sidecars.
//! - [`names`]/[`weights`]: stream `language_model.*` tensors (F32/BF16/F16
//!   dir or `.hfq`, compressed tiers decoded through the runtime registry)
//!   to the GPU as f16 GEMM weights + f32 norms, embedding table host f32.
//! - [`forward`]: the GPU encoder — embed gather × √512, PLE projection,
//!   per-layer bidirectional attention (eager WMMA-GEMM decomposition with
//!   an additive band mask on sliding layers when the sequence outgrows the
//!   window; Q-tiled kernel below the eager threshold), MLP, PLE block,
//!   `layer_scalar`, final norm and the 768-d projection.
//! - [`post`]: mean pooling, Matryoshka truncation, L2 normalize — host
//!   math, unit-testable without a GPU.
//! - [`bundle`]/[`carrier`]: the `ArchModel` bundle and the loader entry
//!   the carrier calls; [`slot_engine`] batches concurrent requests into
//!   one forward.
//!
//! Multimodal scope: the checkpoint also ships a Gemma-4 vision tower, an
//! audio tower and their projectors (`vision_tower.*`, `audio_tower.*`,
//! `embed_vision.*`, `embed_audio.*`). This implementation serves the text
//! embedding path only: the towers are neither loaded nor quantized, and an
//! image/video/audio input is refused with an explicit error at assembly —
//! never silently embedded as placeholder text.
//!
//! Numerics: activations f32, GEMM weights f16 (WMMA f16×f16 f32-accumulate)
//! — the same tradeoff the qwen35-embed encoder makes, ~1e-3 relative vs an
//! f32 reference. The checkpoint ships BF16; the model card's "do not use
//! float16" warning concerns serving the whole network in f16, not the
//! weight-operand precision of an f32-accumulate GEMM.

pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod names;
pub mod post;
pub mod slot_engine;
pub mod weights;

pub use bundle::EmbedGemmaBundle;
pub use carrier::{assemble_input, load_embed_gemma_bundle, load_embed_gemma_from_source, preflight_slot, AssembledInput};
pub use config::{EmbedConfig, InputKind};
pub use forward::{encode_embeddings, encode_embeddings_batch, EncodeInput};
pub use post::{l2_normalize_rows, mean_pool_rows, mrl_truncate};

/// `arch_id` this crate claims — see `docs/architecture-ids.md`.
pub const ARCH_ID: u32 = 18;
