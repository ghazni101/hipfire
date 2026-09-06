// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon weight structs.
//!
//! Three layer weight shapes:
//! - [`DenseLayerWeights`] — layers 0–2: standard MHA (q/k/v/o) + dense SwiGLU MLP.
//! - [`MovaLayerWeights`] — layers 3–47: MoVA attention (q/k/o + v_experts +
//!   v_router + post-attn gate) + sigmoid-routed MoE FFN.
//!
//! K2-Horizon ships experts as **separate 2D tensors** (not 3D stacked like
//! qwen3.5), matching the DeepSeek V4 / MiniMax pattern. The quantizer
//! splits per-expert tensors as `mlp.experts.{E}.gate_proj.weight` etc.

use hipfire_runtime::llama::WeightTensor;
use rdna_compute::GpuTensor;

// ─── Dense layer (layers 0–2) ───────────────────────────────────────────

/// Weights for a dense attention + dense MLP layer.
///
/// Tensor names (safetensors):
/// - `model.layers.{L}.input_layernorm.weight` → `attn_norm`
/// - `model.layers.{L}.self_attn.q_proj.weight` → `wq`
/// - `model.layers.{L}.self_attn.k_proj.weight` → `wk`
/// - `model.layers.{L}.self_attn.v_proj.weight` → `wv`
/// - `model.layers.{L}.self_attn.o_proj.weight` → `wo`
/// - `model.layers.{L}.post_attention_layernorm.weight` → `ffn_norm`
/// - `model.layers.{L}.mlp.gate_proj.weight` → `w_gate`
/// - `model.layers.{L}.mlp.up_proj.weight` → `w_up`
/// - `model.layers.{L}.mlp.down_proj.weight` → `w_down`
pub struct DenseLayerWeights {
    pub attn_norm: GpuTensor, // [dim] — RMSNorm weight
    pub wq: WeightTensor,     // [n_heads * head_dim, dim] = [4096, 2560]
    pub wk: WeightTensor,     // [n_kv_heads * head_dim, dim] = [1024, 2560]
    pub wv: WeightTensor,     // [n_kv_heads * head_dim, dim] = [1024, 2560]
    pub wo: WeightTensor,     // [dim, n_heads * head_dim] = [2560, 4096]
    pub ffn_norm: GpuTensor,  // [dim]
    pub w_gate: WeightTensor, // [intermediate_size, dim] = [6144, 2560]
    pub w_up: WeightTensor,   // [intermediate_size, dim] = [6144, 2560]
    pub w_down: WeightTensor, // [dim, intermediate_size] = [2560, 6144]
}

// ─── MoVA attention (layers 3–47) ───────────────────────────────────────

/// Weights for MoVA (Mixture of Value Attention) — replaces the single
/// `v_proj` with 64 value experts + a router + a post-attention gate.
///
/// Tensor names (safetensors):
/// - `model.layers.{L}.self_attn.q_proj.weight` → `wq`
/// - `model.layers.{L}.self_attn.k_proj.weight` → `wk`
/// - `model.layers.{L}.self_attn.v_router.weight` → `v_router`
/// - `model.layers.{L}.self_attn.v_experts.{E}.weight` → `v_experts[E]`
/// - `model.layers.{L}.self_attn.o_proj.weight` → `wo`
/// - `model.layers.{L}.self_attn.gate_proj.weight` → `attn_gate`
///
/// `v_router` is [mova_num_experts, dim] = [64, 2560].
/// `v_router_bias` is [64] — added to sigmoid scores for selection only.
/// Each `v_experts[E]` is [kv_dim, dim] = [1024, 2560] — produces the
/// per-expert value projection (kv_dim = n_kv_heads * head_dim).
/// `attn_gate` is [dim, dim] = [2560, 2560] — produces the softplus-gated
/// post-attention scalar.
pub struct MovaAttnWeights {
    pub wq: WeightTensor,          // [4096, 2560]
    pub wk: WeightTensor,          // [1024, 2560]
    pub v_router: WeightTensor,    // [64, 2560] — routes to value experts
    pub v_router_bias: Option<GpuTensor>, // [64] — present when moe_gate_bias=true
    pub v_experts: Vec<WeightTensor>, // 64 × [1024, 2560] (kv_dim, not q_dim)
    pub v_expert_ptrs: GpuTensor,  // [2*64] F32 = 64 u64 device ptrs
    pub wo: WeightTensor,          // [2560, 4096]
    pub attn_gate: WeightTensor,   // [2560, 2560] — softplus post-attn gate
}

// ─── Sigmoid-routed MoE FFN (layers 3–47) ───────────────────────────────

/// Per-expert FFN weights for the sigmoid-routed MoE.
///
/// The loader byte-fuses `gate_proj‖up_proj` into a single `gate_up` blob
/// (matching cohere2moe/qwen35), which the indexed MoE GEMV kernels expect.
///
/// Tensor names (safetensors):
/// - `model.layers.{L}.mlp.experts.{E}.gate_proj.weight` → fused into `gate_up`
/// - `model.layers.{L}.mlp.experts.{E}.up_proj.weight`   → fused into `gate_up`
/// - `model.layers.{L}.mlp.experts.{E}.down_proj.weight` → `down`
///
pub struct MoeExpertWeights {
    pub gate_up: WeightTensor, // [2*moe_intermediate_size, dim] = [1536, 2560] — fused gate‖up
    pub down: WeightTensor,    // [dim, moe_intermediate_size] = [2560, 768]
}

/// Shared expert (always-on, 1 per MoE layer).
///
/// Tensor names:
/// - `model.layers.{L}.mlp.shared_experts.gate_proj.weight`
/// - `model.layers.{L}.mlp.shared_experts.up_proj.weight`
/// - `model.layers.{L}.mlp.shared_experts.down_proj.weight`
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [768, 2560]
    pub up: WeightTensor,   // [768, 2560]
    pub down: WeightTensor, // [2560, 768]
}

/// MoE FFN weights for one MoE layer.
///
/// Tensor names:
/// - `model.layers.{L}.mlp.router.weight` → `router` [100, 2560]
/// - `model.layers.{L}.mlp.router.bias` → `router_bias` [100] (if moe_gate_bias)
/// - `model.layers.{L}.mlp.experts.{E}.*` → `experts[E]`
/// - `model.layers.{L}.mlp.shared_experts.*` → `shared`
pub struct MoeFfnWeights {
    pub router: WeightTensor,      // [100, 2560]
    pub router_bias: Option<GpuTensor>, // [100] — present when moe_gate_bias=true
    pub experts: Vec<MoeExpertWeights>, // 100 experts (fused gate_up + down)
    pub expert_gate_up_ptrs: GpuTensor, // [2*100] F32 = 100 u64 device ptrs
    pub expert_down_ptrs: GpuTensor,    // [2*100] F32 = 100 u64 device ptrs
    pub shared: SharedExpertWeights,    // 1 shared expert
}

// ─── Full MoE layer (MoVA attention + MoE FFN) ──────────────────────────

/// Weights for a MoE layer (MoVA attention + sigmoid-routed MoE FFN).
pub struct MovaLayerWeights {
    pub attn_norm: GpuTensor, // [dim]
    pub attn: MovaAttnWeights,
    pub ffn_norm: GpuTensor, // [dim]
    pub ffn: MoeFfnWeights,
}

// ─── Top-level weight container ─────────────────────────────────────────

/// All weights for a K2-Horizon model, split by layer kind.
///
/// `dense_layers[0..3]` are the dense prefix; `moe_layers[0..45]` are the
/// MoVA+MoE body. The embedding and final norm are shared.
pub struct K2HorizonWeights {
    /// Token embedding [vocab_size, dim] = [250624, 2560]
    pub token_embd: GpuTensor,
    /// Dense layers (layers 0–2)
    pub dense_layers: Vec<DenseLayerWeights>,
    /// MoE layers (layers 3–47)
    pub moe_layers: Vec<MovaLayerWeights>,
    /// Final RMSNorm [dim]
    pub final_norm: GpuTensor,
    /// Output/lm_head [vocab_size, dim] = [250624, 2560]
    /// (present when tie_word_embeddings=false)
    pub lm_head: Option<WeightTensor>,
}
