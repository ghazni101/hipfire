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
    pub attn_norm: GpuTensor,    // [dim] — RMSNorm weight
    pub wq: WeightTensor,        // [n_heads * head_dim, dim] = [4096, 2560]
    pub wk: WeightTensor,        // [n_kv_heads * head_dim, dim] = [1024, 2560]
    pub wv: WeightTensor,        // [n_kv_heads * head_dim, dim] = [1024, 2560]
    pub wo: WeightTensor,        // [dim, n_heads * head_dim] = [2560, 4096]
    pub attn_gate: WeightTensor, // [n_heads * head_dim, dim] = [4096, 2560]
    /// Fused [wq‖wk‖wv‖attn_gate] = [10240, 2560] single-GEMV weight +
    /// owning blob. When Some, wq/wk/wv/attn_gate are views into the owner.
    pub attn_fused: Option<(WeightTensor, GpuTensor)>,
    pub ffn_norm: GpuTensor,  // [dim]
    pub w_gate: WeightTensor, // [intermediate_size, dim] = [6144, 2560]
    pub w_up: WeightTensor,   // [intermediate_size, dim] = [6144, 2560]
    /// Fused [w_gate‖w_up] = [12288, 2560] single-GEMV weight + owner.
    /// When Some, w_gate/w_up are views into the owner.
    pub ffn_fused: Option<(WeightTensor, GpuTensor)>,
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
/// `attn_gate` is [n_heads * head_dim, dim] = [4096, 2560] — produces the
/// per-element softplus gate applied to the attention output.
pub struct MovaAttnWeights {
    pub wq: WeightTensor,                 // [4096, 2560]
    pub wk: WeightTensor,                 // [1024, 2560]
    pub v_router: WeightTensor,           // [64, 2560] — routes to value experts
    pub v_router_bias: Option<GpuTensor>, // [64] — loaded iff the HFQ carries the tensor
    pub v_experts: Vec<WeightTensor>,     // 64 × [1024, 2560] (kv_dim, not q_dim)
    /// Owning blob when v_experts were packed into one allocation. The
    /// `v_experts` WeightTensors are non-owning views into it — keep this
    /// alive for the model's lifetime and free it (not the views) on unload.
    pub v_experts_owner: Option<GpuTensor>,
    pub v_expert_ptrs: GpuTensor, // [2*64] F32 = 64 u64 device ptrs
    pub wo: WeightTensor,         // [2560, 4096]
    pub attn_gate: WeightTensor,  // [4096, 2560] — softplus post-attn gate
    /// Fused [wq‖wk‖v_router‖attn_gate] = [9280, 2560] single-GEMV weight +
    /// owning blob. When Some, wq/wk/v_router/attn_gate are views into it.
    pub attn_fused: Option<(WeightTensor, GpuTensor)>,
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
/// - `model.layers.{L}.mlp.gate.weight` → `router` [100, 2560]
/// - `model.layers.{L}.mlp.gate.bias` → `router_bias` [100] (loaded iff present)
/// - `model.layers.{L}.mlp.experts.{E}.*` → `experts[E]`
/// - `model.layers.{L}.mlp.shared_experts.*` → `shared`
pub struct MoeFfnWeights {
    pub router: WeightTensor,           // [100, 2560]
    pub router_bias: Option<GpuTensor>, // [100] — loaded iff the HFQ carries the tensor
    pub experts: Vec<MoeExpertWeights>, // 100 experts (fused gate_up + down)
    /// Owning blobs when experts were packed (gate_up blob + down blob).
    /// The per-expert WeightTensors are non-owning views — keep alive for
    /// the model's lifetime; free the owners (not the views) on unload.
    pub experts_gate_up_owner: Option<GpuTensor>,
    pub experts_down_owner: Option<GpuTensor>,
    pub expert_gate_up_ptrs: GpuTensor, // [2*100] F32 = 100 u64 device ptrs
    pub expert_down_ptrs: GpuTensor,    // [2*100] F32 = 100 u64 device ptrs
    pub shared: SharedExpertWeights,    // 1 shared expert
    /// Fused [router‖shared.gate‖shared.up] = [1636, 2560] single-GEMV
    /// weight + owning blob. When Some, router/shared.gate/shared.up are
    /// views into it (shared.down stays separate — different K).
    pub ffn_fused: Option<(WeightTensor, GpuTensor)>,
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

// ─── Freeing ────────────────────────────────────────────────────────────

impl K2HorizonWeights {
    /// Free every GPU allocation owned by this weight set.
    ///
    /// Packed-expert WeightTensors are non-owning views (`DeviceBuffer`
    /// `Borrowed`) into the `*_owner` blobs — `free_all` on them would be
    /// refused by `Gpu::free_tensor` anyway, so we free the owners once and
    /// drop the views. Non-packed experts own their buffers and are freed
    /// per-expert via `free_all`.
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.final_norm);
        if let Some(lm) = self.lm_head {
            lm.free_all(gpu);
        }

        for layer in self.dense_layers {
            let _ = gpu.free_tensor(layer.attn_norm);
            let _ = gpu.free_tensor(layer.ffn_norm);
            if let Some((_fused, owner)) = layer.attn_fused {
                // wq/wk/wv/attn_gate are views into owner — drop, free once.
                drop(layer.wq);
                drop(layer.wk);
                drop(layer.wv);
                drop(layer.attn_gate);
                let _ = gpu.free_tensor(owner);
            } else {
                layer.wq.free_all(gpu);
                layer.wk.free_all(gpu);
                layer.wv.free_all(gpu);
                layer.attn_gate.free_all(gpu);
            }
            layer.wo.free_all(gpu);
            if let Some((_fused, owner)) = layer.ffn_fused {
                drop(layer.w_gate);
                drop(layer.w_up);
                let _ = gpu.free_tensor(owner);
            } else {
                layer.w_gate.free_all(gpu);
                layer.w_up.free_all(gpu);
            }
            layer.w_down.free_all(gpu);
        }

        for layer in self.moe_layers {
            let _ = gpu.free_tensor(layer.attn_norm);
            let _ = gpu.free_tensor(layer.ffn_norm);

            let attn = layer.attn;
            if let Some((_fused, owner)) = attn.attn_fused {
                drop(attn.wq);
                drop(attn.wk);
                drop(attn.v_router);
                drop(attn.attn_gate);
                let _ = gpu.free_tensor(owner);
            } else {
                attn.wq.free_all(gpu);
                attn.wk.free_all(gpu);
                attn.v_router.free_all(gpu);
                attn.attn_gate.free_all(gpu);
            }
            attn.wo.free_all(gpu);
            if let Some(b) = attn.v_router_bias {
                let _ = gpu.free_tensor(b);
            }
            let _ = gpu.free_tensor(attn.v_expert_ptrs);
            if let Some(owner) = attn.v_experts_owner {
                // Views into the owner — drop them, free the blob once.
                drop(attn.v_experts);
                let _ = gpu.free_tensor(owner);
            } else {
                for e in attn.v_experts {
                    e.free_all(gpu);
                }
            }

            let ffn = layer.ffn;
            if let Some((_fused, owner)) = ffn.ffn_fused {
                // router/shared.gate/shared.up are views into owner.
                drop(ffn.router);
                drop(ffn.shared.gate);
                drop(ffn.shared.up);
                let _ = gpu.free_tensor(owner);
            } else {
                ffn.router.free_all(gpu);
                ffn.shared.gate.free_all(gpu);
                ffn.shared.up.free_all(gpu);
            }
            if let Some(b) = ffn.router_bias {
                let _ = gpu.free_tensor(b);
            }
            let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
            let _ = gpu.free_tensor(ffn.expert_down_ptrs);
            if ffn.experts_gate_up_owner.is_some() || ffn.experts_down_owner.is_some() {
                drop(ffn.experts);
                if let Some(o) = ffn.experts_gate_up_owner {
                    let _ = gpu.free_tensor(o);
                }
                if let Some(o) = ffn.experts_down_owner {
                    let _ = gpu.free_tensor(o);
                }
            } else {
                for e in ffn.experts {
                    e.gate_up.free_all(gpu);
                    e.down.free_all(gpu);
                }
            }
            ffn.shared.down.free_all(gpu);
        }
    }
}
