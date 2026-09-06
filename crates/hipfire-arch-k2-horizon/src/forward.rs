// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon forward pass — single-token decode (`decode_step`).
//!
//! Three layer forward paths:
//! - **Dense layers (0–2)**: standard MHA (q/k/v/o) + dense SwiGLU MLP.
//! - **MoE layers (3–47)**: MoVA attention (q/k + v_experts routing +
//!   softplus post-attn gate) + sigmoid-routed MoE FFN.
//!
//! Both MoVA value-expert routing and MoE FFN routing use GPU-indexed MoE
//! GEMV kernels (`gemv_hfq4g256_moe_*_k8_indexed_batched`), matching the
//! cohere2moe pattern. Router bias is handled by downloading the tiny sigmoid
//! scores (64 or 100 floats), doing topk+normalize+scale on CPU, then
//! uploading indices+weights — the GPU `moe_topk_renorm_k8` kernel cannot
//! handle bias-for-selection-only semantics.
//!
//! KV write + attention uses the dispatch layer (`KvTierPlan` + `AttnParams`),
//! matching the cohere2moe pattern.

use crate::config::K2HorizonConfig;
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv,
    weight_gemv_residual, KvCache, KvCacheExt,
};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── State ──────────────────────────────────────────────────────────────

const DEFAULT_MAX_SEQ: usize = 2048;
const SOFTPLUS_BETA: f32 = std::f32::consts::LN_2;
/// Flash prefill sub-batch size. Smaller = less VRAM. K2-Horizon has 32 heads
/// × 128 head_dim; at max_seq=2048 this is 32×17×130×16 = ~1.1 MB.
const FLASH_PREFILL_SUBBATCH: usize = 16;

/// Per-decode GPU scratch + KV cache for K2-Horizon.
///
/// MoVA and MoE FFN use GPU-indexed MoE GEMV kernels with batched scratch
/// (no per-expert CPU download/upload). The `moe_top_k` constant is the
/// max of `mova_num_experts_per_tok` (4) and `num_experts_per_tok` (8),
/// so both paths can share the same topk buffers.
pub struct K2HorizonState {
    pub kv: KvCache,
    pub pos_buf: DeviceBuffer,
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual stream
    pub h: GpuTensor,      // [hidden]
    pub normed: GpuTensor, // [hidden] — grouped_rmsnorm output

    // attention scratch (shared by dense + MoVA)
    pub fa_q: GpuTensor,        // [n_heads * head_dim] = [4096]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim] = [1024]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim] = [1024] (dense) or [kv_dim] (MoVA)
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim] = [4096]

    // MoVA routing scratch (GPU indexed GEMV)
    pub v_router_logits: GpuTensor, // [mova_num_experts] = [64]
    pub v_topk_indices: GpuTensor,  // [mova_k_top] = [4] i32-in-F32
    pub v_topk_weights: GpuTensor,  // [mova_k_top] = [4]
    pub v_x_rot: GpuTensor,         // [hidden] — FWHT(normed) for MoVA GEMV
    pub v_expanded: GpuTensor,      // [mova_k_top * kv_dim] = [4 * 1024]
    pub attn_gate_out: GpuTensor,   // [n_heads * head_dim] = [4096] — gate_proj output

    // dense FFN scratch
    pub dense_gate: GpuTensor, // [intermediate_size] = [6144]
    pub dense_up: GpuTensor,   // [intermediate_size] = [6144]
    pub dense_act: GpuTensor,  // [intermediate_size] = [6144]

    // MoE FFN routing scratch (GPU indexed GEMV)
    pub moe_router_logits: GpuTensor, // [num_experts] = [100]
    pub moe_topk_indices: GpuTensor,  // [moe_k_top] = [8] i32-in-F32
    pub moe_topk_weights: GpuTensor,  // [moe_k_top] = [8]
    pub ffn_x_rot: GpuTensor,         // [hidden] — FWHT(normed) for MoE GEMV
    pub gate_batch: GpuTensor,        // [moe_k_top * moe_inter] = [8 * 768]
    pub up_batch: GpuTensor,          // [moe_k_top * moe_inter] = [8 * 768]
    pub rot_batch: GpuTensor,         // [moe_k_top * moe_inter] = [8 * 768]
    pub down_expanded: GpuTensor,     // [moe_k_top * hidden] = [8 * 2560]
    pub shared_gate: GpuTensor,       // [moe_intermediate_size] = [768]
    pub shared_up: GpuTensor,         // [moe_intermediate_size] = [768]
    pub shared_act: GpuTensor,        // [moe_intermediate_size] = [768]
    pub shared_down: GpuTensor,       // [hidden] = [2560]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
    pub flash_partials: GpuTensor, // flash attention scratch (pre-allocated)
}

impl K2HorizonState {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let K2HorizonState {
            kv,
            pos_buf,
            max_seq: _,
            n_tokens: _,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            v_router_logits,
            v_topk_indices,
            v_topk_weights,
            v_x_rot,
            v_expanded,
            attn_gate_out,
            dense_gate,
            dense_up,
            dense_act,
            moe_router_logits,
            moe_topk_indices,
            moe_topk_weights,
            ffn_x_rot,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            shared_gate,
            shared_up,
            shared_act,
            shared_down,
            final_norm_buf,
            logits,
            flash_partials,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h, normed, fa_q, fa_k, fa_v, fa_attn_out,
            v_router_logits, v_topk_indices, v_topk_weights, v_x_rot, v_expanded,
            attn_gate_out,
            dense_gate, dense_up, dense_act,
            moe_router_logits, moe_topk_indices, moe_topk_weights,
            ffn_x_rot, gate_batch, up_batch, rot_batch, down_expanded,
            shared_gate, shared_up, shared_act, shared_down,
            final_norm_buf, logits, flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    pub fn new(gpu: &mut Gpu, cfg: &K2HorizonConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(DEFAULT_MAX_SEQ);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &K2HorizonConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let mova_n_exp = cfg.mova_num_experts;
        let mova_k = cfg.mova_num_experts_per_tok;
        let moe_k = cfg.num_experts_per_tok;

        // KV cache: all 48 layers are attention layers.
        let kv = KvCache::new_gpu_q8(
            gpu,
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("k2_horizon: kv cache: {e:?}"))?;

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("k2_horizon: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("k2_horizon: alloc {label}: {e:?}"))
        };

        Ok(K2HorizonState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            normed: alloc(gpu, hidden, "normed")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?, // MoVA v is [kv_dim]
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            v_router_logits: alloc(gpu, mova_n_exp, "v_router_logits")?,
            v_topk_indices: alloc(gpu, mova_k, "v_topk_indices")?,
            v_topk_weights: alloc(gpu, mova_k, "v_topk_weights")?,
            v_x_rot: alloc(gpu, mova_k * hidden, "v_x_rot")?,
            v_expanded: alloc(gpu, mova_k * kv_dim, "v_expanded")?,
            attn_gate_out: alloc(gpu, q_dim, "attn_gate_out")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            moe_router_logits: alloc(gpu, n_exp, "moe_router_logits")?,
            moe_topk_indices: alloc(gpu, moe_k, "moe_topk_indices")?,
            moe_topk_weights: alloc(gpu, moe_k, "moe_topk_weights")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            gate_batch: alloc(gpu, moe_k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, moe_k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, moe_k * moe_inter, "rot_batch")?,
            down_expanded: alloc(gpu, moe_k * hidden, "down_expanded")?,
            shared_gate: alloc(gpu, moe_inter, "shared_gate")?,
            shared_up: alloc(gpu, moe_inter, "shared_up")?,
            shared_act: alloc(gpu, moe_inter, "shared_act")?,
            shared_down: alloc(gpu, hidden, "shared_down")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            flash_partials: alloc(
                gpu,
                cfg.n_heads
                    * ((max_seq + 127) / 128)
                    * (2 + cfg.head_dim)
                    * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("k2_horizon reset: clear kv: {e:?}"))?;
        Ok(())
    }
}

// ─── Forward ────────────────────────────────────────────────────────────

/// Single-token decode step.
///
/// Looks up the embedding for `token_id`, runs all 48 layers, and returns
/// the full logits vector `[vocab_size]`.
pub fn decode_step(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {

    // Stage position on device.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &position.to_ne_bytes())
        .map_err(|e| format!("k2_horizon: stage pos: {e:?}"))?;

    // Embedding lookup → state.h.
    gpu.embedding_lookup_q8(&weights.token_embd, &state.h, token_id, cfg.dim)
        .map_err(|e| format!("k2_horizon: embed lookup: {e:?}"))?;

    let dense_layers = &weights.dense_layers;
    let moe_layers = &weights.moe_layers;

    for (l, layer) in dense_layers.iter().enumerate() {
        forward_dense_layer(cfg, layer, state, gpu, l, position)?;
    }

    for (l, layer) in moe_layers.iter().enumerate() {
        let global_layer = dense_layers.len() + l;
        forward_moe_layer(cfg, layer, state, gpu, global_layer, position)?;
    }


    // Final norm + lm_head.
    gpu.grouped_rmsnorm_f32(
        &state.h,
        &weights.final_norm,
        &state.final_norm_buf,
        1,
        cfg.dim,
        cfg.layernorm_num_groups,
        cfg.norm_eps,
    )
    .map_err(|e| format!("k2_horizon: final norm: {e:?}"))?;


    if let Some(lm_head) = &weights.lm_head {
        weight_gemv(gpu, lm_head, &state.final_norm_buf, &state.logits)
            .map_err(|e| format!("k2_horizon: lm_head: {e}"))?;
    } else {
        // Tied embeddings — would need embed tensor as WeightTensor.
        return Err("k2_horizon: tied embeddings not yet supported".into());
    }

    // Download logits for CPU-side sampling.
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("k2_horizon: download logits: {e:?}"))
}

// ─── KV write + attention helper ────────────────────────────────────────

/// Write K,V to cache at current position, then run flash attention.
/// Uses the dispatch layer's KvTierPlan + AttnParams, matching cohere2moe.
fn attend(
    cfg: &K2HorizonConfig,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    seq_len: usize,
) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: seq_len - 1,
        q8_windowed: true,
        window: 0,
        ..state.kv.tier_inputs()
    })
    .map_err(|e| format!("k2_horizon L{l}: kv tier: {e}"))?;

    let io = AttnParams {
        q: &state.fa_q,
        k: &state.fa_k,
        v: &state.fa_v,
        k_cache: &state.kv.k_gpu[l],
        v_cache: &state.kv.v_gpu[l],
        k_scales: None,
        v_scales: None,
        pos_buf: &state.pos_buf,
        pos: seq_len - 1,
        positions: None,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&state.flash_partials),
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &state.fa_attn_out,
    };

    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
        .map_err(|e| format!("k2_horizon L{l}: attend: {e:?}"))?;
    Ok(())
}

// ─── Dense layer (layers 0–2) ───────────────────────────────────────────

fn forward_dense_layer(
    cfg: &K2HorizonConfig,
    layer: &DenseLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    position: u32,
) -> Result<(), String> {
    let hidden = cfg.dim;
    let eps = cfg.norm_eps;
    let n_groups = cfg.layernorm_num_groups;

    // normed = grouped_rmsnorm(h, attn_norm)
    gpu.grouped_rmsnorm_f32(
        &state.h,
        &layer.attn_norm,
        &state.normed,
        1,
        hidden,
        n_groups,
        eps,
    )
    .map_err(|e| format!("k2_horizon L{l}: input norm: {e:?}"))?;

    // q = q_proj(normed), k = k_proj(normed), v = v_proj(normed)
    weight_gemv(gpu, &layer.wq, &state.normed, &state.fa_q)
        .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &layer.wk, &state.normed, &state.fa_k)
        .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &layer.wv, &state.normed, &state.fa_v)
        .map_err(|e| format!("k2_horizon L{l}: v_proj: {e}"))?;

    // RoPE on Q and K (full rotary, rope_head_dim == head_dim)
    gpu.rope_f32(
        &state.fa_q,
        &state.fa_k,
        &state.pos_buf,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("k2_horizon L{l}: rope: {e:?}"))?;

    // KV write + attention.
    let seq_len = position as usize + 1;
    attend(cfg, state, gpu, l, seq_len)?;

    // h += o_proj(attn_out)
    weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;

    // FFN: normed = grouped_rmsnorm(h, ffn_norm)
    gpu.grouped_rmsnorm_f32(
        &state.h,
        &layer.ffn_norm,
        &state.normed,
        1,
        hidden,
        n_groups,
        eps,
    )
    .map_err(|e| format!("k2_horizon L{l}: ffn norm: {e:?}"))?;

    // dense SwiGLU: down(silu(gate(normed)) * up(normed))
    weight_gemv(gpu, &layer.w_gate, &state.normed, &state.dense_gate)
        .map_err(|e| format!("k2_horizon L{l}: dense gate: {e}"))?;
    weight_gemv(gpu, &layer.w_up, &state.normed, &state.dense_up)
        .map_err(|e| format!("k2_horizon L{l}: dense up: {e}"))?;
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("k2_horizon L{l}: dense silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &layer.w_down, &state.dense_act, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: dense down: {e}"))?;

    Ok(())
}

// ─── MoE layer (layers 3–47): MoVA attention + sigmoid MoE FFN ──────────

fn forward_moe_layer(
    cfg: &K2HorizonConfig,
    layer: &MovaLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    position: u32,
) -> Result<(), String> {
    let hidden = cfg.dim;
    let eps = cfg.norm_eps;
    let n_groups = cfg.layernorm_num_groups;
    let attn = &layer.attn;
    let ffn = &layer.ffn;

    // ── Attention branch ──────────────────────────────────────────────

    // normed = grouped_rmsnorm(h, attn_norm)
    gpu.grouped_rmsnorm_f32(
        &state.h,
        &layer.attn_norm,
        &state.normed,
        1,
        hidden,
        n_groups,
        eps,
    )
    .map_err(|e| format!("k2_horizon L{l}: input norm: {e:?}"))?;

    // q = q_proj(normed), k = k_proj(normed)
    weight_gemv(gpu, &attn.wq, &state.normed, &state.fa_q)
        .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &attn.wk, &state.normed, &state.fa_k)
        .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;

    // MoVA: v = combine_routed_experts(normed)
    forward_mova_value_routing(cfg, attn, state, gpu, l)?;

    // RoPE on Q and K
    gpu.rope_f32(
        &state.fa_q,
        &state.fa_k,
        &state.pos_buf,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("k2_horizon L{l}: rope: {e:?}"))?;

    // KV write + attention.
    let seq_len = position as usize + 1;
    attend(cfg, state, gpu, l, seq_len)?;

    // softplus post-attention gate: attn_out *= softplus(gate_proj(normed))
    //   K2-Horizon uses softplus(x, beta=ln(2)) = log(1 + 2^x).
    //   The GPU softplus_f32 kernel computes log(1 + exp(x)) with no beta.
    //   Pre-scale x by ln(2) so softplus(ln2 * x) = log(1 + exp(ln2 * x)) = log(1 + 2^x).
    weight_gemv(gpu, &attn.attn_gate, &state.normed, &state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    gpu.scale_f32(&state.attn_gate_out, SOFTPLUS_BETA)
        .map_err(|e| format!("k2_horizon L{l}: gate pre-scale: {e:?}"))?;
    gpu.softplus_f32(&state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;
    gpu.mul_f32(&state.fa_attn_out, &state.attn_gate_out, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: gate mul: {e:?}"))?;

    // h += o_proj(attn_out)
    weight_gemv_residual(gpu, &attn.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;

    // ── FFN branch: sigmoid-routed MoE ────────────────────────────────

    // normed = grouped_rmsnorm(h, ffn_norm)
    gpu.grouped_rmsnorm_f32(
        &state.h,
        &layer.ffn_norm,
        &state.normed,
        1,
        hidden,
        n_groups,
        eps,
    )
    .map_err(|e| format!("k2_horizon L{l}: ffn norm: {e:?}"))?;

    forward_sigmoid_moe_ffn(cfg, ffn, state, gpu, l)?;

    Ok(())
}

// ─── MoVA value-expert routing (GPU indexed GEMV) ───────────────────────

/// MoVA attention value routing using GPU-indexed MoE GEMV kernels:
/// 1. router_logits = v_router(normed)  [mova_num_experts]
/// 2. sigmoid(router_logits)
/// 3. CPU top-k (download 64 floats, add bias for selection, topk, normalize, scale)
/// 4. Upload topk_indices + topk_weights
/// 5. rotate_x_mq_for(normed) → v_x_rot
/// 6. gemv_hfq4g256_moe_down_k8_indexed_batched_expanded (v_experts are single
///    linear [kv_dim, hidden], so this is a "down" GEMV with m=kv_dim, k=hidden)
/// 7. silu_f32 on expanded output
/// 8. zero fa_v, then moe_down_combine_k8_batched (weighted sum into fa_v)
///
/// Output lands in `state.fa_v` ([kv_dim]).
fn forward_mova_value_routing(
    cfg: &K2HorizonConfig,
    attn: &crate::weights::MovaAttnWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    let mova_top_k = cfg.mova_num_experts_per_tok;
    let scaling = cfg.router_scaling_factor;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let hidden = cfg.dim;

    weight_gemv(gpu, &attn.v_router, &state.normed, &state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: v_router: {e}"))?;
    // sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: v_router sigmoid: {e:?}"))?;

    // Download sigmoid scores (tiny: 64 floats = 256 bytes) for CPU top-k.
    // The GPU moe_topk_renorm_k8 kernel cannot handle bias-for-selection-only,
    // so we do topk+normalize+scale on CPU and upload indices+weights.
    let scores = gpu
        .download_f32(&state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: download v_router scores: {e:?}"))?;

    // If v_router bias is present, add it to selection scores only.
    let selection_scores: Vec<f32> = if let Some(bias) = &attn.v_router_bias {
        let bias_vals = gpu
            .download_f32(bias)
            .map_err(|e| format!("k2_horizon L{l}: download v_router bias: {e:?}"))?;
        scores
            .iter()
            .zip(bias_vals.iter())
            .map(|(s, b)| s + b)
            .collect()
    } else {
        scores.clone()
    };

    // Top-k by selection scores.
    let mut indexed: Vec<(usize, f32)> = selection_scores
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk: Vec<(usize, f32)> = indexed.into_iter().take(mova_top_k).collect();

    // Gather original sigmoid scores for selected experts, normalize, scale.
    let routing_weights: Vec<f32> = {
        let raw: Vec<f32> = topk.iter().map(|(idx, _)| scores[*idx]).collect();
        let sum: f32 = raw.iter().sum();
        if sum > 0.0 {
            raw.iter().map(|w| (w / sum) * scaling).collect()
        } else {
            vec![scaling / mova_top_k as f32; mova_top_k]
        }
    };

    // Upload topk indices (as i32) and weights to GPU.
    let idx_bytes: Vec<u8> = topk
        .iter()
        .flat_map(|(idx, _)| (*idx as i32).to_ne_bytes())
        .collect();
    let w_bytes: Vec<u8> = routing_weights
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&state.v_topk_indices.buf, &idx_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload v_topk_indices: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&state.v_topk_weights.buf, &w_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload v_topk_weights: {e:?}"))?;

    // FWHT-rotate normed for MQ4G256 prerotated GEMV.
    // The down indexed kernel expects rot_batch as [k_top × K], but all
    // v_experts share the same input — rotate once into the first slice,
    // then replicate to the remaining k_top-1 slices via dtod copies.
    rotate_x_mq_for(gpu, &attn.v_experts[0], &state.normed, &state.v_x_rot, hidden)
        .map_err(|e| format!("k2_horizon L{l}: v rotate: {e:?}"))?;
    let hidden_bytes = hidden * 4;
    for k in 1..mova_top_k {
        gpu.memcpy_dtod_at_auto(
            &state.v_x_rot.buf,
            k * hidden_bytes,
            &state.v_x_rot.buf,
            0,
            hidden_bytes,
        )
        .map_err(|e| format!("k2_horizon L{l}: v_x_rot replicate: {e:?}"))?;
    }

    gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
        &attn.v_expert_ptrs,
        &state.v_topk_indices,
        &state.v_x_rot,
        &state.v_expanded,
        kv_dim,
        hidden,
        mova_top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_expert indexed gemv: {e:?}"))?;

    // silu on expanded output (v = Σ w_e * silu(v_experts[e](normed)))
    gpu.silu_f32(&state.v_expanded, &state.v_expanded)
        .map_err(|e| format!("k2_horizon L{l}: v_expert silu: {e:?}"))?;

    // Zero fa_v, then combine: fa_v += Σ w_k * v_expanded[k]
    gpu.zero_f32(&state.fa_v)
        .map_err(|e| format!("k2_horizon L{l}: zero fa_v: {e:?}"))?;
    gpu.moe_down_combine_k8_batched(
        &state.v_expanded,
        &state.v_topk_weights,
        &state.fa_v,
        kv_dim,
        mova_top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_combine: {e:?}"))?;

    Ok(())
}

// ─── Sigmoid-routed MoE FFN (GPU indexed GEMV) ──────────────────────────

/// Sigmoid-routed MoE FFN using GPU-indexed MoE GEMV kernels:
/// 1. router_logits = router(normed)  [num_experts]
/// 2. sigmoid(router_logits)
/// 3. CPU top-k (download 100 floats, add bias for selection, topk, normalize, scale)
/// 4. Upload topk_indices + topk_weights
/// 5. rotate_x_mq_for(normed) → ffn_x_rot
/// 6. gemv_hfq4g256_moe_gate_up_k8_indexed_batched → gate_batch + up_batch
/// 7. fused_silu_mul_rotate_mq_batched_for → rot_batch
/// 8. gemv_hfq4g256_moe_down_k8_indexed_batched_expanded → down_expanded
/// 9. moe_down_combine_k8_batched → h += Σ w_k * down_k (in-place residual)
/// 10. shared expert (SwiGLU GEMV) → add to h
fn forward_sigmoid_moe_ffn(
    cfg: &K2HorizonConfig,
    ffn: &crate::weights::MoeFfnWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    let top_k = cfg.num_experts_per_tok;
    let scaling = cfg.router_scaling_factor;
    let hidden = cfg.dim;
    let moe_inter = cfg.moe_intermediate_size;

    // router_logits = router(normed)
    weight_gemv(gpu, &ffn.router, &state.normed, &state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: moe router: {e}"))?;

    // sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: moe sigmoid: {e:?}"))?;

    // Download sigmoid scores (tiny: 100 floats = 400 bytes) for CPU top-k.
    let scores = gpu
        .download_f32(&state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: download moe scores: {e:?}"))?;

    // If router bias is present, add it to selection scores only.
    let selection_scores: Vec<f32> = if let Some(bias) = &ffn.router_bias {
        let bias_vals = gpu
            .download_f32(bias)
            .map_err(|e| format!("k2_horizon L{l}: download router bias: {e:?}"))?;
        scores
            .iter()
            .zip(bias_vals.iter())
            .map(|(s, b)| s + b)
            .collect()
    } else {
        scores.clone()
    };

    // Top-k by selection scores.
    let mut indexed: Vec<(usize, f32)> = selection_scores
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Gather original sigmoid scores for selected experts, normalize, scale.
    let routing_weights: Vec<f32> = {
        let raw: Vec<f32> = topk.iter().map(|(idx, _)| scores[*idx]).collect();
        let sum: f32 = raw.iter().sum();
        if cfg.norm_topk_prob && sum > 0.0 {
            raw.iter().map(|w| (w / sum) * scaling).collect()
        } else {
            raw.iter().map(|w| w * scaling).collect()
        }
    };

    // Upload topk indices (as i32) and weights to GPU.
    let idx_bytes: Vec<u8> = topk
        .iter()
        .flat_map(|(idx, _)| (*idx as i32).to_ne_bytes())
        .collect();
    let w_bytes: Vec<u8> = routing_weights
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&state.moe_topk_indices.buf, &idx_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload moe_topk_indices: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&state.moe_topk_weights.buf, &w_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload moe_topk_weights: {e:?}"))?;

    rotate_x_mq_for(gpu, &ffn.experts[0].gate_up, &state.normed, &state.ffn_x_rot, hidden)
        .map_err(|e| format!("k2_horizon L{l}: ffn rotate: {e:?}"))?;

    // Indexed MoE gate_up GEMV: all top_k experts in one kernel launch.
    // gate_batch and up_batch are [top_k * moe_inter] each.
    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &ffn.expert_gate_up_ptrs,
        &state.moe_topk_indices,
        &state.ffn_x_rot,
        &state.gate_batch,
        &state.up_batch,
        2 * moe_inter,
        hidden,
        top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: gate_up indexed gemv: {e:?}"))?;

    // Fused silu_mul + FWHT rotation: rot_batch = silu(gate) * up, then rotate.
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        &ffn.experts[0].down,
        &state.gate_batch,
        &state.up_batch,
        &state.rot_batch,
        moe_inter,
        top_k,
    )
    .map_err(|e| format!("k2_horizon L{l}: silu_mul_rotate: {e:?}"))?;

    // Indexed MoE down GEMV: all top_k experts in one kernel launch.
    // down_expanded is [top_k * hidden] f32.
    gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
        &ffn.expert_down_ptrs,
        &state.moe_topk_indices,
        &state.rot_batch,
        &state.down_expanded,
        hidden,
        moe_inter,
        top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: down indexed gemv: {e:?}"))?;
    // Combine: h += Σ w_k * down_k (in-place on residual).
    gpu.moe_down_combine_k8_batched(
        &state.down_expanded,
        &state.moe_topk_weights,
        &state.h,
        hidden,
        top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: moe combine: {e:?}"))?;
    // Shared expert (always-on SwiGLU).
    weight_gemv(gpu, &ffn.shared.gate, &state.normed, &state.shared_gate)
        .map_err(|e| format!("k2_horizon L{l}: shared gate: {e}"))?;
    weight_gemv(gpu, &ffn.shared.up, &state.normed, &state.shared_up)
        .map_err(|e| format!("k2_horizon L{l}: shared up: {e}"))?;
    gpu.silu_mul_f32(&state.shared_gate, &state.shared_up, &state.shared_act)
        .map_err(|e| format!("k2_horizon L{l}: shared silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &ffn.shared.down, &state.shared_act, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: shared down: {e}"))?;

    Ok(())
}
