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
//! MoVA value-expert routing uses GPU-side `deepseek4_moe_topk_bias_aware_f32`
//! for bias-aware top-K, then downloads 4 expert indices (16 bytes) for
//! per-expert `weight_gemv`. MoE FFN routing uses the same GPU top-K, then
//! GPU-indexed MoE GEMV kernels (`gemv_hfq4g256_moe_*_k8_indexed_batched`).
//! GPU-side sampling (`argmax_f32` / `sample_top_p`) eliminates the ~1 MB
//! logits download per token. PM4 retained-replay is not yet enabled — the
//! MoVA D2H index download breaks capture.
//! KV write + attention uses the dispatch layer (`KvTierPlan` + `AttnParams`),
//! matching the cohere2moe pattern.

use crate::config::K2HorizonConfig;
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_runtime::llama::{
    rotate_x_mq_for, weight_gemv, weight_gemv_residual, KvCache, KvCacheExt,
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

    // MoVA routing scratch
    pub v_router_logits: GpuTensor, // [mova_num_experts] = [64]
    pub v_topk_indices: GpuTensor,  // [mova_k_top] = [4] i32-in-F32
    pub v_topk_weights: GpuTensor,  // [mova_k_top] = [4]
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
    pub scratch_act: GpuTensor,    // [moe_inter] — silu_mul output for per-expert down
    pub scratch_down: GpuTensor,   // [hidden] — per-expert down GEMV output
    pub scratch_v: GpuTensor,      // [kv_dim] — per-expert MoVA v_expert output

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
    pub flash_partials: GpuTensor, // flash attention scratch (pre-allocated)
    pub sample_buf: GpuTensor,   // [2] F32 — (token_id, rng_state) from GPU sampler
    pub repeat_buf: GpuTensor,   // [64] F32 — repeat penalty window (unused, required by kernel)
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
            scratch_act, scratch_down, scratch_v,
            final_norm_buf, logits,
            sample_buf, repeat_buf,
            flash_partials,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h, normed, fa_q, fa_k, fa_v, fa_attn_out,
            v_router_logits, v_topk_indices, v_topk_weights,
            attn_gate_out,
            scratch_act, scratch_down, scratch_v,
            ffn_x_rot, gate_batch, up_batch, rot_batch, down_expanded,
            final_norm_buf, logits, flash_partials,
            sample_buf, repeat_buf,
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
            scratch_act: alloc(gpu, moe_inter, "scratch_act")?,
            scratch_down: alloc(gpu, hidden, "scratch_down")?,
            scratch_v: alloc(gpu, kv_dim, "scratch_v")?,
            shared_down: alloc(gpu, hidden, "shared_down")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            sample_buf: alloc(gpu, 2, "sample_buf")?,
            repeat_buf: alloc(gpu, 64, "repeat_buf")?,
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
/// Run the full forward (embedding → 48 layers → final norm → lm_head).
/// Logits land in `state.logits` on GPU. No download.
///
/// PM4 retained-replay is not yet enabled for K2-Horizon: the MoVA
/// per-expert GEMV path downloads 4 expert indices (16 bytes) per layer
/// to select weight tensors on the host, which breaks PM4 capture. The
/// GPU routing (deepseek4_moe_topk_bias_aware_f32) and GPU sampling
/// (argmax_f32 / sample_top_p) are active — eliminating the CPU sort
/// and 1MB logits download. PM4 can be enabled once a broadcast-capable
/// indexed v_expert kernel eliminates the MoVA D2H sync.
fn forward_only(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
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
        return Err("k2_horizon: tied embeddings not yet supported".into());
    }
    Ok(())
}
/// Use [`decode_step_sampled`] instead when the caller can sample on GPU
/// (avoids the ~1 MB logits download per token).
pub fn decode_step(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    forward_only(cfg, weights, state, gpu, token_id, position)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("k2_horizon: download logits: {e:?}"))
}

/// Decode step with GPU-side sampling. Runs the forward, then samples
/// directly from the on-GPU logits — no ~1 MB D2H download. Returns
/// `(token_id, new_rng_state)`.
///
/// `temp <= 1e-6` → greedy argmax (4-byte D2H). `temp > 0` → top-p
/// nucleus sampling (8-byte D2H: token + new RNG).
pub fn decode_step_sampled(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    temp: f32,
    top_p: f32,
    rng_state: u32,
) -> Result<(u32, u32), String> {
    forward_only(cfg, weights, state, gpu, token_id, position)?;
    if temp <= 1e-6 {
        let tok = gpu
            .argmax_f32(&state.logits, cfg.vocab_size)
            .map_err(|e| format!("k2_horizon: argmax: {e:?}"))?;
        Ok((tok, rng_state))
    } else {
        let (tok, new_rng) = gpu
            .sample_top_p(
                &state.logits,
                &state.sample_buf,
                &state.repeat_buf,
                cfg.vocab_size,
                temp,
                top_p,
                rng_state,
                0,
                0.0,
            )
            .map_err(|e| format!("k2_horizon: sample_top_p: {e:?}"))?;
        Ok((tok, new_rng))
    }
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
        flash_mode: 0, // let KvTierPlan decide (non-flash for short ctx, flash for long)
        capture_mode: false,
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

    // softplus post-attention gate: attn_out *= softplus(gate_proj(normed))
    //   F.softplus(x, beta=ln(2)) = log(1 + exp(ln2 * x)) / ln2
    //   GPU softplus_f32 computes log(1 + exp(x)) — pre-scale by ln2, post-divide by ln2.
    weight_gemv(gpu, &layer.attn_gate, &state.normed, &state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    gpu.scale_f32(&state.attn_gate_out, SOFTPLUS_BETA)
        .map_err(|e| format!("k2_horizon L{l}: gate pre-scale: {e:?}"))?;
    gpu.softplus_f32(&state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;
    gpu.scale_f32(&state.attn_gate_out, 1.0 / SOFTPLUS_BETA)
        .map_err(|e| format!("k2_horizon L{l}: gate post-scale: {e:?}"))?;
    gpu.mul_f32(&state.fa_attn_out, &state.attn_gate_out, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: gate mul: {e:?}"))?;

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
    //   F.softplus(x, beta=ln(2)) = log(1 + exp(ln2 * x)) / ln2
    //   GPU softplus_f32 computes log(1 + exp(x)) — pre-scale by ln2, post-divide by ln2.
    weight_gemv(gpu, &attn.attn_gate, &state.normed, &state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    gpu.scale_f32(&state.attn_gate_out, SOFTPLUS_BETA)
        .map_err(|e| format!("k2_horizon L{l}: gate pre-scale: {e:?}"))?;
    gpu.softplus_f32(&state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;
    gpu.scale_f32(&state.attn_gate_out, 1.0 / SOFTPLUS_BETA)
        .map_err(|e| format!("k2_horizon L{l}: gate post-scale: {e:?}"))?;
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

// ─── MoVA value-expert routing (GPU topk + per-expert GEMV) ────────────

/// MoVA attention value routing:
/// 1. router GEMV → v_router_logits [64]
/// 2. sigmoid(router_logits) in-place
/// 3. deepseek4_moe_topk_bias_aware_f32 → v_topk_indices + v_topk_weights (GPU)
/// 4. Download 4 indices (16 bytes) to select experts on host
/// 5. Per-expert: weight_gemv + silu_f32 + scale_f32 + add_inplace_f32
///
/// The per-expert loop (4 experts) uses weight_gemv which handles FWHT
/// rotation internally. The 16-byte index download is the only D2H sync
/// in this path — much cheaper than the previous 2× 256-byte downloads +
/// CPU sort.
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

    // 1. router GEMV → v_router_logits [64]
    weight_gemv(gpu, &attn.v_router, &state.normed, &state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: v_router: {e}"))?;

    // 2. sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: v_router sigmoid: {e:?}"))?;

    // 3. GPU top-K with bias-aware semantics. Single kernel launch, no D2H
    //    for the routing decision itself.
    let bias = attn
        .v_router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon L{l}: v_router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.v_router_logits,
        bias,
        &state.v_topk_indices,
        &state.v_topk_weights,
        cfg.mova_num_experts as i32,
        mova_top_k as i32,
        scaling,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_router topk: {e:?}"))?;

    // 4. Download the 4 selected expert indices (16 bytes) to select the
    //    correct weight tensors on the host. This is the only D2H sync.
    let idx_floats = gpu
        .download_f32(&state.v_topk_indices)
        .map_err(|e| format!("k2_horizon L{l}: download v_topk_indices: {e:?}"))?;
    let expert_indices: Vec<usize> = idx_floats
        .iter()
        .map(|f| {
            let bits = f.to_bits() as i32;
            bits as usize
        })
        .collect();

    // 5. Download the routing weights (16 bytes) for the per-expert scale.
    let routing_weights = gpu
        .download_f32(&state.v_topk_weights)
        .map_err(|e| format!("k2_horizon L{l}: download v_topk_weights: {e:?}"))?;

    // 6. Zero fa_v, then per-expert GEMV + silu + scale + accumulate.
    gpu.zero_f32(&state.fa_v)
        .map_err(|e| format!("k2_horizon L{l}: zero fa_v: {e:?}"))?;

    for (k, &expert_idx) in expert_indices.iter().enumerate() {
        let v_expert = &attn.v_experts[expert_idx];
        weight_gemv(gpu, v_expert, &state.normed, &state.scratch_v)
            .map_err(|e| format!("k2_horizon L{l}: v_expert[{k}]: {e}"))?;
        gpu.silu_f32(&state.scratch_v, &state.scratch_v)
            .map_err(|e| format!("k2_horizon L{l}: v_expert[{k}] silu: {e:?}"))?;
        gpu.scale_f32(&state.scratch_v, routing_weights[k])
            .map_err(|e| format!("k2_horizon L{l}: v_expert[{k}] scale: {e:?}"))?;
        gpu.add_inplace_f32(&state.fa_v, &state.scratch_v)
            .map_err(|e| format!("k2_horizon L{l}: v_expert[{k}] add: {e:?}"))?;
    }

    Ok(())
}

// ─── Sigmoid-routed MoE FFN ─────────────────────────────────────────────

/// Sigmoid-routed MoE FFN with fully GPU-side routing:
/// 1. router GEMV → moe_router_logits [num_experts]
/// 2. sigmoid(router_logits) in-place
/// 3. deepseek4_moe_topk_bias_aware_f32 → topk_indices + topk_weights (GPU)
/// 4. rotate_x_mq_for(normed) → ffn_x_rot
/// 5. gemv_hfq4g256_moe_gate_up_k8_indexed_batched → gate_batch + up_batch
/// 6. fused_silu_mul_rotate_mq_batched → rot_batch (silu(gate)*up + FWHT)
/// 7. gemv_hfq4g256_moe_down_k8_indexed_batched_expanded → down_expanded
/// 8. moe_down_combine_k8_batched → h += Σ weight[k] * down_expanded[k]
/// 9. shared expert (SwiGLU GEMV) → add to h
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

    // 1. router GEMV → moe_router_logits [100]
    weight_gemv(gpu, &ffn.router, &state.normed, &state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: moe router: {e}"))?;

    // 2. sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: moe sigmoid: {e:?}"))?;

    // 3. GPU top-K with bias-aware semantics. Single kernel launch, no D2H.
    let bias = ffn
        .router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon L{l}: router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.moe_router_logits,
        bias,
        &state.moe_topk_indices,
        &state.moe_topk_weights,
        cfg.num_experts as i32,
        top_k as i32,
        scaling,
    )
    .map_err(|e| format!("k2_horizon L{l}: moe topk: {e:?}"))?;

    // 4. FWHT-rotate normed for the indexed gate_up GEMV.
    rotate_x_mq_for(gpu, &ffn.experts[0].gate_up, &state.normed, &state.ffn_x_rot, hidden)
        .map_err(|e| format!("k2_horizon L{l}: ffn rotate: {e:?}"))?;

    // 5. Indexed MoE gate_up GEMV: all top_k experts in one kernel launch.
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

    // 6. Fused silu_mul + FWHT rotate: rot_batch = FWHT(silu(gate) * up)
    //    Processes all top_k expert streams in one launch.
    gpu.fused_silu_mul_rotate_mq_batched(
        &state.gate_batch,
        &state.up_batch,
        &state.rot_batch,
        moe_inter,
        top_k,
    )
    .map_err(|e| format!("k2_horizon L{l}: fused silu_mul rotate: {e:?}"))?;

    // 7. Indexed down GEMV: all top_k experts in one kernel launch.
    //    Writes [top_k × hidden] to down_expanded.
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

    // 8. Combine: h += Σ weight[k] * down_expanded[k]
    gpu.moe_down_combine_k8_batched(
        &state.down_expanded,
        &state.moe_topk_weights,
        &state.h,
        hidden,
        top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: moe combine: {e:?}"))?;

    // 9. Shared expert (always-on SwiGLU).
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
