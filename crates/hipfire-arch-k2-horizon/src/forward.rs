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
//! The MoVA value-expert routing and MoE FFN routing use host-side top-k
//! (download router logits → CPU topk → dispatch selected experts via
//! per-expert GEMV). The weighted accumulation is done on CPU for
//! correctness; a GPU fused-accumulate optimization can come later.
//!
//! KV write + attention uses the dispatch layer (`KvTierPlan` + `AttnParams`),
//! matching the cohere2moe pattern.

use crate::config::K2HorizonConfig;
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::llama::{weight_gemv, weight_gemv_residual, KvCache, KvCacheExt, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── State ──────────────────────────────────────────────────────────────

const DEFAULT_MAX_SEQ: usize = 2048;
const SOFTPLUS_BETA: f32 = std::f32::consts::LN_2;
/// Flash prefill sub-batch size. Smaller = less VRAM. K2-Horizon has 32 heads
/// × 128 head_dim; at max_seq=2048 this is 32×17×130×16 = ~1.1 MB.
const FLASH_PREFILL_SUBBATCH: usize = 16;

/// Per-decode GPU scratch + KV cache for K2-Horizon.
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
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim] = [1024] (dense) or [n_heads * head_dim] (MoVA)
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim] = [4096]

    // MoVA routing scratch
    pub v_router_logits: GpuTensor, // [mova_num_experts] = [64]
    pub v_expert_out: GpuTensor,    // [kv_dim] — one expert's output
    pub attn_gate_out: GpuTensor,   // [n_heads * head_dim] = [4096] — gate_proj output

    // dense FFN scratch
    pub dense_gate: GpuTensor, // [intermediate_size] = [6144]
    pub dense_up: GpuTensor,   // [intermediate_size] = [6144]
    pub dense_act: GpuTensor,  // [intermediate_size] = [6144]

    // MoE FFN routing scratch
    pub moe_router_logits: GpuTensor, // [num_experts] = [100]
    pub moe_expert_gate: GpuTensor,   // [moe_intermediate_size] = [768]
    pub moe_expert_up: GpuTensor,     // [moe_intermediate_size] = [768]
    pub moe_expert_act: GpuTensor,    // [moe_intermediate_size] = [768]
    pub moe_expert_down: GpuTensor,   // [hidden] = [2560] — reused as upload target
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
            v_expert_out,
            attn_gate_out,
            dense_gate,
            dense_up,
            dense_act,
            moe_router_logits,
            moe_expert_gate,
            moe_expert_up,
            moe_expert_act,
            moe_expert_down,
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
            v_router_logits, v_expert_out, attn_gate_out,
            dense_gate, dense_up, dense_act,
            moe_router_logits, moe_expert_gate, moe_expert_up, moe_expert_act,
            moe_expert_down, shared_gate, shared_up, shared_act, shared_down,
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
            fa_v: alloc(gpu, q_dim, "fa_v")?, // MoVA v is [n_heads * head_dim]
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            v_router_logits: alloc(gpu, mova_n_exp, "v_router_logits")?,
            v_expert_out: alloc(gpu, kv_dim, "v_expert_out")?,
            attn_gate_out: alloc(gpu, q_dim, "attn_gate_out")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            moe_router_logits: alloc(gpu, n_exp, "moe_router_logits")?,
            moe_expert_gate: alloc(gpu, moe_inter, "moe_expert_gate")?,
            moe_expert_up: alloc(gpu, moe_inter, "moe_expert_up")?,
            moe_expert_act: alloc(gpu, moe_inter, "moe_expert_act")?,
            moe_expert_down: alloc(gpu, hidden, "moe_expert_down")?,
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
    gpu.embedding_lookup(&weights.token_embd, &state.h, token_id, cfg.dim)
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

// ─── MoVA value-expert routing ──────────────────────────────────────────

/// MoVA attention value routing:
/// 1. router_logits = v_router(normed)  [mova_num_experts]
/// 2. sigmoid(router_logits)
/// 3. top-k by sigmoid score
/// 4. normalize top-k weights, scale by router_scaling_factor
/// 5. v = Σ w_e * silu(v_experts[e](normed))
///
/// Output lands in `state.fa_v` ([n_kv_heads * head_dim]).
///
/// Weighted accumulation is done on CPU for correctness: download each
/// expert's silu output, scale by weight, accumulate, then upload.
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

    // router_logits = v_router(normed)
    weight_gemv(
        gpu,
        &attn.v_router,
        &state.normed,
        &state.v_router_logits,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_router: {e}"))?;

    // sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: v_router sigmoid: {e:?}"))?;

    // Download router scores for CPU-side top-k.
    let scores = gpu
        .download_f32(&state.v_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: download v_router scores: {e:?}"))?;

    // Top-k selection by sigmoid score.
    let mut indexed: Vec<(usize, f32)> =
        scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk: Vec<(usize, f32)> = indexed.into_iter().take(mova_top_k).collect();

    // Normalize top-k weights and scale.
    let sum: f32 = topk.iter().map(|(_, w)| w).sum();
    let weights: Vec<f32> = if sum > 0.0 {
        topk.iter().map(|(_, w)| (w / sum) * scaling).collect()
    } else {
        vec![scaling / mova_top_k as f32; mova_top_k]
    };

    // v = Σ w_e * silu(v_experts[e](normed))
    // CPU accumulation: download each expert's silu output, scale, sum.
    let mut combined = vec![0.0f32; kv_dim];

    for (rank, (expert_idx, _)) in topk.iter().enumerate() {
        let expert = &attn.v_experts[*expert_idx];
        let w = weights[rank];

        // v_expert_out = v_experts[e](normed)
        weight_gemv(gpu, expert, &state.normed, &state.v_expert_out)
            .map_err(|e| format!("k2_horizon L{l}: v_expert[{expert_idx}]: {e}"))?;

        // silu(v_expert_out) — in-place (x and out can be the same tensor)
        gpu.silu_f32(&state.v_expert_out, &state.v_expert_out)
            .map_err(|e| format!("k2_horizon L{l}: v_expert silu: {e:?}"))?;

        // Download, scale, accumulate on CPU.
        let expert_out = gpu
            .download_f32(&state.v_expert_out)
            .map_err(|e| format!("k2_horizon L{l}: download v_expert: {e:?}"))?;
        for (acc, &val) in combined.iter_mut().zip(expert_out.iter()) {
            *acc += w * val;
        }
    }

    // Upload combined v → fa_v.
    let combined_bytes = unsafe {
        std::slice::from_raw_parts(combined.as_ptr() as *const u8, combined.len() * 4)
    };
    gpu.hip
        .memcpy_htod(&state.fa_v.buf, combined_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload v_combined: {e:?}"))?;

    Ok(())
}

// ─── Sigmoid-routed MoE FFN ─────────────────────────────────────────────

/// Sigmoid-routed MoE FFN:
/// 1. router_logits = router(normed)  [num_experts]
/// 2. sigmoid(router_logits)
/// 3. top-k by sigmoid score (bias added to selection scores only)
/// 4. normalize top-k weights (norm_topk_prob=true), scale by 2.5
/// 5. routed = Σ w_e * down(silu(gate_e(normed)) * up_e(normed))
/// 6. shared = shared_experts(normed)
/// 7. h += routed + shared
///
/// Weighted accumulation on CPU (download expert down output, scale, sum).
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

    // router_logits = router(normed)
    weight_gemv(
        gpu,
        &ffn.router,
        &state.normed,
        &state.moe_router_logits,
    )
    .map_err(|e| format!("k2_horizon L{l}: moe router: {e}"))?;

    // sigmoid(router_logits) — in-place
    gpu.sigmoid_f32(&state.moe_router_logits)
        .map_err(|e| format!("k2_horizon L{l}: moe sigmoid: {e:?}"))?;

    // Download for CPU-side top-k.
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

    // Routed experts — CPU accumulation.
    let mut routed_combined = vec![0.0f32; hidden];

    for (rank, (expert_idx, _)) in topk.iter().enumerate() {
        let expert = &ffn.experts[*expert_idx];
        let w = routing_weights[rank];

        // gate = gate_proj(normed), up = up_proj(normed)
        weight_gemv(gpu, &expert.gate, &state.normed, &state.moe_expert_gate)
            .map_err(|e| format!("k2_horizon L{l}E{expert_idx}: gate: {e}"))?;
        weight_gemv(gpu, &expert.up, &state.normed, &state.moe_expert_up)
            .map_err(|e| format!("k2_horizon L{l}E{expert_idx}: up: {e}"))?;

        // act = silu(gate) * up
        gpu.silu_mul_f32(&state.moe_expert_gate, &state.moe_expert_up, &state.moe_expert_act)
            .map_err(|e| format!("k2_horizon L{l}E{expert_idx}: silu_mul: {e:?}"))?;

        // down = down_proj(act)
        weight_gemv(gpu, &expert.down, &state.moe_expert_act, &state.moe_expert_down)
            .map_err(|e| format!("k2_horizon L{l}E{expert_idx}: down: {e}"))?;

        // Download, scale, accumulate on CPU.
        let down_out = gpu
            .download_f32(&state.moe_expert_down)
            .map_err(|e| format!("k2_horizon L{l}E{expert_idx}: download down: {e:?}"))?;
        for (acc, &val) in routed_combined.iter_mut().zip(down_out.iter()) {
            *acc += w * val;
        }
    }

    // Shared expert (always-on).
    weight_gemv(gpu, &ffn.shared.gate, &state.normed, &state.shared_gate)
        .map_err(|e| format!("k2_horizon L{l}: shared gate: {e}"))?;
    weight_gemv(gpu, &ffn.shared.up, &state.normed, &state.shared_up)
        .map_err(|e| format!("k2_horizon L{l}: shared up: {e}"))?;
    gpu.silu_mul_f32(&state.shared_gate, &state.shared_up, &state.shared_act)
        .map_err(|e| format!("k2_horizon L{l}: shared silu_mul: {e:?}"))?;
    weight_gemv(gpu, &ffn.shared.down, &state.shared_act, &state.shared_down)
        .map_err(|e| format!("k2_horizon L{l}: shared down: {e}"))?;

    // h += routed_combined + shared_down
    // Upload routed_combined into moe_expert_down (reused as upload target),
    // add shared_down, then add to h.
    let routed_bytes = unsafe {
        std::slice::from_raw_parts(routed_combined.as_ptr() as *const u8, routed_combined.len() * 4)
    };
    gpu.hip
        .memcpy_htod(&state.moe_expert_down.buf, routed_bytes)
        .map_err(|e| format!("k2_horizon L{l}: upload routed: {e:?}"))?;
    gpu.add_inplace_f32(&state.moe_expert_down, &state.shared_down)
        .map_err(|e| format!("k2_horizon L{l}: shared add: {e:?}"))?;
    gpu.add_inplace_f32(&state.h, &state.moe_expert_down)
        .map_err(|e| format!("k2_horizon L{l}: residual add: {e:?}"))?;

    Ok(())
}
