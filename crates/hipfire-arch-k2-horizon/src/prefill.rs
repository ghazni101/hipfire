// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon batched prefill — processes N tokens per step instead of 1.
//!
//! Wired into `generate_k2_horizon` behind `HIPFIRE_K2_PREFILL_SEQ` (default
//! ON = batched). The prior illegal-memory-access (hipMemcpy D2H code 700)
//! traced to a partial-warp `__shfl_down` hazard in
//! `deepseek4_moe_topk_bias_aware_batched_f32` when `n_exp % 32 != 0`, fixed
//! on 2026-09-11. Sequential `decode_step` prefill remains the fallback on
//! error and when `HIPFIRE_K2_PREFILL_SEQ=1`.
//!
//! Uses `weight_gemm` for batched MQ4G256V2 GEMM, batched RMSNorm, batched
//! RoPE, batched top-K routing, and the V2 indexed MoE GEMV kernels (which
//! already accept `batch_size` in grid Z). The KV cache is shared with the
//! decode state. After prefill, the decode loop takes over with PM4 replay.

use crate::config::K2HorizonConfig;
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, weight_gemm, weight_gemv,
    KvCacheExt,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Maximum tokens per prefill chunk. Larger = faster but more VRAM.
/// At hidden=2560, batch=256 uses ~2.6 MB per buffer × ~20 buffers ≈ 52 MB.
const PREFILL_MAX_BATCH: usize = 256;

/// Batched scratch buffers for prefill. All are [MAX_BATCH × dim] shaped.
pub struct PrefillScratch {
    pub tokens: GpuTensor,        // [MAX_BATCH] i32 token IDs
    pub positions: GpuTensor,     // [MAX_BATCH] i32 positions
    pub h: GpuTensor,             // [MAX_BATCH × hidden]
    pub normed: GpuTensor,        // [MAX_BATCH × hidden]
    pub fa_q: GpuTensor,          // [MAX_BATCH × q_dim]
    pub fa_k: GpuTensor,          // [MAX_BATCH × kv_dim]
    pub fa_v: GpuTensor,          // [MAX_BATCH × kv_dim]
    pub fa_attn_out: GpuTensor,   // [MAX_BATCH × q_dim]
    pub attn_gate_out: GpuTensor, // [MAX_BATCH × q_dim]
    pub scratch_h: GpuTensor,     // [MAX_BATCH × hidden]
    // Dense FFN
    pub dense_gate: GpuTensor, // [MAX_BATCH × dense_inter]
    pub dense_up: GpuTensor,   // [MAX_BATCH × dense_inter]
    pub dense_act: GpuTensor,  // [MAX_BATCH × dense_inter]
    // MoVA routing
    pub v_router_logits: GpuTensor, // [MAX_BATCH × mova_n_exp]
    pub v_topk_indices: GpuTensor,  // [MAX_BATCH × mova_k]
    pub v_topk_weights: GpuTensor,  // [MAX_BATCH × mova_k]
    pub v_x_rot: GpuTensor,         // [MAX_BATCH × hidden]
    pub v_rot_batch: GpuTensor,     // [MAX_BATCH × mova_k × hidden]
    pub v_expanded: GpuTensor,      // [MAX_BATCH × mova_k × kv_dim]
    // MoE FFN routing
    pub moe_router_logits: GpuTensor, // [MAX_BATCH × moe_n_exp]
    pub moe_topk_indices: GpuTensor,  // [MAX_BATCH × moe_k]
    pub moe_topk_weights: GpuTensor,  // [MAX_BATCH × moe_k]
    pub ffn_x_rot: GpuTensor,         // [MAX_BATCH × hidden]
    pub gate_batch: GpuTensor,        // [MAX_BATCH × moe_k × moe_inter]
    pub up_batch: GpuTensor,          // [MAX_BATCH × moe_k × moe_inter]
    pub rot_batch: GpuTensor,         // [MAX_BATCH × moe_k × moe_inter]
    pub down_expanded: GpuTensor,     // [MAX_BATCH × moe_k × hidden]
    pub shared_gate: GpuTensor,       // [MAX_BATCH × moe_inter]
    pub shared_up: GpuTensor,         // [MAX_BATCH × moe_inter]
    pub shared_act: GpuTensor,        // [MAX_BATCH × moe_inter]
    // Final
    pub final_norm_buf: GpuTensor, // [MAX_BATCH × hidden]
    pub logits: GpuTensor,         // [vocab] — last token only
}

impl PrefillScratch {
    pub fn new(gpu: &mut Gpu, cfg: &K2HorizonConfig) -> Result<Self, String> {
        let mb = PREFILL_MAX_BATCH;
        let hidden = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let mova_k = cfg.mova_num_experts_per_tok;
        let mova_n = cfg.mova_num_experts;
        let moe_k = cfg.num_experts_per_tok;
        let moe_n = cfg.num_experts;
        let vocab = cfg.vocab_size;

        let alloc = |gpu: &mut Gpu, n: usize, name: &str| -> Result<GpuTensor, String> {
            gpu.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("prefill alloc {name}: {e:?}"))
        };
        let alloc_i32 = |gpu: &mut Gpu, n: usize, name: &str| -> Result<GpuTensor, String> {
            gpu.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("prefill alloc_i32 {name}: {e:?}"))
        };

        Ok(Self {
            tokens: alloc_i32(gpu, mb, "tokens")?,
            positions: alloc_i32(gpu, mb, "positions")?,
            h: alloc(gpu, mb * hidden, "h")?,
            normed: alloc(gpu, mb * hidden, "normed")?,
            fa_q: alloc(gpu, mb * q_dim, "fa_q")?,
            fa_k: alloc(gpu, mb * kv_dim, "fa_k")?,
            fa_v: alloc(gpu, mb * kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, mb * q_dim, "fa_attn_out")?,
            attn_gate_out: alloc(gpu, mb * q_dim, "attn_gate_out")?,
            scratch_h: alloc(gpu, mb * hidden, "scratch_h")?,
            dense_gate: alloc(gpu, mb * dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, mb * dense_inter, "dense_up")?,
            dense_act: alloc(gpu, mb * dense_inter, "dense_act")?,
            v_router_logits: alloc(gpu, mb * mova_n, "v_router_logits")?,
            v_topk_indices: alloc(gpu, mb * mova_k, "v_topk_indices")?,
            v_topk_weights: alloc(gpu, mb * mova_k, "v_topk_weights")?,
            v_x_rot: alloc(gpu, mb * hidden, "v_x_rot")?,
            v_rot_batch: alloc(gpu, mb * mova_k * hidden, "v_rot_batch")?,
            v_expanded: alloc(gpu, mb * mova_k * kv_dim, "v_expanded")?,
            moe_router_logits: alloc(gpu, mb * moe_n, "moe_router_logits")?,
            moe_topk_indices: alloc(gpu, mb * moe_k, "moe_topk_indices")?,
            moe_topk_weights: alloc(gpu, mb * moe_k, "moe_topk_weights")?,
            ffn_x_rot: alloc(gpu, mb * hidden, "ffn_x_rot")?,
            gate_batch: alloc(gpu, mb * moe_k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, mb * moe_k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, mb * moe_k * moe_inter, "rot_batch")?,
            down_expanded: alloc(gpu, mb * moe_k * hidden, "down_expanded")?,
            shared_gate: alloc(gpu, mb * moe_inter, "shared_gate")?,
            shared_up: alloc(gpu, mb * moe_inter, "shared_up")?,
            shared_act: alloc(gpu, mb * moe_inter, "shared_act")?,
            final_norm_buf: alloc(gpu, mb * hidden, "final_norm_buf")?,
            logits: alloc(gpu, vocab, "logits")?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let PrefillScratch {
            tokens,
            positions,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            attn_gate_out,
            scratch_h,
            dense_gate,
            dense_up,
            dense_act,
            v_router_logits,
            v_topk_indices,
            v_topk_weights,
            v_x_rot,
            v_rot_batch,
            v_expanded,
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
            final_norm_buf,
            logits,
        } = self;
        for t in [
            tokens,
            positions,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            attn_gate_out,
            scratch_h,
            dense_gate,
            dense_up,
            dense_act,
            v_router_logits,
            v_topk_indices,
            v_topk_weights,
            v_x_rot,
            v_rot_batch,
            v_expanded,
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
            final_norm_buf,
            logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// View the first `n * dim` elements of a [MAX_BATCH × dim] buffer as a flat [n*dim] tensor.
fn view(ps: &PrefillScratch, buf: &GpuTensor, n: usize, dim: usize) -> GpuTensor {
    let _ = ps;
    GpuTensor {
        buf: unsafe { buf.buf.alias() },
        shape: vec![n * dim],
        dtype: DType::F32,
    }
}

/// Batched attention helper for prefill.
fn attend_batch(
    cfg: &K2HorizonConfig,
    ps: &PrefillScratch,
    kv: &crate::forward::K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    n: usize,
    start_pos: usize,
) -> Result<(), String> {
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    let q = view(ps, &ps.fa_q, n, q_dim);
    let k = view(ps, &ps.fa_k, n, kv_dim);
    let v = view(ps, &ps.fa_v, n, kv_dim);
    let out = view(ps, &ps.fa_attn_out, n, q_dim);
    let positions = view(ps, &ps.positions, n, 1);

    let ctx = DispatchCtx::new(gpu);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: start_pos + n - 1,
        q8_windowed: true,
        window: 0,
        flash_mode: 0,
        capture_mode: false,
        batch_size: n,
        ..kv.kv.tier_inputs()
    })
    .map_err(|e| format!("k2_horizon prefill L{l}: kv tier: {e}"))?;

    let io = AttnParams {
        q: &q,
        k: &k,
        v: &v,
        k_cache: &kv.kv.k_gpu[l],
        v_cache: &kv.kv.v_gpu[l],
        k_scales: None,
        v_scales: None,
        pos_buf: &kv.pos_buf,
        pos: start_pos + n - 1,
        positions: Some(&positions),
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        physical_cap: kv.kv.physical_cap,
        batch_size: n,
        max_ctx_len: start_pos + n,
        flash_partials: Some(&kv.flash_partials),
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &out,
    };

    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
        .map_err(|e| format!("k2_horizon prefill L{l}: attend: {e:?}"))?;
    Ok(())
}

/// Batched dense layer (layers 0–2).
fn forward_dense_layer_batch(
    cfg: &K2HorizonConfig,
    layer: &DenseLayerWeights,
    ps: &PrefillScratch,
    kv: &crate::forward::K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    n: usize,
    start_pos: usize,
) -> Result<(), String> {
    let hidden = cfg.dim;
    let eps = cfg.norm_eps;
    let n_groups = cfg.layernorm_num_groups;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let dense_inter = cfg.intermediate_size;

    let h = view(ps, &ps.h, n, hidden);
    let normed = view(ps, &ps.normed, n, hidden);
    let fa_q = view(ps, &ps.fa_q, n, q_dim);
    let fa_k = view(ps, &ps.fa_k, n, kv_dim);
    let fa_v = view(ps, &ps.fa_v, n, kv_dim);
    let fa_attn_out = view(ps, &ps.fa_attn_out, n, q_dim);
    let attn_gate_out = view(ps, &ps.attn_gate_out, n, q_dim);
    let scratch_h = view(ps, &ps.scratch_h, n, hidden);
    let positions = view(ps, &ps.positions, n, 1);

    // normed = grouped_rmsnorm(h, attn_norm)
    gpu.grouped_rmsnorm_f32(&h, &layer.attn_norm, &normed, n, hidden, n_groups, eps)
        .map_err(|e| format!("k2_horizon prefill L{l}: input norm: {e:?}"))?;

    // q/k/v projections (batched GEMM)
    weight_gemm(gpu, &layer.wq, &normed, &fa_q, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: q_proj: {e}"))?;
    weight_gemm(gpu, &layer.wk, &normed, &fa_k, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: k_proj: {e}"))?;
    weight_gemm(gpu, &layer.wv, &normed, &fa_v, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: v_proj: {e}"))?;

    // Batched RoPE
    gpu.rope_batched_f32(
        &fa_q,
        &fa_k,
        &positions,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
        n,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: rope: {e:?}"))?;

    // KV write + batched attention
    attend_batch(cfg, ps, kv, gpu, l, n, start_pos)?;

    // softplus post-attention gate (fused, element-wise on [n × q_dim])
    weight_gemm(gpu, &layer.attn_gate, &normed, &attn_gate_out, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: attn gate: {e}"))?;
    gpu.softplus_gate_f32(&attn_gate_out, &fa_attn_out)
        .map_err(|e| format!("k2_horizon prefill L{l}: softplus gate: {e:?}"))?;

    // h += o_proj(attn_out)
    weight_gemm(gpu, &layer.wo, &fa_attn_out, &scratch_h, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: o_proj: {e}"))?;
    gpu.add_inplace_f32(&h, &scratch_h)
        .map_err(|e| format!("k2_horizon prefill L{l}: o_proj add: {e:?}"))?;

    // FFN norm
    gpu.grouped_rmsnorm_f32(&h, &layer.ffn_norm, &normed, n, hidden, n_groups, eps)
        .map_err(|e| format!("k2_horizon prefill L{l}: ffn norm: {e:?}"))?;

    // Dense SwiGLU
    let dense_gate = view(ps, &ps.dense_gate, n, dense_inter);
    let dense_up = view(ps, &ps.dense_up, n, dense_inter);
    let dense_act = view(ps, &ps.dense_act, n, dense_inter);
    weight_gemm(gpu, &layer.w_gate, &normed, &dense_gate, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: dense gate: {e}"))?;
    weight_gemm(gpu, &layer.w_up, &normed, &dense_up, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: dense up: {e}"))?;
    gpu.silu_mul_f32(&dense_gate, &dense_up, &dense_act)
        .map_err(|e| format!("k2_horizon prefill L{l}: dense silu_mul: {e:?}"))?;
    weight_gemm(gpu, &layer.w_down, &dense_act, &scratch_h, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: dense down: {e}"))?;
    gpu.add_inplace_f32(&h, &scratch_h)
        .map_err(|e| format!("k2_horizon prefill L{l}: dense add: {e:?}"))?;

    Ok(())
}

/// Batched MoVA value-expert routing.
fn forward_mova_value_routing_batch(
    cfg: &K2HorizonConfig,
    attn: &crate::weights::MovaAttnWeights,
    ps: &PrefillScratch,
    gpu: &mut Gpu,
    l: usize,
    n: usize,
) -> Result<(), String> {
    let mova_top_k = cfg.mova_num_experts_per_tok;
    let scaling = cfg.router_scaling_factor;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let hidden = cfg.dim;

    let normed = view(ps, &ps.normed, n, hidden);
    let v_router_logits = view(ps, &ps.v_router_logits, n, cfg.mova_num_experts);
    let v_topk_indices = view(ps, &ps.v_topk_indices, n, mova_top_k);
    let v_topk_weights = view(ps, &ps.v_topk_weights, n, mova_top_k);
    let v_x_rot = view(ps, &ps.v_x_rot, n, hidden);
    let v_rot_batch = view(ps, &ps.v_rot_batch, n * mova_top_k, hidden);
    let v_expanded = view(ps, &ps.v_expanded, n * mova_top_k, kv_dim);
    let fa_v = view(ps, &ps.fa_v, n, kv_dim);

    // 1. router GEMM → [n × 64]
    weight_gemm(gpu, &attn.v_router, &normed, &v_router_logits, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: v_router: {e}"))?;

    // 2. sigmoid in-place
    gpu.sigmoid_f32(&v_router_logits)
        .map_err(|e| format!("k2_horizon prefill L{l}: v_router sigmoid: {e:?}"))?;

    // 3. Batched GPU top-K
    let bias = attn
        .v_router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon prefill L{l}: v_router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_batched_f32(
        &v_router_logits,
        bias,
        &v_topk_indices,
        &v_topk_weights,
        cfg.mova_num_experts as i32,
        mova_top_k as i32,
        scaling,
        n as i32,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: v_router topk: {e:?}"))?;
    // 4. FWHT-rotate normed → [n × hidden]
    rotate_x_mq_batched_for(gpu, &attn.v_experts[0], &normed, &v_x_rot, hidden, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: v rotate: {e:?}"))?;

    // 5. Replicate [n × hidden] → [n × k_top × hidden] for indexed down GEMV
    gpu.replicate_batched_f32(&v_x_rot, &v_rot_batch, hidden, mova_top_k, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: v replicate: {e:?}"))?;

    // 6. V2 indexed MoE down GEMV: all v_experts in one launch, batch_size=n
    gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
        &attn.v_expert_ptrs,
        &v_topk_indices,
        &v_rot_batch,
        &v_expanded,
        kv_dim,
        hidden,
        mova_top_k,
        n,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: v_expert indexed gemv: {e:?}"))?;

    // 7. SiLU activation on all expert outputs
    gpu.silu_f32(&v_expanded, &v_expanded)
        .map_err(|e| format!("k2_horizon prefill L{l}: v_expert silu: {e:?}"))?;

    // 8. Zero fa_v, then accumulate weighted expert outputs
    gpu.zero_f32(&fa_v)
        .map_err(|e| format!("k2_horizon prefill L{l}: zero fa_v: {e:?}"))?;
    gpu.moe_down_combine_k8_batched(&v_expanded, &v_topk_weights, &fa_v, kv_dim, mova_top_k, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: v_expert combine: {e:?}"))?;

    Ok(())
}

/// Batched sigmoid-routed MoE FFN.
fn forward_sigmoid_moe_ffn_batch(
    cfg: &K2HorizonConfig,
    ffn: &crate::weights::MoeFfnWeights,
    ps: &PrefillScratch,
    gpu: &mut Gpu,
    l: usize,
    n: usize,
) -> Result<(), String> {
    let top_k = cfg.num_experts_per_tok;
    let scaling = cfg.router_scaling_factor;
    let hidden = cfg.dim;
    let moe_inter = cfg.moe_intermediate_size;

    let normed = view(ps, &ps.normed, n, hidden);
    let h = view(ps, &ps.h, n, hidden);
    let moe_router_logits = view(ps, &ps.moe_router_logits, n, cfg.num_experts);
    let moe_topk_indices = view(ps, &ps.moe_topk_indices, n, top_k);
    let moe_topk_weights = view(ps, &ps.moe_topk_weights, n, top_k);
    let ffn_x_rot = view(ps, &ps.ffn_x_rot, n, hidden);
    let gate_batch = view(ps, &ps.gate_batch, n * top_k, moe_inter);
    let up_batch = view(ps, &ps.up_batch, n * top_k, moe_inter);
    let rot_batch = view(ps, &ps.rot_batch, n * top_k, moe_inter);
    let down_expanded = view(ps, &ps.down_expanded, n * top_k, hidden);
    let shared_gate = view(ps, &ps.shared_gate, n, moe_inter);
    let shared_up = view(ps, &ps.shared_up, n, moe_inter);
    let shared_act = view(ps, &ps.shared_act, n, moe_inter);

    // 1. router GEMM → [n × 100]
    weight_gemm(gpu, &ffn.router, &normed, &moe_router_logits, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: moe router: {e}"))?;

    // 2. sigmoid in-place
    gpu.sigmoid_f32(&moe_router_logits)
        .map_err(|e| format!("k2_horizon prefill L{l}: moe sigmoid: {e:?}"))?;

    // 3. Batched GPU top-K
    let bias = ffn
        .router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon prefill L{l}: router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_batched_f32(
        &moe_router_logits,
        bias,
        &moe_topk_indices,
        &moe_topk_weights,
        cfg.num_experts as i32,
        top_k as i32,
        scaling,
        n as i32,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: moe topk: {e:?}"))?;

    // 4. FWHT-rotate normed → [n × hidden]
    rotate_x_mq_batched_for(gpu, &ffn.experts[0].gate_up, &normed, &ffn_x_rot, hidden, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: ffn rotate: {e:?}"))?;

    // 5. V2 indexed gate_up GEMV: all top_k experts, batch_size=n
    gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
        &ffn.expert_gate_up_ptrs,
        &moe_topk_indices,
        &ffn_x_rot,
        &gate_batch,
        &up_batch,
        2 * moe_inter,
        hidden,
        top_k,
        n,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: gate_up indexed gemv: {e:?}"))?;

    // 6. Fused silu_mul + FWHT rotate → [n × k_top × moe_inter]
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        &ffn.experts[0].down,
        &gate_batch,
        &up_batch,
        &rot_batch,
        moe_inter,
        n * top_k,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: fused silu_mul rotate: {e:?}"))?;

    // 7. V2 indexed down GEMV: all top_k experts, batch_size=n
    gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
        &ffn.expert_down_ptrs,
        &moe_topk_indices,
        &rot_batch,
        &down_expanded,
        hidden,
        moe_inter,
        top_k,
        n,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: down indexed gemv: {e:?}"))?;

    // 8. Combine: h += Σ weight[k] * down_expanded[k]
    gpu.moe_down_combine_k8_batched(&down_expanded, &moe_topk_weights, &h, hidden, top_k, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: moe combine: {e:?}"))?;

    // 9. Shared expert (batched SwiGLU)
    weight_gemm(gpu, &ffn.shared.gate, &normed, &shared_gate, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: shared gate: {e}"))?;
    weight_gemm(gpu, &ffn.shared.up, &normed, &shared_up, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: shared up: {e}"))?;
    gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_act)
        .map_err(|e| format!("k2_horizon prefill L{l}: shared silu_mul: {e:?}"))?;

    // shared down: need residual add. weight_gemm doesn't do residual,
    // so: scratch_h = shared_down, then h += scratch_h
    let scratch_h = view(ps, &ps.scratch_h, n, hidden);
    weight_gemm(gpu, &ffn.shared.down, &shared_act, &scratch_h, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: shared down: {e}"))?;
    gpu.add_inplace_f32(&h, &scratch_h)
        .map_err(|e| format!("k2_horizon prefill L{l}: shared add: {e:?}"))?;

    Ok(())
}

/// Batched MoE layer (layers 3–47): MoVA attention + sigmoid MoE FFN.
fn forward_moe_layer_batch(
    cfg: &K2HorizonConfig,
    layer: &MovaLayerWeights,
    ps: &PrefillScratch,
    kv: &crate::forward::K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    n: usize,
    start_pos: usize,
) -> Result<(), String> {
    let hidden = cfg.dim;
    let eps = cfg.norm_eps;
    let n_groups = cfg.layernorm_num_groups;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let attn = &layer.attn;
    let ffn = &layer.ffn;

    // ── Attention branch ──
    let h = view(ps, &ps.h, n, hidden);
    let normed = view(ps, &ps.normed, n, hidden);
    let fa_q = view(ps, &ps.fa_q, n, q_dim);
    let fa_k = view(ps, &ps.fa_k, n, kv_dim);
    let fa_attn_out = view(ps, &ps.fa_attn_out, n, q_dim);
    let attn_gate_out = view(ps, &ps.attn_gate_out, n, q_dim);
    let scratch_h = view(ps, &ps.scratch_h, n, hidden);
    let positions = view(ps, &ps.positions, n, 1);

    // normed = grouped_rmsnorm(h, attn_norm) — must precede q/k projections
    // AND the MoVA router (both consume `normed`).
    gpu.grouped_rmsnorm_f32(&h, &layer.attn_norm, &normed, n, hidden, n_groups, eps)
        .map_err(|e| format!("k2_horizon prefill L{l}: input norm: {e:?}"))?;

    // q/k projections (batched GEMM). MoVA has no v_proj — v comes from
    // the routed value experts below.
    weight_gemm(gpu, &attn.wq, &normed, &fa_q, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: q_proj: {e}"))?;
    weight_gemm(gpu, &attn.wk, &normed, &fa_k, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: k_proj: {e}"))?;

    forward_mova_value_routing_batch(cfg, attn, ps, gpu, l, n)?;

    gpu.rope_batched_f32(
        &fa_q,
        &fa_k,
        &positions,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
        n,
    )
    .map_err(|e| format!("k2_horizon prefill L{l}: rope: {e:?}"))?;

    attend_batch(cfg, ps, kv, gpu, l, n, start_pos)?;

    weight_gemm(gpu, &attn.attn_gate, &normed, &attn_gate_out, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: attn gate: {e}"))?;
    gpu.softplus_gate_f32(&attn_gate_out, &fa_attn_out)
        .map_err(|e| format!("k2_horizon prefill L{l}: softplus gate: {e:?}"))?;

    weight_gemm(gpu, &attn.wo, &fa_attn_out, &scratch_h, n)
        .map_err(|e| format!("k2_horizon prefill L{l}: o_proj: {e}"))?;
    gpu.add_inplace_f32(&h, &scratch_h)
        .map_err(|e| format!("k2_horizon prefill L{l}: o_proj add: {e:?}"))?;

    // ── FFN branch ──
    gpu.grouped_rmsnorm_f32(&h, &layer.ffn_norm, &normed, n, hidden, n_groups, eps)
        .map_err(|e| format!("k2_horizon prefill L{l}: ffn norm: {e:?}"))?;

    forward_sigmoid_moe_ffn_batch(cfg, ffn, ps, gpu, l, n)?;

    Ok(())
}

/// Process a chunk of tokens through all layers. Returns the last token's logits.
pub fn forward_prefill_chunk(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    ps: &PrefillScratch,
    kv: &crate::forward::K2HorizonState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let n = tokens.len();
    if n > PREFILL_MAX_BATCH {
        return Err(format!(
            "k2_horizon prefill: chunk {n} > max {PREFILL_MAX_BATCH}"
        ));
    }
    let hidden = cfg.dim;

    // Upload token IDs
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    let tokens_view = view(ps, &ps.tokens, n, 1);
    gpu.hip
        .memcpy_htod(&tokens_view.buf, tokens_bytes)
        .map_err(|e| format!("k2_horizon prefill: upload tokens: {e:?}"))?;

    // Upload positions
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    let positions_view = view(ps, &ps.positions, n, 1);
    gpu.hip
        .memcpy_htod(&positions_view.buf, positions_bytes)
        .map_err(|e| format!("k2_horizon prefill: upload positions: {e:?}"))?;

    // Batched embedding lookup → [n × hidden]
    let h = view(ps, &ps.h, n, hidden);
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &h, &tokens_view, n, cfg.dim)
        .map_err(|e| format!("k2_horizon prefill: embed lookup: {e:?}"))?;

    // Run all layers
    for (l, layer) in weights.dense_layers.iter().enumerate() {
        forward_dense_layer_batch(cfg, layer, ps, kv, gpu, l, n, start_pos)?;
    }
    for (l, layer) in weights.moe_layers.iter().enumerate() {
        let global_l = weights.dense_layers.len() + l;
        forward_moe_layer_batch(cfg, layer, ps, kv, gpu, global_l, n, start_pos)?;
    }

    // Final norm (batched) + lm_head (last token only)
    let h_all = view(ps, &ps.h, n, hidden);
    let final_norm_all = view(ps, &ps.final_norm_buf, n, hidden);
    gpu.grouped_rmsnorm_f32(
        &h_all,
        &weights.final_norm,
        &final_norm_all,
        n,
        hidden,
        cfg.layernorm_num_groups,
        cfg.norm_eps,
    )
    .map_err(|e| format!("k2_horizon prefill: final norm: {e:?}"))?;

    // Extract last row from final_norm_buf for lm_head (offset by (n-1)*hidden floats)
    let last_row_offset_bytes = (n - 1) * hidden * 4;
    let final_norm_last = GpuTensor {
        buf: unsafe {
            DeviceBuffer::from_raw(
                (ps.final_norm_buf.buf.as_ptr() as *mut u8).add(last_row_offset_bytes)
                    as *mut std::ffi::c_void,
                hidden * 4,
            )
        },
        shape: vec![hidden],
        dtype: DType::F32,
    };

    if let Some(lm_head) = &weights.lm_head {
        weight_gemv(gpu, lm_head, &final_norm_last, &ps.logits)
            .map_err(|e| format!("k2_horizon prefill: lm_head: {e}"))?;
    } else {
        return Err("k2_horizon: tied embeddings not yet supported".into());
    }

    gpu.download_f32(&ps.logits)
        .map_err(|e| format!("k2_horizon prefill: download logits: {e:?}"))
}

/// Full prefill: process all prompt tokens in chunks.
/// Returns the last token's logits for sampling the first decode token.
pub fn forward_prefill_batch(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    ps: &PrefillScratch,
    kv: &mut crate::forward::K2HorizonState,
    gpu: &mut Gpu,
    prompt_ids: &[u32],
) -> Result<Vec<f32>, String> {
    let mut last_logits: Vec<f32> = Vec::new();
    let mut start_pos = kv.n_tokens;

    for chunk in prompt_ids.chunks(PREFILL_MAX_BATCH) {
        last_logits = forward_prefill_chunk(cfg, weights, ps, kv, gpu, chunk, start_pos)?;
        start_pos += chunk.len();
        kv.n_tokens = start_pos;
    }

    Ok(last_logits)
}
