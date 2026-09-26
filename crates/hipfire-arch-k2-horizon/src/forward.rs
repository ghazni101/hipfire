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
//! Both MoVA value-expert routing and MoE FFN routing use GPU-side
//! `deepseek4_moe_topk_bias_aware_f32` for bias-aware top-K (no D2H),
//! followed by V2 indexed MoE GEMV kernels
//! (`gemv_mq4g256v2_moe_*_k8_indexed_batched`) that decode fp16 per-128
//! headers (MQ4G256V2 / qt=44). GPU-side sampling (`argmax_f32` /
//! `sample_top_p`) eliminates the ~1 MB logits download per token.
//! PM4 retained-replay is enabled: the forward body (layers + final norm
//! + lm_head) is captured once during warmup and replayed via PM4/AQL on
//! subsequent decode steps. KV write + attention uses the dispatch layer
//! (`KvTierPlan` + `AttnParams`), matching the cohere2moe pattern.

use crate::config::K2HorizonConfig;
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, weight_gemv, weight_gemv_prerotated, KvCache, KvCacheExt,
};
use rdna_compute::{DType, Gpu, GpuTensor};
// ─── State ──────────────────────────────────────────────────────────────

/// Validated VRAM ceiling per family on gfx1100 (24.5 GB usable):
/// dense K2-7B ran 40960 with margin (64k OOM'd at KV alloc, 26 MB free),
/// so the dense default must never exceed the demonstrated envelope.
/// MoVA-36B @ mq4l weights (~20.6 GB on-device) leave ~3.9 GB for KV —
/// Q8 KV is 104.5 KB/token (48 L × 8 KVH × 128 hd × 2 × 136 B), so the
/// mq4l MoVA ceiling is 24576 (2.57 GB, ~0.1 GB margin). 40960 OOM'd
/// reproducibly (20 MB free at KV alloc). MQ3G256Lloyd weights (~14 GB)
/// free ~10.5 GB, which covers 65536 tokens (6.7 GB KV) with ~3.8 GB of
/// state/scratch headroom; 131072 does not fit (13.4 GB KV alone).
const DEFAULT_MAX_SEQ_DENSE: usize = 40960;
const DEFAULT_MAX_SEQ_MOVA: usize = 24576;
const DEFAULT_MAX_SEQ_MOVA_MQ3L: usize = 65536;

fn max_seq_cap(cfg: &K2HorizonConfig, expert_dtype: Option<DType>) -> usize {
    if cfg.mova_num_experts > 0 {
        if expert_dtype == Some(DType::MQ3G256Lloyd) {
            DEFAULT_MAX_SEQ_MOVA_MQ3L
        } else {
            DEFAULT_MAX_SEQ_MOVA
        }
    } else {
        DEFAULT_MAX_SEQ_DENSE
    }
}

/// Flash prefill sub-batch size. Smaller = less VRAM. K2-Horizon has 32 heads
/// × 128 head_dim; at max_seq=40960 this is 32×321×130×16 ≈ 21 MB per
/// sub-batch row set (sized in PrefillScratch, not here).
pub(crate) const FLASH_PREFILL_SUBBATCH: usize = 16;

/// Per-decode GPU scratch + KV cache for K2-Horizon.
///
/// MoVA and MoE FFN use GPU-indexed MoE GEMV kernels with batched scratch
/// (no per-expert CPU download/upload). `moe_top_k` is the max of
/// `mova_num_experts_per_tok` (4) and `num_experts_per_tok` (8); the two
/// paths keep separate topk buffers (v_* vs moe_*) sized to that bound.
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
    // MoVA routing scratch (fully GPU-side, no D2H)
    pub v_router_logits: GpuTensor, // [mova_num_experts] = [64]
    pub v_topk_indices: GpuTensor,  // [mova_k_top] = [4] i32-in-F32
    pub v_topk_weights: GpuTensor,  // [mova_k_top] = [4]
    pub v_rot_batch: GpuTensor, // [mova_k_top * hidden] = [4 * 2560] — replicated rotated input for indexed GEMV
    pub v_expanded: GpuTensor,  // [mova_k_top * kv_dim] = [4 * 1024] — per-expert GEMV output
    pub attn_gate_out: GpuTensor, // [n_heads * head_dim] = [4096] — gate_proj output
    /// Fused attention-GEMV output: dense [wq‖wk‖wv‖gate] = [10240],
    /// MoVA [wq‖wk‖v_router‖gate] = [9280]. Views are sliced per layer.
    pub attn_fused_out: GpuTensor,
    // dense FFN scratch
    pub dense_gate: GpuTensor, // [intermediate_size] = [6144]
    pub dense_up: GpuTensor,   // [intermediate_size] = [6144]
    pub dense_act: GpuTensor,  // [intermediate_size] = [6144]

    // MoE FFN routing scratch (GPU indexed GEMV)
    pub moe_router_logits: GpuTensor, // [num_experts] = [100]
    pub moe_topk_indices: GpuTensor,  // [moe_k_top] = [8] i32-in-F32
    pub moe_topk_weights: GpuTensor,  // [moe_k_top] = [8]
    pub gate_batch: GpuTensor,        // [moe_k_top * moe_inter] = [8 * 768]
    pub up_batch: GpuTensor,          // [moe_k_top * moe_inter] = [8 * 768]
    pub rot_batch: GpuTensor,         // [moe_k_top * moe_inter] = [8 * 768]
    pub down_expanded: GpuTensor,     // [moe_k_top * hidden] = [8 * 2560]
    pub shared_gate: GpuTensor,       // [moe_intermediate_size] = [768]
    pub shared_up: GpuTensor,         // [moe_intermediate_size] = [768]
    pub shared_act: GpuTensor,        // [moe_intermediate_size] = [768]
    /// Fused FFN-GEMV output: dense [gate‖up] = [12288],
    /// MoE [router‖shared_gate‖shared_up] = [1636]. Views sliced per layer.
    pub ffn_fused_out: GpuTensor,
    pub scratch_h: GpuTensor, // [hidden] — temp for weight_gemv + add_inplace (PM4-safe residual)
    pub normed_rot: GpuTensor, // [hidden] — FWHT(normed), rotated once per layer, reused by all normed-reading projections
    pub proj_rot: GpuTensor, // [max(q_dim, dense_inter, moe_inter)] — rotate scratch for non-normed GEMV inputs (o_proj, w_down, shared down)

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
    pub flash_partials: GpuTensor, // flash attention scratch (pre-allocated)
    pub sample_buf: GpuTensor,     // [2] F32 — (token_id, rng_state) from GPU sampler
    pub repeat_buf: GpuTensor,     // [64] F32 — repeat penalty window (unused, required by kernel)

    // PM4 retained-replay warmup tracking
    pub retained_warmed_up: bool,
    pub retained_state_poisoned: bool,
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
            v_rot_batch,
            v_expanded,
            attn_gate_out,
            attn_fused_out,
            dense_gate,
            dense_up,
            dense_act,
            moe_router_logits,
            moe_topk_indices,
            moe_topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            shared_gate,
            shared_up,
            shared_act,
            ffn_fused_out,
            normed_rot,
            proj_rot,
            scratch_h,
            final_norm_buf,
            logits,
            sample_buf,
            repeat_buf,
            flash_partials,
            retained_warmed_up: _,
            retained_state_poisoned: _,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            v_router_logits,
            v_topk_indices,
            v_topk_weights,
            v_rot_batch,
            v_expanded,
            attn_gate_out,
            attn_fused_out,
            dense_gate,
            dense_up,
            dense_act,
            moe_router_logits,
            moe_topk_indices,
            moe_topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            shared_gate,
            normed_rot,
            proj_rot,
            shared_up,
            shared_act,
            ffn_fused_out,
            scratch_h,
            final_norm_buf,
            logits,
            flash_partials,
            sample_buf,
            repeat_buf,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    pub fn new(gpu: &mut Gpu, cfg: &K2HorizonConfig) -> Result<Self, String> {
        // Config-only path has no weight dtype in scope → conservative cap.
        let max_seq = cfg
            .max_position_embeddings
            .min(max_seq_cap(cfg, None));
        Self::new_with_max_seq(gpu, cfg, max_seq, None)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &K2HorizonConfig,
        max_seq: usize,
        expert_dtype: Option<DType>,
    ) -> Result<Self, String> {
        let max_seq = max_seq
            .min(cfg.max_position_embeddings)
            .min(max_seq_cap(cfg, expert_dtype));
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
        let kv = KvCache::new_gpu_q8(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq)
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
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            v_router_logits: alloc(gpu, mova_n_exp, "v_router_logits")?,
            v_topk_indices: alloc(gpu, mova_k, "v_topk_indices")?,
            v_topk_weights: alloc(gpu, mova_k, "v_topk_weights")?,
            v_rot_batch: alloc(gpu, mova_k * hidden, "v_rot_batch")?,
            v_expanded: alloc(gpu, mova_k * kv_dim, "v_expanded")?,
            attn_gate_out: alloc(gpu, q_dim, "attn_gate_out")?,
            // Dense fused [q‖k‖v‖gate] = 2*q_dim + 2*kv_dim; MoVA fused
            // [q‖k‖router‖gate] = 2*q_dim + kv_dim + mova_n_exp. Max covers both.
            attn_fused_out: alloc(gpu, 2 * q_dim + 2 * kv_dim, "attn_fused_out")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            moe_router_logits: alloc(gpu, n_exp, "moe_router_logits")?,
            moe_topk_indices: alloc(gpu, moe_k, "moe_topk_indices")?,
            moe_topk_weights: alloc(gpu, moe_k, "moe_topk_weights")?,
            gate_batch: alloc(gpu, moe_k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, moe_k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, moe_k * moe_inter, "rot_batch")?,
            normed_rot: alloc(gpu, hidden, "normed_rot")?,
            proj_rot: alloc(gpu, q_dim.max(dense_inter).max(moe_inter), "proj_rot")?,
            down_expanded: alloc(gpu, moe_k * hidden, "down_expanded")?,
            shared_gate: alloc(gpu, moe_inter, "shared_gate")?,
            shared_up: alloc(gpu, moe_inter, "shared_up")?,
            shared_act: alloc(gpu, moe_inter, "shared_act")?,
            // Dense fused [gate‖up] = 2*dense_inter; MoE fused
            // [router‖shared_gate‖shared_up] = n_exp + 2*moe_inter. Max covers both.
            ffn_fused_out: alloc(
                gpu,
                (2 * dense_inter).max(n_exp + 2 * moe_inter),
                "ffn_fused_out",
            )?,
            scratch_h: alloc(gpu, hidden, "scratch_h")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            sample_buf: alloc(gpu, 2, "sample_buf")?,
            repeat_buf: alloc(gpu, 64, "repeat_buf")?,
            // Decode needs partials for a single query row; the ×SUBBATCH
            // prefill-sized buffer lives in PrefillScratch (freed after
            // prefill) so it isn't resident during decode. Size against the
            // q8 decode kernel's tile (q8_flash_tile_size, 32 on gfx1100) —
            // NOT attn_tile_size (128) — or the buffer is 4× undersized and
            // the kernel writes OOB.
            flash_partials: {
                let tile = rdna_compute::attention::q8_flash_tile_size(
                    &gpu.arch,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    max_seq,
                );
                let max_tiles = (max_seq + tile - 1) / tile;
                alloc(
                    gpu,
                    cfg.n_heads * max_tiles * (2 + cfg.head_dim),
                    "flash_partials",
                )?
            },
            retained_warmed_up: false,
            retained_state_poisoned: false,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.retained_warmed_up = false;
        self.retained_state_poisoned = false;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("k2_horizon reset: clear kv: {e:?}"))?;
        Ok(())
    }
}

// ─── Forward ────────────────────────────────────────────────────────────

/// Stage per-token inputs that must run OUTSIDE PM4 capture:
/// position H2D + embedding lookup. The token_id is a host arg to the
/// embedding kernel (not a device buffer), so it cannot be captured.
pub fn prepare_decode_inputs(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    if (position as usize) >= state.max_seq {
        return Err(format!(
            "k2_horizon: position {} >= max_seq {}",
            position, state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &position.to_ne_bytes())
        .map_err(|e| format!("k2_horizon: stage pos: {e:?}"))?;
    gpu.embedding_lookup_q8(&weights.token_embd, &state.h, token_id, cfg.dim)
        .map_err(|e| format!("k2_horizon: embed lookup: {e:?}"))?;
    Ok(())
}

/// Run the decode body (layers + final norm + lm_head). This is the
/// PM4-capturable region — all kernels use `launch_maybe_blob` with
/// device-pointer kernarg blobs. The position is read from `pos_buf`
/// (staged by `prepare_decode_inputs`), so it remains dynamic across
/// PM4 replays.
pub fn run_decode_body(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
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

/// Grouped RMSNorm + one shared FWHT rotation of `normed` into
/// `state.normed_rot`. All MQ4G256V2 projections that read `normed` in a
/// layer share the same fixed FWHT rotation (weight-independent — the AWQ
/// branch is the only weight-dependent variant), so rotating once and
/// reusing the buffer across wq/wk/v_router/attn_gate/router/gate/up
/// eliminates ~8 redundant `mq_rotate_x` launches per layer.
pub(crate) fn norm_and_rotate(
    cfg: &K2HorizonConfig,
    norm_w: &GpuTensor,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    // Fused grouped-RMSNorm + FWHT when the chunk is a whole number of
    // 256-element FWHT groups (K2-Horizon: 2560/2 = 1280 = 5 groups).
    // One launch instead of two; writes both `normed` and `normed_rot`.
    if cfg.dim % cfg.layernorm_num_groups == 0 && (cfg.dim / cfg.layernorm_num_groups) % 256 == 0 {
        gpu.grouped_rmsnorm_rotate_mq(
            &state.h,
            norm_w,
            &state.normed,
            &state.normed_rot,
            1,
            cfg.dim,
            cfg.layernorm_num_groups,
            cfg.norm_eps,
        )
        .map_err(|e| format!("k2_horizon L{l}: fused norm+rotate: {e:?}"))?;
        return Ok(());
    }
    gpu.grouped_rmsnorm_f32(
        &state.h,
        norm_w,
        &state.normed,
        1,
        cfg.dim,
        cfg.layernorm_num_groups,
        cfg.norm_eps,
    )
    .map_err(|e| format!("k2_horizon L{l}: norm: {e:?}"))?;
    gpu.rotate_x_mq(&state.normed, &state.normed_rot, cfg.dim)
        .map_err(|e| format!("k2_horizon L{l}: normed rotate: {e:?}"))?;
    Ok(())
}

/// Residual GEMV: `h += W · x` where `x` is rotated into `proj_rot` first.
/// On gfx1100 + MQ4G256V2 uses the PM4-safe single-row residual kernel
/// (no private scratch — the dual-row `gemv_mq4g256v2_residual` is rejected
/// by PM4 dispatch). Falls back to gemv + add_inplace elsewhere.
pub(crate) fn gemv_residual_prerotated(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    state: &K2HorizonState,
    h: &GpuTensor,
) -> rdna_compute::HipResult<()> {
    gpu.maybe_capture_activation(&w.buf, x, 1, w.k);
    gpu.rotate_x_mq(x, &state.proj_rot, w.k)?;
    if w.gpu_dtype == DType::MQ4G256V2 && gpu.arch_caps.is_gfx1100() {
        return gpu.gemv_mq4g256v2_residual_noscratch(&w.buf, &state.proj_rot, h, w.m, w.k);
    }
    weight_gemv_prerotated(gpu, w, x, Some(&state.proj_rot), &state.scratch_h)?;
    gpu.add_inplace_f32(h, &state.scratch_h)
}

/// GEMV against `normed`, consuming the shared `normed_rot` when the weight
/// uses the fixed FWHT rotation (non-AWQ). Falls back to plain
/// `weight_gemv` for AWQ-scaled or non-rotating dtypes so the shared buffer
/// is never fed a rotation it wasn't built for.
pub(crate) fn gemv_normed(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
    state: &K2HorizonState,
    y: &GpuTensor,
) -> rdna_compute::HipResult<()> {
    let use_shared =
        w.awq_scale.is_none() && hipfire_dispatch::types::dtype_needs_rotation(w.gpu_dtype);
    if use_shared {
        weight_gemv_prerotated(gpu, w, &state.normed, Some(&state.normed_rot), y)
    } else {
        weight_gemv(gpu, w, &state.normed, y)
    }
}

/// Full forward (prepare + body). Used by the non-PM4 path and during
/// retained-replay warmup.
fn forward_only(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    crate::lowered::run_decode_body_dispatch(cfg, weights, state, gpu, position)
}

/// Decode step returning full logits. Used for prefill (sequential
/// per-token decode). PM4 retained-replay is not routed here — prefill
/// has growing `seq_len` which changes attention kernel variants,
/// invalidating a captured PM4 packet.
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

/// Decode step with GPU-side sampling. Routes to PM4 retained-replay
/// when the Redline controller is enabled. Returns `(token_id, new_rng_state)`.
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
    blocked: &[u32],
) -> Result<(u32, u32), String> {
    if state.retained_state_poisoned {
        return Err("k2_horizon: retained state poisoned until reset".to_string());
    }
    if gpu.replay.is_enabled() {
        gpu.replay.set_forward_eligible(true);
        return decode_step_sampled_with_retained_replay(
            cfg, weights, state, gpu, token_id, position, temp, top_p, rng_state, blocked,
        );
    }
    forward_only(cfg, weights, state, gpu, token_id, position)?;
    sample_from_logits(cfg, state, gpu, temp, top_p, rng_state, blocked)
}

/// Sample from on-GPU logits. `temp <= 1e-6` → argmax, else top-p.
/// `blocked` token ids are forced to -INF before sampling (the
/// `sampler::sample` `blocked_tokens` mechanism) so the model cannot
/// re-open a think block once the think cap has latched.
fn sample_from_logits(
    cfg: &K2HorizonConfig,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    temp: f32,
    top_p: f32,
    rng_state: u32,
    blocked: &[u32],
) -> Result<(u32, u32), String> {
    // Unconditional -INF writes for blocked tokens (one 4-byte H2D each),
    // matching hipfire_runtime::sampler::sample. Runs outside the retained
    // capture window (sample_from_logits is called post-replay/post-capture),
    // so it never invalidates a captured PM4 packet.
    if !blocked.is_empty() {
        let neg_inf: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        for &tok in blocked {
            if (tok as usize) < cfg.vocab_size {
                let _ = gpu
                    .hip
                    .memcpy_htod_offset(&state.logits.buf, (tok as usize) * 4, &neg_inf);
            }
        }
    }
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

/// PM4 retained-replay lifecycle for GPU-sampled decode. Follows the
/// lfm2moe pattern: warmup → capture → replay.
///
/// 1. **Warmup** (first decode step): run `forward_only` normally, set
///    `retained_warmed_up = true`. The Redline controller observes the
///    kernel launches and arms itself.
/// 2. **Capture** (second decode step): stage inputs outside capture,
///    then `begin_auto_capture_if_armed` + `run_decode_body` inside the
///    capture window. `finish_capture` + `prepare_pm4_prefix` builds the
///    replay packet.
/// 3. **Replay** (subsequent decode steps): stage inputs, then
///    `replay_pm4` / `replay_linear_aql` replays the captured packet.
///    The position is dynamic via `pos_buf`; all device pointers are
fn decode_step_sampled_with_retained_replay(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    temp: f32,
    top_p: f32,
    rng_state: u32,
    blocked: &[u32],
) -> Result<(u32, u32), String> {
    if state.retained_state_poisoned {
        return Err("k2_horizon: retained state poisoned until reset".to_string());
    }

    // 1. Warmup: run the full forward normally so the Redline controller
    //    can observe and arm itself.
    if !state.retained_warmed_up {
        forward_only(cfg, weights, state, gpu, token_id, position)?;
        state.retained_warmed_up = true;
        return sample_from_logits(cfg, state, gpu, temp, top_p, rng_state, blocked);
    }

    // 2. Stage per-token inputs outside the capture window.
    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)?;

    // 3. Try PM4/AQL replay first (if the controller has a prepared route).
    if !gpu.replay.should_route_aql() && !gpu.replay.should_route_pm4() {
        let _ = gpu.replay.begin_auto_capture_if_armed();
        if gpu.replay.state() == rdna_compute::replay::ReplayState::Armed {
            let _ = gpu.replay.begin_capture();
        }
    }

    if gpu.replay.should_route_aql() || gpu.replay.should_route_pm4() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("k2_horizon: retained adapter sync: {e:?}"))?;
        let replay_result = if gpu.replay.should_route_aql() {
            unsafe { gpu.replay.replay_linear_aql(position as usize) }.map(|_| ())
        } else {
            unsafe { gpu.replay.replay_pm4(position as usize) }.map(|_| ())
        };
        match replay_result {
            Ok(()) => {
                return sample_from_logits(cfg, state, gpu, temp, top_p, rng_state, blocked);
            }
            Err(reason) => {
                let msg = format!("K2-Horizon retained replay failed: {reason}");
                gpu.replay.poison(msg);
                state.retained_state_poisoned = true;
                return Err(reason);
            }
        }
    }

    // 4. Capture path: run the decode body inside the capture window. Route
    //    through run_decode_body_dispatch (not run_decode_body directly) so
    //    the captured tape matches whatever the warmup step ran — with
    //    HIPFIRE_FORWARD_LOWERED=1 the warmup executes the lowered sequence
    //    and a hand-path capture would record a different kernel stream.
    crate::lowered::run_decode_body_dispatch(cfg, weights, state, gpu, position)?;

    // 5. Finalize capture and prepare the replay packet.
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("k2_horizon: retained capture sync: {e:?}"))?;
        let capture = gpu
            .replay
            .finish_capture()
            .map_err(|e| format!("k2_horizon: finish capture: {e}"))?;
        let launches = gpu.replay.recorded_launches().len();
        let prepare = if gpu.replay.uses_pm4_transport() {
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        };
        match prepare {
            Ok(()) => {
                eprintln!(
                    "[K2-Horizon redline] retained route ready: capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                );
            }
            Err(reason) => {
                gpu.replay.poison(format!(
                    "K2-Horizon Redline prepare after warmup failed: {reason}"
                ));
                eprintln!("[K2-Horizon redline] falling back to HIP: {reason}");
            }
        }
    } else if gpu.replay.state() == rdna_compute::replay::ReplayState::RecordingWarmup {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("k2_horizon: retained capture sync: {e:?}"))?;
        if let Ok(capture) = gpu.replay.finish_capture() {
            let launches = gpu.replay.recorded_launches().len();
            let prepare = if gpu.replay.uses_pm4_transport() {
                gpu.replay
                    .prepare_pm4_prefix(gpu.device_id as usize, launches)
                    .map(|_| ())
            } else {
                if launches >= 2 {
                    gpu.replay
                        .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                        .map(|_| ())
                } else {
                    Err("no captured launch sequence".to_owned())
                }
            };
            match prepare {
                Ok(()) => eprintln!(
                    "[K2-Horizon redline] retained route ready (manual): capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                ),
                Err(reason) => {
                    gpu.replay
                        .poison(format!("K2-Horizon Redline manual prepare failed: {reason}"));
                    eprintln!("[K2-Horizon redline] falling back to HIP: {reason}");
                }
            }
        }
    }

    // 6. Sample from the on-GPU logits.
    sample_from_logits(cfg, state, gpu, temp, top_p, rng_state, blocked)
}

pub(crate) fn attend(
    cfg: &K2HorizonConfig,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    seq_len: usize,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
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
        q,
        k,
        v,
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
    // normed = grouped_rmsnorm(h, attn_norm); normed_rot = FWHT(normed)
    // rotated ONCE, reused by wq/wk/wv/attn_gate below.
    norm_and_rotate(cfg, &layer.attn_norm, state, gpu, l)?;

    // q/k/v/gate: one fused GEMV when the packed [wq‖wk‖wv‖gate] weight is
    // present (4 launches → 1); per-projection GEMVs otherwise. Views into
    // attn_fused_out: q@[0..q_dim), k@[q_dim..q_dim+kv_dim),
    // v@[q_dim+kv_dim..q_dim+2kv_dim), gate@[q_dim+2kv_dim..2q_dim+2kv_dim).
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let (q_t, k_t, v_t, gate_t);
    if let Some((fused, _owner)) = &layer.attn_fused {
        gemv_normed(gpu, fused, state, &state.attn_fused_out)
            .map_err(|e| format!("k2_horizon L{l}: fused qkv+gate: {e}"))?;
        q_t = state.attn_fused_out.sub_offset(0, q_dim);
        k_t = state.attn_fused_out.sub_offset(q_dim, kv_dim);
        v_t = state.attn_fused_out.sub_offset(q_dim + kv_dim, kv_dim);
        gate_t = state.attn_fused_out.sub_offset(q_dim + 2 * kv_dim, q_dim);
    } else {
        gemv_normed(gpu, &layer.wq, state, &state.fa_q)
            .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
        gemv_normed(gpu, &layer.wk, state, &state.fa_k)
            .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;
        gemv_normed(gpu, &layer.wv, state, &state.fa_v)
            .map_err(|e| format!("k2_horizon L{l}: v_proj: {e}"))?;
        gemv_normed(gpu, &layer.attn_gate, state, &state.attn_gate_out)
            .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
        q_t = state.fa_q.sub_offset(0, q_dim);
        k_t = state.fa_k.sub_offset(0, kv_dim);
        v_t = state.fa_v.sub_offset(0, kv_dim);
        gate_t = state.attn_gate_out.sub_offset(0, q_dim);
    }

    // RoPE on Q and K (full rotary, rope_head_dim == head_dim)
    gpu.rope_f32(
        &q_t,
        &k_t,
        &state.pos_buf,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("k2_horizon L{l}: rope: {e:?}"))?;

    // KV write + attention.
    let seq_len = position as usize + 1;
    attend(cfg, state, gpu, l, seq_len, &q_t, &k_t, &v_t)?;

    // softplus post-attention gate: attn_out *= softplus_beta(gate_proj(normed))
    //   Fused: replaces scale→softplus→scale→mul (4 launches) with 1 launch.
    gpu.softplus_gate_f32(&gate_t, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;

    // h += o_proj(attn_out) — o_proj reads fa_attn_out (q_dim), rotated into
    // proj_rot; PM4-safe residual GEMV folds the add_inplace on gfx1100.
    gemv_residual_prerotated(gpu, &layer.wo, &state.fa_attn_out, state, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;

    // FFN: normed = grouped_rmsnorm(h, ffn_norm) + shared rotation.
    norm_and_rotate(cfg, &layer.ffn_norm, state, gpu, l)?;

    // dense SwiGLU: down(silu(gate(normed)) * up(normed)). Fused
    // [w_gate‖w_up] GEMV → ffn_fused_out views when packed (2 launches → 1).
    let dense_inter = cfg.intermediate_size;
    let (dg, du);
    if let Some((fused, _owner)) = &layer.ffn_fused {
        gemv_normed(gpu, fused, state, &state.ffn_fused_out)
            .map_err(|e| format!("k2_horizon L{l}: fused gate_up: {e}"))?;
        dg = state.ffn_fused_out.sub_offset(0, dense_inter);
        du = state.ffn_fused_out.sub_offset(dense_inter, dense_inter);
    } else {
        gemv_normed(gpu, &layer.w_gate, state, &state.dense_gate)
            .map_err(|e| format!("k2_horizon L{l}: dense gate: {e}"))?;
        gemv_normed(gpu, &layer.w_up, state, &state.dense_up)
            .map_err(|e| format!("k2_horizon L{l}: dense up: {e}"))?;
        dg = state.dense_gate.sub_offset(0, dense_inter);
        du = state.dense_up.sub_offset(0, dense_inter);
    }
    gpu.silu_mul_f32(&dg, &du, &state.dense_act)
        .map_err(|e| format!("k2_horizon L{l}: dense silu_mul: {e:?}"))?;
    // w_down reads dense_act (not normed) — rotate into proj_rot scratch;
    // PM4-safe residual GEMV folds the add_inplace on gfx1100.
    gemv_residual_prerotated(gpu, &layer.w_down, &state.dense_act, state, &state.h)
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
    let attn = &layer.attn;
    let ffn = &layer.ffn;

    // ── Attention branch ──────────────────────────────────────────────

    // normed = grouped_rmsnorm(h, attn_norm); normed_rot = FWHT(normed)
    // rotated ONCE, reused by wq/wk/v_router/attn_gate below.
    norm_and_rotate(cfg, &layer.attn_norm, state, gpu, l)?;

    // q/k/v_router/gate: one fused GEMV when the packed
    // [wq‖wk‖v_router‖gate] weight is present (4 launches → 1). Views into
    // attn_fused_out: q@[0..q_dim), k@[q_dim..q_dim+kv_dim),
    // router@[q_dim+kv_dim..q_dim+kv_dim+mova_n_exp),
    // gate@[q_dim+kv_dim+mova_n_exp..2q_dim+kv_dim+mova_n_exp).
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let mova_n = cfg.mova_num_experts;
    let (q_t, k_t, gate_t);
    if let Some((fused, _owner)) = &attn.attn_fused {
        gemv_normed(gpu, fused, state, &state.attn_fused_out)
            .map_err(|e| format!("k2_horizon L{l}: fused qk+router+gate: {e}"))?;
        q_t = state.attn_fused_out.sub_offset(0, q_dim);
        k_t = state.attn_fused_out.sub_offset(q_dim, kv_dim);
        gate_t = state
            .attn_fused_out
            .sub_offset(q_dim + kv_dim + mova_n, q_dim);
        // v_router logits already sit in the fused output — the routing
        // helper reads them from attn_fused_out when attn_fused is set.
        forward_mova_value_routing(cfg, attn, state, gpu, l)?;
    } else {
        gemv_normed(gpu, &attn.wq, state, &state.fa_q)
            .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
        gemv_normed(gpu, &attn.wk, state, &state.fa_k)
            .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;
        q_t = state.fa_q.sub_offset(0, q_dim);
        k_t = state.fa_k.sub_offset(0, kv_dim);
        gate_t = state.attn_gate_out.sub_offset(0, q_dim);
        // MoVA: v = combine_routed_experts(normed) — router GEMV inside.
        forward_mova_value_routing(cfg, attn, state, gpu, l)?;
        gemv_normed(gpu, &attn.attn_gate, state, &state.attn_gate_out)
            .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    }

    // RoPE on Q and K
    gpu.rope_f32(
        &q_t,
        &k_t,
        &state.pos_buf,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
    )
    .map_err(|e| format!("k2_horizon L{l}: rope: {e:?}"))?;

    // KV write + attention. MoVA v comes from the routed experts (fa_v).
    let seq_len = position as usize + 1;
    let v_t = state.fa_v.sub_offset(0, kv_dim);
    attend(cfg, state, gpu, l, seq_len, &q_t, &k_t, &v_t)?;
    //   Fused: replaces scale→softplus→scale→mul (4 launches) with 1 launch.
    gpu.softplus_gate_f32(&gate_t, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;

    // h += o_proj(attn_out) — PM4-safe residual GEMV folds the add_inplace.
    gemv_residual_prerotated(gpu, &attn.wo, &state.fa_attn_out, state, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;

    // ── FFN branch: sigmoid-routed MoE ────────────────────────────────

    // normed = grouped_rmsnorm(h, ffn_norm) + shared rotation.
    norm_and_rotate(cfg, &layer.ffn_norm, state, gpu, l)?;

    forward_sigmoid_moe_ffn(cfg, ffn, state, gpu, l)?;

    Ok(())
}

// ─── MoVA value-expert routing (fully GPU-side, PM4-capturable) ─────────

/// MoVA attention value routing — fully GPU-side, PM4-capturable:
/// 1. router GEMV → v_router_logits [64] (skipped when attn_fused — the
///    fused [wq‖wk‖v_router‖gate] GEMV already wrote them into
///    attn_fused_out)
/// 2. sigmoid(router_logits) in-place
/// 3. deepseek4_moe_topk_bias_aware_f32 → v_topk_indices + v_topk_weights (GPU)
/// 4. replicate_batched_f32(normed_rot) → v_rot_batch (shared rotate, PM4-safe)
/// 5. gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded → v_expanded [k×kv_dim]
/// 6. silu(v_expanded) in-place
/// 7. moe_down_combine_k8_batched → fa_v = Σ weight[k] · v_expanded[k]
///
/// Uses V2 indexed MoE GEMV kernels that decode fp16 per-128 headers
/// (MQ4G256V2 / qt=44), replacing the V1 kernels that read f32 per-256
/// headers. No D2H sync — fully PM4-capturable.
pub(crate) fn forward_mova_value_routing(
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

    // 1. router GEMV → v_router_logits [64] — reads normed via shared
    //    normed_rot. When the fused [wq‖wk‖v_router‖gate] weight is loaded,
    //    the logits already sit in attn_fused_out — skip the GEMV and view
    //    them in place.
    let router_logits;
    if attn.attn_fused.is_some() {
        let q_dim = cfg.n_heads * cfg.head_dim;
        router_logits = state
            .attn_fused_out
            .sub_offset(q_dim + kv_dim, cfg.mova_num_experts);
    } else {
        gemv_normed(gpu, &attn.v_router, state, &state.v_router_logits)
            .map_err(|e| format!("k2_horizon L{l}: v_router: {e}"))?;
        router_logits = state.v_router_logits.sub_offset(0, cfg.mova_num_experts);
    }

    // 2-3. Sigmoid + GPU top-K fused: the kernel applies sigmoid to the raw
    //    logits in its load phase, then does bias-aware top-K (bias added to
    //    sigmoid scores for selection only; weights use raw sigmoid scores —
    //    matches reference calc_router_weights). One launch, no D2H.
    let bias = attn
        .v_router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon L{l}: v_router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_sigmoid_f32(
        &router_logits,
        bias,
        &state.v_topk_indices,
        &state.v_topk_weights,
        cfg.mova_num_experts as i32,
        mova_top_k as i32,
        scaling,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_router topk: {e:?}"))?;
    // 4. Replicate the rotated normed input across k_top slices for the
    //    indexed kernel's [N × K_TOP × K] rot_batch layout. All expert
    //    weights load with awq_scale=None (wt_from_raw / load_packed_experts
    //    never set it), so the shared normed_rot is always the source.
    //    replicate_batched_f32 (not memcpy_dtod_at_auto) so the copy is a
    //    recorded kernel launch — PM4-capturable.
    gpu.replicate_batched_f32(&state.normed_rot, &state.v_rot_batch, hidden, mova_top_k, 1)
        .map_err(|e| format!("k2_horizon L{l}: v_rot_batch replicate: {e:?}"))?;

    // 5. V2 indexed MoE GEMV: all mova_top_k v_experts in one kernel launch.
    //    Writes [mova_top_k × kv_dim] to v_expanded.
    match attn.v_experts[0].gpu_dtype {
        DType::MQ4G256V2 => gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &attn.v_expert_ptrs,
            &state.v_topk_indices,
            &state.v_rot_batch,
            &state.v_expanded,
            kv_dim,
            hidden,
            mova_top_k,
            1,
        ),
        DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_down_indexed_batched_expanded(
            &attn.v_expert_ptrs,
            &state.v_topk_indices,
            &state.v_rot_batch,
            &state.v_expanded,
            kv_dim,
            hidden,
            mova_top_k,
            1,
        ),
        other => return Err(format!(
            "k2_horizon L{l}: v_expert dtype {other:?} — needs MQ4G256V2 or MQ3G256Lloyd"
        )),
    }
    .map_err(|e| format!("k2_horizon L{l}: v_expert indexed gemv: {e:?}"))?;

    // 6-7. Fused SiLU + weighted combine with overwrite semantics:
    //    fa_v = Σ_k w[k] · silu(v_expanded[k]). One launch replaces the
    //    silu_f32 + zero_f32 + combine sequence.
    gpu.moe_down_combine_silu_overwrite_k8_batched(
        &state.v_expanded,
        &state.v_topk_weights,
        &state.fa_v,
        kv_dim,
        mova_top_k,
        1,
    )
    .map_err(|e| format!("k2_horizon L{l}: v_expert combine: {e:?}"))?;

    Ok(())
}

// ─── Sigmoid-routed MoE FFN ─────────────────────────────────────────────

/// Sigmoid-routed MoE FFN with fully GPU-side routing:
/// 1. router GEMV → moe_router_logits [num_experts]
/// 2. sigmoid(router_logits) in-place
/// 3. deepseek4_moe_topk_bias_aware_f32 → topk_indices + topk_weights (GPU)
/// 4. shared normed_rot feeds the indexed gate_up GEMV (no per-expert rotate)
/// 5. gemv_mq4g256v2_moe_gate_up_k8_indexed_batched → gate_batch + up_batch
/// 6. fused_silu_mul_rotate_mq_batched → rot_batch (silu(gate)*up + FWHT)
/// 7. gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded → down_expanded
/// 8. moe_down_combine_k8_batched → h += Σ weight[k] * down_expanded[k]
/// 9. shared expert (SwiGLU GEMV) → add to h
pub(crate) fn forward_sigmoid_moe_ffn(
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

    // 1. router GEMV → moe_router_logits [100] — reads normed via shared
    //    normed_rot. When the fused [router‖shared_gate‖shared_up] weight
    //    is loaded, one GEMV produces router logits + shared gate + shared
    //    up into ffn_fused_out (3 launches → 1); views are sliced below.
    let (router_logits, shared_gate_t, shared_up_t);
    if let Some((fused, _owner)) = &ffn.ffn_fused {
        gemv_normed(gpu, fused, state, &state.ffn_fused_out)
            .map_err(|e| format!("k2_horizon L{l}: fused router+shared: {e}"))?;
        router_logits = state.ffn_fused_out.sub_offset(0, cfg.num_experts);
        shared_gate_t = state.ffn_fused_out.sub_offset(cfg.num_experts, moe_inter);
        shared_up_t = state
            .ffn_fused_out
            .sub_offset(cfg.num_experts + moe_inter, moe_inter);
    } else {
        gemv_normed(gpu, &ffn.router, state, &state.moe_router_logits)
            .map_err(|e| format!("k2_horizon L{l}: moe router: {e}"))?;
        router_logits = state.moe_router_logits.sub_offset(0, cfg.num_experts);
        shared_gate_t = state.shared_gate.sub_offset(0, moe_inter);
        shared_up_t = state.shared_up.sub_offset(0, moe_inter);
    }

    // 2-3. Sigmoid + GPU top-K fused: sigmoid applied inside the topk
    //    kernel's load phase. Single launch, no D2H.
    let bias = ffn
        .router_bias
        .as_ref()
        .ok_or_else(|| format!("k2_horizon L{l}: router_bias missing"))?;
    gpu.deepseek4_moe_topk_bias_aware_sigmoid_f32(
        &router_logits,
        bias,
        &state.moe_topk_indices,
        &state.moe_topk_weights,
        cfg.num_experts as i32,
        top_k as i32,
        scaling,
    )
    .map_err(|e| format!("k2_horizon L{l}: moe topk: {e:?}"))?;

    // 4. Shared normed_rot feeds the indexed gate_up GEMV — all expert
    //    weights load with awq_scale=None (wt_from_raw / load_packed_experts
    //    never set it), so the fixed FWHT rotation always applies.

    // 5. V2 indexed MoE gate_up GEMV: all top_k experts in one kernel launch.
    match ffn.experts[0].gate_up.gpu_dtype {
        DType::MQ4G256V2 => gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
            &ffn.expert_gate_up_ptrs,
            &state.moe_topk_indices,
            &state.normed_rot,
            &state.gate_batch,
            &state.up_batch,
            2 * moe_inter,
            hidden,
            top_k,
            1,
        ),
        DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_gate_up_indexed_batched(
            &ffn.expert_gate_up_ptrs,
            &state.moe_topk_indices,
            &state.normed_rot,
            &state.gate_batch,
            &state.up_batch,
            2 * moe_inter,
            hidden,
            top_k,
            1,
        ),
        other => return Err(format!(
            "k2_horizon L{l}: expert gate_up dtype {other:?} — needs MQ4G256V2 or MQ3G256Lloyd"
        )),
    }
    .map_err(|e| format!("k2_horizon L{l}: gate_up indexed gemv: {e:?}"))?;

    // 6. Fused silu_mul + FWHT rotate: rot_batch = FWHT(silu(gate) * up)
    //    Processes all top_k expert streams in one launch.
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        &ffn.experts[0].down,
        &state.gate_batch,
        &state.up_batch,
        &state.rot_batch,
        moe_inter,
        top_k,
    )
    .map_err(|e| format!("k2_horizon L{l}: fused silu_mul rotate: {e:?}"))?;

    // 7. V2 indexed down GEMV: all top_k experts in one kernel launch.
    //    Writes [top_k × hidden] to down_expanded.
    match ffn.experts[0].down.gpu_dtype {
        DType::MQ4G256V2 => gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &ffn.expert_down_ptrs,
            &state.moe_topk_indices,
            &state.rot_batch,
            &state.down_expanded,
            hidden,
            moe_inter,
            top_k,
            1,
        ),
        DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_down_indexed_batched_expanded(
            &ffn.expert_down_ptrs,
            &state.moe_topk_indices,
            &state.rot_batch,
            &state.down_expanded,
            hidden,
            moe_inter,
            top_k,
            1,
        ),
        other => return Err(format!(
            "k2_horizon L{l}: expert down dtype {other:?} — needs MQ4G256V2 or MQ3G256Lloyd"
        )),
    }
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

    // 9. Shared expert (always-on SwiGLU). With ffn_fused, gate/up already
    //    sit in ffn_fused_out views; otherwise run the two GEMVs. down reads
    //    shared_act, rotated into proj_rot.
    if ffn.ffn_fused.is_none() {
        gemv_normed(gpu, &ffn.shared.gate, state, &state.shared_gate)
            .map_err(|e| format!("k2_horizon L{l}: shared gate: {e}"))?;
        gemv_normed(gpu, &ffn.shared.up, state, &state.shared_up)
            .map_err(|e| format!("k2_horizon L{l}: shared up: {e}"))?;
    }
    gpu.silu_mul_f32(&shared_gate_t, &shared_up_t, &state.shared_act)
        .map_err(|e| format!("k2_horizon L{l}: shared silu_mul: {e:?}"))?;
    // Shared down: PM4-safe single-row residual GEMV (no private scratch —
    // the dual-row gemv_mq4g256v2_residual is PM4-rejected). Folds
    // rotate + gemv + add_inplace into rotate + residual-gemv.
    gemv_residual_prerotated(gpu, &ffn.shared.down, &state.shared_act, state, &state.h)
        .map_err(|e| format!("k2_horizon L{l}: shared down: {e}"))?;

    Ok(())
}
