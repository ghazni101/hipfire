// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon lowered forward (#397 Ship 6) — LayerProgram / SuperOp substrate.
//!
//! Each layer is lowered once into a `LayerProgram = Vec<SuperOp>`. At decode
//! time `run_layer_program` loops over the pre-resolved super-ops and calls
//! the arch-provided `ForwardBindings` impl, which dispatches to the existing
//! block functions. This is a dispatch/fusion-routing optimization, orthogonal
//! to PM4/AQL retained-replay graph capture.
//!
//! Layer shapes:
//! - **Dense** (layers 0–2): `[Attend, Proj, ResidualGemv]`
//! - **MoE** (layers 3–47): `[Escape(K2HorizonMovaAttn), Moe]`
//!
//! Behind `HIPFIRE_FORWARD_LOWERED` (default off). The hand path in
//! `forward.rs` remains the live path; this is oracle-validated before flip.

use crate::config::K2HorizonConfig;
use crate::forward::{
    attend, forward_mova_value_routing, forward_sigmoid_moe_ffn, gemv_normed, norm_and_rotate,
    K2HorizonState,
};
use crate::weights::{DenseLayerWeights, K2HorizonWeights, MovaLayerWeights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{weight_gemv, weight_gemv_prerotated};
use rdna_compute::Gpu;

// ── Opcodes (encoded in OpBinding.weights[0]) ───────────────────────────

mod k2_op {
    /// Dense FFN gate+up projection (Proj super-op).
    pub const DENSE_GATE_UP: u32 = 0;
    /// Dense FFN down projection + residual (ResidualGemv super-op).
    pub const DENSE_DOWN: u32 = 1;
}

// ── Layer variants ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K2Variant {
    Dense,
    Moe,
}

#[inline]
fn k2_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one K2-Horizon decoder layer to a coarse super-op LayerProgram.
/// Pure (no GpuTensor) → unit-testable.
fn k2_lower_variant(v: K2Variant) -> superop::LayerProgram {
    use k2_op::{DENSE_DOWN, DENSE_GATE_UP};
    use SuperOpKind::{Attend, Escape, Moe, Proj, ResidualGemv};
    match v {
        K2Variant::Dense => vec![
            k2_superop(Attend, 0),
            k2_superop(Proj, DENSE_GATE_UP),
            k2_superop(ResidualGemv, DENSE_DOWN),
        ],
        K2Variant::Moe => vec![
            k2_superop(Escape(superop::EscapeKind::K2HorizonMovaAttn), 0),
            k2_superop(Moe, 0),
        ],
    }
}

// ── Block functions (split from forward.rs for lowered dispatch) ────────

/// Dense attention block: norm + QKV + RoPE + attend + softplus_gate + o_proj.
/// Mirrors `forward_dense_layer`'s attention half — uses the shared
/// `normed_rot` (norm_and_rotate) and `proj_rot` (o_proj) buffers so the
/// lowered path is numerically identical to the hand path.
fn dense_attention_block(
    cfg: &K2HorizonConfig,
    layer: &DenseLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    position: u32,
) -> Result<(), String> {
    // normed = grouped_rmsnorm(h, attn_norm); normed_rot = FWHT(normed) once.
    norm_and_rotate(cfg, &layer.attn_norm, state, gpu, l)?;

    gemv_normed(gpu, &layer.wq, state, &state.fa_q)
        .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
    gemv_normed(gpu, &layer.wk, state, &state.fa_k)
        .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;
    gemv_normed(gpu, &layer.wv, state, &state.fa_v)
        .map_err(|e| format!("k2_horizon L{l}: v_proj: {e}"))?;

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

    let seq_len = position as usize + 1;
    attend(cfg, state, gpu, l, seq_len)?;

    gemv_normed(gpu, &layer.attn_gate, state, &state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    gpu.softplus_gate_f32(&state.attn_gate_out, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;

    // o_proj reads fa_attn_out (q_dim), rotated into proj_rot.
    gpu.rotate_x_mq(&state.fa_attn_out, &state.proj_rot, layer.wo.k)
        .map_err(|e| format!("k2_horizon L{l}: o rotate: {e:?}"))?;
    weight_gemv_prerotated(
        gpu,
        &layer.wo,
        &state.fa_attn_out,
        Some(&state.proj_rot),
        &state.scratch_h,
    )
    .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;
    gpu.add_inplace_f32(&state.h, &state.scratch_h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj add: {e:?}"))?;

    Ok(())
}

/// Dense FFN gate+up block: norm + gate_proj + up_proj. Mirrors the FFN half
/// of `forward_dense_layer` — shared `normed_rot` via norm_and_rotate.
fn dense_gate_up_block(
    cfg: &K2HorizonConfig,
    layer: &DenseLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    norm_and_rotate(cfg, &layer.ffn_norm, state, gpu, l)?;

    gemv_normed(gpu, &layer.w_gate, state, &state.dense_gate)
        .map_err(|e| format!("k2_horizon L{l}: dense gate: {e}"))?;
    gemv_normed(gpu, &layer.w_up, state, &state.dense_up)
        .map_err(|e| format!("k2_horizon L{l}: dense up: {e}"))?;

    Ok(())
}

/// Dense FFN down block: silu_mul + down_proj + residual add. w_down reads
/// dense_act (not normed) — rotated into proj_rot, matching the hand path.
fn dense_down_block(
    cfg: &K2HorizonConfig,
    layer: &DenseLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("k2_horizon L{l}: dense silu_mul: {e:?}"))?;
    gpu.rotate_x_mq(&state.dense_act, &state.proj_rot, cfg.intermediate_size)
        .map_err(|e| format!("k2_horizon L{l}: down rotate: {e:?}"))?;
    weight_gemv_prerotated(
        gpu,
        &layer.w_down,
        &state.dense_act,
        Some(&state.proj_rot),
        &state.scratch_h,
    )
    .map_err(|e| format!("k2_horizon L{l}: dense down: {e}"))?;
    gpu.add_inplace_f32(&state.h, &state.scratch_h)
        .map_err(|e| format!("k2_horizon L{l}: dense down add: {e:?}"))?;
    Ok(())
}

/// MoE attention block: norm + Q/K + MoVA routing + RoPE + attend +
/// softplus_gate + o_proj. (The irregular Escape op.) Mirrors
/// `forward_moe_layer`'s attention half — populates `normed_rot` via
/// norm_and_rotate so forward_mova_value_routing reads a fresh rotation.
fn moe_attention_block(
    cfg: &K2HorizonConfig,
    layer: &MovaLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
    position: u32,
) -> Result<(), String> {
    let attn = &layer.attn;

    // normed = grouped_rmsnorm(h, attn_norm); normed_rot = FWHT(normed) once.
    norm_and_rotate(cfg, &layer.attn_norm, state, gpu, l)?;

    gemv_normed(gpu, &attn.wq, state, &state.fa_q)
        .map_err(|e| format!("k2_horizon L{l}: q_proj: {e}"))?;
    gemv_normed(gpu, &attn.wk, state, &state.fa_k)
        .map_err(|e| format!("k2_horizon L{l}: k_proj: {e}"))?;

    forward_mova_value_routing(cfg, attn, state, gpu, l)?;

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

    let seq_len = position as usize + 1;
    attend(cfg, state, gpu, l, seq_len)?;

    gemv_normed(gpu, &attn.attn_gate, state, &state.attn_gate_out)
        .map_err(|e| format!("k2_horizon L{l}: attn gate: {e}"))?;
    gpu.softplus_gate_f32(&state.attn_gate_out, &state.fa_attn_out)
        .map_err(|e| format!("k2_horizon L{l}: softplus gate: {e:?}"))?;

    // o_proj reads fa_attn_out (q_dim), rotated into proj_rot.
    gpu.rotate_x_mq(&state.fa_attn_out, &state.proj_rot, attn.wo.k)
        .map_err(|e| format!("k2_horizon L{l}: o rotate: {e:?}"))?;
    weight_gemv_prerotated(
        gpu,
        &attn.wo,
        &state.fa_attn_out,
        Some(&state.proj_rot),
        &state.scratch_h,
    )
    .map_err(|e| format!("k2_horizon L{l}: o_proj: {e}"))?;
    gpu.add_inplace_f32(&state.h, &state.scratch_h)
        .map_err(|e| format!("k2_horizon L{l}: o_proj add: {e:?}"))?;

    Ok(())
}

/// MoE FFN block: norm + sigmoid-routed MoE FFN + shared expert. Populates
/// `normed_rot` via norm_and_rotate so forward_sigmoid_moe_ffn's router and
/// expert GEMVs read a fresh rotation (not a stale buffer).
fn moe_ffn_block(
    cfg: &K2HorizonConfig,
    layer: &MovaLayerWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    l: usize,
) -> Result<(), String> {
    norm_and_rotate(cfg, &layer.ffn_norm, state, gpu, l)?;

    forward_sigmoid_moe_ffn(cfg, &layer.ffn, state, gpu, l)?;

    Ok(())
}

// ── ForwardBindings impl ────────────────────────────────────────────────

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct K2HorizonBindings<'a> {
    cfg: &'a K2HorizonConfig,
    dense_layer: Option<&'a DenseLayerWeights>,
    moe_layer: Option<&'a MovaLayerWeights>,
    state: &'a mut K2HorizonState,
    l: usize,
    position: u32,
}

fn k2_forward_lowered_enabled() -> bool {
    use std::sync::LazyLock;
    // Default OFF until oracle-validated (module doc: "oracle-validated before
    // flip"). Opt in with HIPFIRE_FORWARD_LOWERED=1. Keeping it off also keeps
    // warmup and PM4 capture on the same kernel sequence — the capture path
    // calls run_decode_body (hand path) directly.
    static F: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            == Some("1")
    });
    *F
}

impl<'a> ForwardBindings for K2HorizonBindings<'a> {
    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match self.dense_layer {
            Some(layer) => {
                dense_attention_block(self.cfg, layer, self.state, gpu, self.l, self.position)
            }
            None => Err("run_attend on non-Dense layer".into()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, self.dense_layer) {
            (k2_op::DENSE_GATE_UP, Some(layer)) => {
                dense_gate_up_block(self.cfg, layer, self.state, gpu, self.l)
            }
            _ => Err(format!("run_proj bad opcode {code} / non-Dense layer")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, self.dense_layer) {
            (k2_op::DENSE_DOWN, Some(layer)) => {
                dense_down_block(self.cfg, layer, self.state, gpu, self.l)
            }
            _ => Err(format!(
                "run_residual_gemv bad opcode {code} / non-Dense layer"
            )),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match self.moe_layer {
            Some(layer) => moe_ffn_block(self.cfg, layer, self.state, gpu, self.l),
            None => Err("run_moe on non-MoE layer".into()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_escape(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        match kind {
            superop::EscapeKind::K2HorizonMovaAttn => match self.moe_layer {
                Some(layer) => {
                    moe_attention_block(self.cfg, layer, self.state, gpu, self.l, self.position)
                }
                None => Err("K2HorizonMovaAttn on non-MoE layer".into()),
            },
            _ => Err(format!("k2_horizon has no Escape super-op ({kind:?})")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "k2_horizon has no standalone Norm super-op".into(),
        ))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "k2_horizon has no Recurrent super-op".into(),
        ))
    }
    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("k2_horizon has no Conv super-op".into()))
    }
}

// ── Lowered decode path ─────────────────────────────────────────────────

/// Cached HIPFIRE_FORWARD_LOWERED toggle. Default OFF (opt in with "1") until
/// the lowered path is oracle-validated, per the module doc.

/// Lowered (#397 Ship 6) decode body: layers + final norm + lm_head.
/// Behaviorally equivalent to `run_decode_body` in `forward.rs`.
fn run_decode_body_lowered(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);

    for (l, layer) in weights.dense_layers.iter().enumerate() {
        let program = k2_lower_variant(K2Variant::Dense);
        let mut bind = K2HorizonBindings {
            cfg,
            dense_layer: Some(layer),
            moe_layer: None,
            state,
            l,
            position,
        };
        superop::run_layer_program(gpu, &ctx, &program, &mut bind)
            .map_err(|e| format!("k2_horizon L{l}: lowered run_layer_program: {e}"))?;
    }

    for (l, layer) in weights.moe_layers.iter().enumerate() {
        let global_l = weights.dense_layers.len() + l;
        let program = k2_lower_variant(K2Variant::Moe);
        let mut bind = K2HorizonBindings {
            cfg,
            dense_layer: None,
            moe_layer: Some(layer),
            state,
            l: global_l,
            position,
        };
        superop::run_layer_program(gpu, &ctx, &program, &mut bind)
            .map_err(|e| format!("k2_horizon L{global_l}: lowered run_layer_program: {e}"))?;
    }

    // Final norm + lm_head
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

/// Entry point: run the decode body via the lowered LayerProgram path
/// when `HIPFIRE_FORWARD_LOWERED=1`, else fall through to the hand path.
/// Called from `run_decode_body` in `forward.rs`.
pub fn run_decode_body_dispatch(
    cfg: &K2HorizonConfig,
    weights: &K2HorizonWeights,
    state: &mut K2HorizonState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    if k2_forward_lowered_enabled() {
        run_decode_body_lowered(cfg, weights, state, gpu, position)
    } else {
        // Fall through to the hand path.
        crate::forward::run_decode_body(cfg, weights, state, gpu, position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use superop::SuperOpKind;

    #[test]
    fn dense_layer_program_shape() {
        let p = k2_lower_variant(K2Variant::Dense);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].kind, SuperOpKind::Attend);
        assert_eq!(p[1].kind, SuperOpKind::Proj);
        assert_eq!(p[2].kind, SuperOpKind::ResidualGemv);
    }

    #[test]
    fn moe_layer_program_shape() {
        let p = k2_lower_variant(K2Variant::Moe);
        assert_eq!(p.len(), 2);
        assert_eq!(
            p[0].kind,
            SuperOpKind::Escape(superop::EscapeKind::K2HorizonMovaAttn)
        );
        assert_eq!(p[1].kind, SuperOpKind::Moe);
    }
}
