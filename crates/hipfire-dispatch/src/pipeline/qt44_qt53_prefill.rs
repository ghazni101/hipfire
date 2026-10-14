// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact QT44/QT53 top-10 grouped-prefill lowering.
//!
//! This module is deliberately narrow.  It is reached only after the sealed
//! MoE predicate has admitted the immutable grouped-kernel geometry and replicated route.
//! Generic MoE prefill continues to own the k=8 path; no QT53 format is
//! admitted by this module through a representative-dtype fallback.

use super::layer_ops::hip;
use crate::families::gemv::WeightRef;
use crate::families::moe::{MoePrefillParams, MoeRouteCapability, MoeTileExchangeParams};
use crate::types::DispatchError;
use rdna_compute::moe::SharedExpertActivation;
use rdna_compute::tensor_ops::{bf16_scaled_add_batched, Bf16ScaledAddBatched};
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn f32_view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

#[inline]
fn grouped_geometry(p: &MoePrefillParams<'_>) -> bool {
    let Some(policy) = p.route_policy else {
        return false;
    };
    p.batch_size > 0
        && crate::families::moe::grouped_route_geometry_supported(
            &policy,
            p.n_exp,
            p.k_top,
            p.gate_up_k,
            p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
        )
}

fn require_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if grouped_geometry(p) {
        Ok(())
    } else {
        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "grouped-prefill-shape",
            arch: "",
            quant: "declared route",
        })
    }
}

const F32_BYTES: usize = std::mem::size_of::<f32>();
const BF16_BYTES: usize = 2;
/// Bytes the BF16 combine's rank-order scratch holds per route slot
/// (`moe_combine_order_top10`: `tokens * 10 * 8`).
const ORDER_BYTES_PER_SLOT: usize = 8;

fn scratch_overflow() -> DispatchError {
    DispatchError::Hip("qt44/qt53 grouped prefill: scratch extent overflows".into())
}

fn scratch_mul(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_mul(b).ok_or_else(scratch_overflow)
}

/// Bytes of the six scratch buffers whose size depends on the route the call
/// actually takes.  The same record is the requirement a route derives
/// ([`ScratchRoute::requirement`]) and the capacity the call's tensors hold
/// ([`ScratchBytes::held`]), so validation, lowering and the tests compare one
/// type.  A buffer the route never touches requires 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScratchBytes {
    pub(crate) gate_batch: usize,
    pub(crate) up_batch: usize,
    pub(crate) rot_batch: usize,
    pub(crate) down_expanded: usize,
    pub(crate) y_gate_up_grouped: usize,
    pub(crate) y_down_grouped: usize,
}

impl ScratchBytes {
    fn fields(&self) -> [(&'static str, usize); 6] {
        [
            ("gate_batch", self.gate_batch),
            ("up_batch", self.up_batch),
            ("rot_batch", self.rot_batch),
            ("down_expanded", self.down_expanded),
            ("y_gate_up_grouped", self.y_gate_up_grouped),
            ("y_down_grouped", self.y_down_grouped),
        ]
    }

    /// Physical bytes each scratch tensor of `p` can hold: the smaller of its
    /// logical extent and its device buffer, so neither forged shape metadata
    /// nor a short allocation passes.
    pub(crate) fn held(p: &MoePrefillParams<'_>) -> Self {
        Self {
            gate_batch: tensor_bytes(p.gate_batch),
            up_batch: tensor_bytes(p.up_batch),
            rot_batch: tensor_bytes(p.rot_batch),
            down_expanded: tensor_bytes(p.down_expanded),
            y_gate_up_grouped: tensor_bytes(p.y_gate_up_grouped),
            y_down_grouped: tensor_bytes(p.y_down_grouped),
        }
    }

    /// `Err` naming the first buffer of `held` that is below this requirement.
    pub(crate) fn require_within(&self, held: &Self) -> Result<(), DispatchError> {
        for ((name, need), (_, have)) in self.fields().into_iter().zip(held.fields()) {
            if have < need {
                return Err(DispatchError::Hip(format!(
                    "qt44/qt53 grouped prefill scratch: {name} holds {have} bytes, \
                     the selected route needs {need}; refusing before launch"
                )));
            }
        }
        Ok(())
    }
}

fn tensor_bytes(tensor: &GpuTensor) -> usize {
    let logical = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .and_then(|elements| elements.checked_mul(tensor.dtype.size()))
        .unwrap_or(0);
    logical.min(tensor.buf.size())
}

/// F32 `[rows, down_m]` shared-down rows, which the unfused shared down
/// writes into `down_expanded` whenever the BF16 epilogue does not take it.
fn shared_down_bytes(rows: usize, down_m: usize) -> Result<usize, DispatchError> {
    scratch_mul(scratch_mul(rows, down_m)?, F32_BYTES)
}

/// Rank-order scratch of the BF16-row combine (`tokens * k_top * 8` bytes).
fn combine_order_bytes(rows: usize, k_top: usize) -> Result<usize, DispatchError> {
    scratch_mul(scratch_mul(rows, k_top)?, ORDER_BYTES_PER_SLOT)
}

/// Invocation geometry the byte requirements depend on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchDims {
    pub(crate) rows: usize,
    pub(crate) k_top: usize,
    pub(crate) mi: usize,
    pub(crate) down_m: usize,
    /// Padded grouped rows (path 2 only; 0 on path 1).
    pub(crate) grouped_rows: usize,
    /// The call carries a shared expert whose F32 down may land in
    /// `down_expanded` (no `shared_down_out`).
    pub(crate) shared_down: bool,
}

/// The kernel arms one invocation takes, from the same predicates the stage
/// lowering below branches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchRoute {
    pub(crate) path2: bool,
    pub(crate) sym_iu4: bool,
    pub(crate) gateup_bf16: bool,
    pub(crate) gateup_silu: bool,
    pub(crate) down_wmma: bool,
    pub(crate) indexed_down: bool,
}

impl ScratchRoute {
    pub(crate) fn of(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> Self {
        Self {
            path2: use_path2,
            sym_iu4: sym_iu4(gpu, p, use_path2),
            gateup_bf16: use_path2 && gateup_bf16(gpu, p),
            gateup_silu: use_path2 && gateup_silu(gpu, p),
            down_wmma: use_path2 && down_wmma(gpu, p),
            indexed_down: indexed_down(gpu, p, use_path2),
        }
    }

    /// Exact bytes each scratch buffer must hold for this route at `dims`:
    /// - symmetric IU4: BF16 `[G, mi]` SwiGLU rows, BF16 `[G, down_m]` down
    ///   rows, the combine's rank order;
    /// - path-2 F32 arms: F32 `[G, 2 mi]` / `[G, down_m]` (BF16 on the gfx11
    ///   WMMA arms), plus the unscattered activation unless gate/up stores
    ///   SwiGLU for the down's own unscatter;
    /// - indexed down (<= 8 rows) and path 1: the `[rows * k_top, ...]`
    ///   batch buffers and the expanded down rows;
    /// - every route that has a shared expert: its F32 `[rows, down_m]` down
    ///   in `down_expanded`, because the BF16 epilogue may decline at run time.
    pub(crate) fn requirement(&self, dims: &ScratchDims) -> Result<ScratchBytes, DispatchError> {
        let slots = scratch_mul(dims.rows, dims.k_top)?;
        let activation = scratch_mul(scratch_mul(slots, dims.mi)?, F32_BYTES)?;
        let expanded = scratch_mul(scratch_mul(slots, dims.down_m)?, F32_BYTES)?;
        let grouped = dims.grouped_rows;
        let mut need = ScratchBytes::default();
        if dims.shared_down {
            need.down_expanded = shared_down_bytes(dims.rows, dims.down_m)?;
        }
        if !self.path2 {
            need.gate_batch = activation;
            need.up_batch = activation;
            need.rot_batch = activation;
            need.down_expanded = need.down_expanded.max(expanded);
            return Ok(need);
        }
        let grouped_bytes = |width: usize, elem: usize| -> Result<usize, DispatchError> {
            scratch_mul(scratch_mul(grouped, width)?, elem)
        };
        let order = combine_order_bytes(dims.rows, dims.k_top)?;
        if self.sym_iu4 {
            need.y_gate_up_grouped = grouped_bytes(dims.mi, BF16_BYTES)?;
            need.y_down_grouped = grouped_bytes(dims.down_m, BF16_BYTES)?;
            need.down_expanded = need.down_expanded.max(order);
        } else {
            let gate_up_width = scratch_mul(2, dims.mi)?;
            need.y_gate_up_grouped = match (self.gateup_bf16, self.gateup_silu) {
                (true, true) => grouped_bytes(dims.mi, BF16_BYTES)?,
                (true, false) => grouped_bytes(gate_up_width, BF16_BYTES)?,
                (false, _) => grouped_bytes(gate_up_width, F32_BYTES)?,
            };
            if !self.gateup_silu {
                need.rot_batch = activation;
            }
            if self.down_wmma {
                need.y_down_grouped = grouped_bytes(dims.down_m, BF16_BYTES)?;
                need.down_expanded = need.down_expanded.max(order);
            } else if !self.indexed_down {
                need.y_down_grouped = grouped_bytes(dims.down_m, F32_BYTES)?;
            }
        }
        if self.indexed_down {
            need.rot_batch = activation;
            need.down_expanded = need.down_expanded.max(expanded);
        }
        Ok(need)
    }
}

/// The bytes this invocation's route needs from the call's scratch, for the
/// route [`ScratchRoute::of`] derives at `p`'s actual rows.
pub(crate) fn scratch_requirement(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<ScratchBytes, DispatchError> {
    require_geometry(p)?;
    ScratchRoute::of(gpu, p, use_path2).requirement(&ScratchDims {
        rows: p.batch_size,
        k_top: p.k_top,
        mi: p.mi,
        down_m: p.down_m,
        grouped_rows: if use_path2 { grouped_rows } else { 0 },
        shared_down: p.prelude.shared.is_some() && p.shared_down_out.is_none(),
    })
}

/// Preflight of one invocation: every scratch tensor holds the bytes the
/// selected route needs.  Run before the call's first launch by
/// `SealedMoeCall::validate_for_gpu`, and again by each stage that writes
/// scratch.
pub(crate) fn validate_scratch(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    scratch_requirement(gpu, p, use_path2, grouped_rows)?.require_within(&ScratchBytes::held(p))?;
    if p.shared_down_out.is_some() && p.prelude.shared.is_some() {
        require_shared_rows(p, shared_down_bytes(p.batch_size, p.down_m)?)?;
    }
    Ok(())
}

/// The tensor the unfused shared down writes (`shared_down_out`, else
/// `down_expanded`) holds at least `need` bytes.
fn require_shared_rows(p: &MoePrefillParams<'_>, need: usize) -> Result<(), DispatchError> {
    let (name, tensor) = match p.shared_down_out {
        Some(out) => ("shared_down_out", out),
        None => ("down_expanded", p.down_expanded),
    };
    let have = tensor_bytes(tensor);
    if have < need {
        return Err(DispatchError::Hip(format!(
            "qt44/qt53 grouped prefill scratch: {name} holds {have} bytes, \
             the unfused shared down needs {need}; refusing before launch"
        )));
    }
    Ok(())
}

/// Whether the routed down of this call leaves its `rows * k_top` expanded
/// rows in slot order in `down_expanded` for the canonical top-10 combine to
/// read: the decode-kernel down (path 1, or path 2's few-row indexed down) with
/// no expert-stage table. This is the layout an EP row gather fills.
pub(crate) fn writes_slot_ordered_rows(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    p.expert_stage_ptrs.is_none()
        && !sym_iu4(gpu, p, use_path2)
        && (!use_path2 || indexed_down(gpu, p, use_path2))
}

/// Prepare the routed expert activation basis.  The route's router/shared
/// projections are BF16 and consume the natural input, while routed QT44
/// gate/up consumes the FWHT basis.
pub(crate) fn input_basis(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if gateup_rotates_f16(gpu, p, use_path2) {
        return Ok(());
    }
    hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))
}

/// Whether the path-2 F16 WMMA gate/up rotates the activation straight to
/// F16 itself (the bytes the F32 rotation + conversion produce): only when no
/// shared projection reads the F32 basis `x_rot_batch`.
fn gateup_rotates_f16(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2
        && gateup_bf16(gpu, p)
        && p.prelude.shared.as_ref().is_none_or(|shared| {
            [&shared.weights.selector, &shared.weights.gate]
                .iter()
                .all(|w| w.dtype != DType::MQ4G256V2)
        })
}

fn batch_projection(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> Result<(), DispatchError> {
    match weight.dtype {
        // BF16 rows read the natural activation; this route materializes the
        // packed basis once per layer, so no per-projection rotation is owed.
        DType::BF16 => super::project_weight(gpu, weight, x, y, batch_size, None),
        DType::F32 => hip(gpu.gemm_f32_batched(weight.buf, x, y, weight.m, weight.k, batch_size)),
        DType::MQ4G256V2 => {
            hip(gpu.gemm_mq4g256v2(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        DType::MQ4G128V2 => {
            hip(gpu.gemm_mq4g128v2_batched(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        // Qwen4's few-row decode copies: rows <= 8 run the decode GEMV per row.
        DType::Q8_0 => {
            hip(gpu.gemm_q8_0_batched_f32_chunked(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qt44-qt53-shared-projection-dtype",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// The grouped route's router is a row-batched projection even though the
/// generic MoE table intentionally has no BF16 router entry on RDNA3.
pub(crate) fn router_projection(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // Projected by the shared gate/up stage's launch instead.
    if router_with_shared(gpu, p) || router_with_selector(gpu, p) || router_shares_f16(gpu, p) {
        return Ok(());
    }
    batch_projection(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
    )
}

/// Whether a few-row forward (<= 8 rows, where the grouped BF16 row kernel is
/// bitwise each projection's own) projects the router together with the
/// all-BF16 shared selector/gate/up in one launch (the shared gate/up stage,
/// which the route always runs right after the router stage).
fn router_with_shared(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    gpu.route_rows(p.batch_size) <= 8
        && p.prelude.router.dtype == DType::BF16
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            [
                &shared.weights.selector,
                &shared.weights.gate,
                &shared.weights.up,
            ]
            .iter()
            .all(|w| w.dtype == DType::BF16)
        })
}

/// Whether a few-row forward on gfx11+ projects the router together with the
/// BF16 shared selector (one K, both reading `x_norm_batch`) in one grouped
/// BF16 row launch when the shared gate/up are not BF16 (the decode Q8
/// copies). The grouped row kernel computes each projection bitwise as its
/// own launch; the shared gate/up stage runs right after the router stage.
fn router_with_selector(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    (2..=8).contains(&gpu.route_rows(p.batch_size))
        && gpu.arch_caps.has_gfx11_plus_simt()
        && !router_with_shared(gpu, p)
        && p.prelude.router.dtype == DType::BF16
        && p.x_norm_batch.dtype == DType::F32
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            shared.weights.selector.dtype == DType::BF16
                && shared.weights.selector.k == p.prelude.router.k
        })
}

/// Whether the router and the all-BF16 shared selector/gate/up (one K) all
/// take the F16 WMMA route: the shared gate/up stage (which the route always
/// runs right after the router stage, with no writer of `x_norm_batch` in
/// between) then launches the router's own GEMM off the same F16 conversion
/// of `x_norm_batch` instead of converting it a second time.
fn router_shares_f16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    let router = &p.prelude.router;
    gpu.route_rows(p.batch_size) > 8
        && p.x_norm_batch.dtype == DType::F32
        && router.dtype == DType::BF16
        && gpu.qwen4_f16_wmma_applies(router.buf, router.k, p.batch_size)
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            [
                &shared.weights.selector,
                &shared.weights.gate,
                &shared.weights.up,
            ]
            .iter()
            .all(|w| {
                w.dtype == DType::BF16
                    && w.k == router.k
                    && gpu.qwen4_f16_wmma_applies(w.buf, w.k, p.batch_size)
            })
        })
}

/// Whether this call's shared expert runs the fused gate/up PeaceMaker launch
/// (`HIPFIRE_QWEN4_SHARED_GU_PM`, [`Gpu::qwen4_shared_gu_pm_applies`]): the
/// router and selector on the shared F16 conversion ([`router_shares_f16`]),
/// the BF16 recipe, a BF16 shared down on the F16 WMMA route (which then
/// reads the F16 hidden the launch leaves in `rotated`'s bytes), and not path
/// 2's fused unscatter activation, which reads the F32 gate/up outputs this
/// route does not produce. The one decision of the gate/up, activation and
/// down stages: it depends only on the call's operands and the device.
fn shared_gu_pm(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    router_shares_f16(gpu, p)
        && p.recipe.bf16_round_trip()
        && !(use_path2 && shared_in_unscatter(gpu, p))
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            let w = &shared.weights;
            w.gate.m == shared.intermediate
                && w.up.m == shared.intermediate
                && w.up.k == w.gate.k
                && w.down.dtype == DType::BF16
                && w.down.k == shared.intermediate
                && gpu.qwen4_f16_wmma_applies(w.down.buf, w.down.k, p.batch_size)
                && gpu.qwen4_shared_gu_pm_applies(w.gate.buf, w.up.buf, w.gate.m, w.gate.k, p.batch_size)
        })
}

/// The shared down GEMM's input tensor: the F16 hidden the fused gate/up left in
/// `rotated`'s bytes ([`shared_gu_pm`]), else the F32 activation.
fn shared_down_input(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    rotated: &GpuTensor,
    intermediate: usize,
) -> GpuTensor {
    if shared_gu_pm(gpu, p, use_path2) {
        rotated.prefix_as_f16(p.batch_size * intermediate)
    } else {
        rotated.shallow_clone()
    }
}

/// Shared selector and gate/up are ordinary batched projections.  This is the
/// typed route's BF16 exception to the generic MoE prefill table: the common
/// family remains fail-closed for QT53 while this exact route uses the native
/// BF16 XF32 GEMM launcher.
pub(crate) fn shared_gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let weights = &shared.weights;
    if router_with_shared(gpu, p) {
        // All read the natural activation: one launch (one F16 conversion).
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&p.prelude.router, p.prelude.router_logits),
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else if router_shares_f16(gpu, p) {
        let router = &p.prelude.router;
        if shared_gu_pm(gpu, p, use_path2) {
            // The router and selector groups (their own launches, off one F16
            // conversion), then the gate/up GEMMs, the BF16-recipe activation
            // (selector sigmoid in place) and the hidden's F16 conversion as ONE
            // PeaceMaker launch, the F16 hidden left in `rotated`'s bytes.
            let hidden = shared.rotated.prefix_as_f16(p.batch_size * shared.intermediate);
            let launched = hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_shared_gu_pm(
                (router.buf, p.prelude.router_logits, router.m),
                (weights.selector.buf, shared.scalar, weights.selector.m),
                weights.gate.buf,
                weights.up.buf,
                &hidden,
                p.x_norm_batch,
                weights.gate.m,
                router.k,
                p.batch_size,
            ))?;
            if !launched {
                return Err(DispatchError::Hip(
                    "qt44/qt53 router + shared fused gate/up PM route did not apply".into(),
                ));
            }
            return Ok(());
        }
        // The router group first (its own launches, as `router_projection`
        // would issue them), then the shared group, off one F16 conversion.
        let launched = hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_groups(
            &[
                &[(router.buf, p.prelude.router_logits, router.m)],
                &[
                    (weights.selector.buf, shared.scalar, weights.selector.m),
                    (weights.gate.buf, shared.gate_out, weights.gate.m),
                    (weights.up.buf, shared.up_out, weights.up.m),
                ],
            ],
            p.x_norm_batch,
            router.k,
            p.batch_size,
        ))?;
        if !launched {
            return Err(DispatchError::Hip(
                "qt44/qt53 router + shared F16 WMMA route did not apply".into(),
            ));
        }
    } else if [&weights.selector, &weights.gate, &weights.up]
        .iter()
        .all(|w| w.dtype == DType::BF16)
    {
        // All read the natural activation: one shared F16 conversion.
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else {
        if router_with_selector(gpu, p) {
            super::layer_ops::project_weights(
                gpu,
                p.x_norm_batch,
                p.batch_size,
                None,
                &[
                    (&p.prelude.router, p.prelude.router_logits),
                    (&weights.selector, shared.scalar),
                ],
            )?;
        } else {
            let x_selector = match shared.weights.selector.dtype {
                DType::BF16 | DType::F32 => p.x_norm_batch,
                DType::MQ4G256V2 => p.x_rot_batch,
                _ => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "qt44-qt53-shared-selector-dtype",
                        arch: "",
                        quant: "unsupported",
                    })
                }
            };
            batch_projection(
                gpu,
                &shared.weights.selector,
                x_selector,
                shared.scalar,
                p.batch_size,
            )?;
        }
        let x_gate = match shared.weights.gate.dtype {
            DType::BF16 | DType::F32 | DType::Q8_0 => p.x_norm_batch,
            DType::MQ4G256V2 => p.x_rot_batch,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "qt44-qt53-shared-gate-dtype",
                    arch: "",
                    quant: "unsupported",
                })
            }
        };
        let (gate, up) = (&shared.weights.gate, &shared.weights.up);
        if gate.dtype == DType::Q8_0
            && up.dtype == DType::Q8_0
            && gate.k == 2560
            && up.k == 2560
            && gpu.gemv_q8_0_staged_rows_supported(2560, gpu.route_rows(p.batch_size))
        {
            // Few-row verify: the Q8 decode copies of the shared gate and up
            // read the same rows; one launch, each row the single kernel's.
            hip(gpu.gemv_q8_0_k2560_staged_rows_x2(
                gate.buf,
                shared.gate_out,
                gate.m,
                up.buf,
                shared.up_out,
                up.m,
                x_gate,
                p.batch_size,
            ))?;
        } else {
            batch_projection(gpu, gate, x_gate, shared.gate_out, p.batch_size)?;
            batch_projection(gpu, up, x_gate, shared.up_out, p.batch_size)?;
        }
    }
    // BF16 recipe: every reader rounds what it reads (the top-10 router its
    // logits, the shared activation its selector, gate and up), so no
    // separate round trip is owed here.
    Ok(())
}

/// Apply the shared expert activation after the selector and gate/up projections.
pub(crate) fn shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // Path 2's fused unscatter launch runs it (see `shared_in_unscatter`).
    if use_path2 && shared_in_unscatter(gpu, p) {
        return Ok(());
    }
    // The fused shared gate/up already left the activation (and the sigmoid selector).
    if shared_gu_pm(gpu, p, use_path2) {
        return Ok(());
    }
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    // The live rows only: the shared buffers are sized for the chunk cap.
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    if p.recipe.bf16_round_trip() {
        // The selector's round trip and sigmoid ride in the same launch; the
        // shared down's scaled add rounds the selector again on read.
        hip(gpu.shared_expert_activation_bf16_f32(
            &gate,
            &up,
            &rotated,
            shared.scalar,
            p.batch_size,
        ))?;
    } else {
        hip(gpu.silu_mul_f32(&gate, &up, &rotated))?;
    }
    if shared.weights.down.dtype == DType::MQ4G128V2 {
        hip(gpu.rotate_x_mq_128_v2(
            shared.rotated,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        ))?;
    }
    Ok(())
}

/// Fold the shared expert into the residual at the recipe's chosen placement,
/// applying the selector sigmoid exactly once.
pub(crate) fn shared_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // The unfused shared down writes F32 rows into `shared_down_out` (an EP
    // row gather owns `down_expanded` until its peers finish reading it), else
    // into `down_expanded`; the BF16 epilogue below may decline at run time, so
    // the fallback is always owed.
    require_shared_rows(p, shared_down_bytes(p.batch_size, p.down_m)?)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    // The fused shared gate/up leaves the activation as F16 in `rotated`'s bytes.
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    if down.dtype == DType::BF16 && p.recipe.bf16_round_trip()
        && hip(gpu.qwen4_shared_down_bf16_epi(
            down.buf, &hidden, target, shared.scalar,
            down.m, down.k, p.batch_size,
        ))?
    {
        return Ok(());
    }
    let out = f32_view(p.shared_down_out.unwrap_or(p.down_expanded), 0, p.batch_size * p.down_m);
    match down.dtype {
        DType::BF16 | DType::F32 | DType::Q8_0 | DType::MQ4G256V2 | DType::MQ4G128V2 => {
            // The shared activation is already in the basis consumed by this
            // projection (G128 for QT53); the BF16 storage boundary, if any,
            // was applied by the recipe before this launch.
            if hidden.dtype == DType::F16 {
                // The fused F16 hidden (BF16 down on the F16 WMMA route, see
                // `shared_gu_pm`): the GEMM `batch_projection` would launch off
                // the F32 activation's conversion, minus the conversion.
                hip(gpu.gemm_bf16_xf16_f16_wmma(down.buf, &hidden, &out, down.m, down.k, p.batch_size))?;
            } else {
                batch_projection(gpu, down, &hidden, &out, p.batch_size)?;
            }
        }
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "qt44-qt53-shared-down-dtype",
                arch: "",
                quant: "unsupported",
            })
        }
    }
    if p.recipe.bf16_round_trip() {
        // The scaled add rounds `out`, the scalar and the residual to BF16
        // itself and stores a BF16 value, so no round trip brackets it.
        bf16_scaled_add_batched(
            gpu,
            &Bf16ScaledAddBatched {
                residual: target,
                value: &out,
                scalar: shared.scalar,
                rows: p.batch_size,
                elements: p.down_m,
            },
        )
        .map_err(|error| DispatchError::Hip(error.to_string()))?;
    } else {
        hip(gpu.sigmoid_scaled_residual_add_batched_f32(
            target,
            &out,
            shared.scalar,
            p.batch_size,
            p.down_m,
        ))?;
    }
    Ok(())
}

/// The HC write that follows a sealed MoE call and may ride the shared down's
/// GEMM epilogue (`HIPFIRE_QWEN4_HC_FUSE` >= 3): its streams, the gates the
/// paired HC read produced, and the tensor it reads as `mixed` (which must be
/// the routed target).
pub(crate) struct HcSharedDown {
    pub streams: GpuTensor,
    pub gates: GpuTensor,
    pub mixed: *mut std::ffi::c_void,
    pub rows: usize,
    pub hidden: usize,
    pub state_bf16: bool,
    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the next HC read's operands, when the whole
    /// MoE tail runs as one row kernel (see [`row_fold_applies`]).
    pub row_fold: Option<HcRowFoldNext>,
}

/// The HC read that follows the HC write of a row-folded MoE call, and that
/// read's paired write: the row kernel computes the read's norm (its F16 row)
/// and that write's gate logits.
pub(crate) struct HcRowFoldNext {
    pub norm_weight: GpuTensor,
    pub gate_weight: GpuTensor,
    pub next_gates: GpuTensor,
}

impl HcRowFoldNext {
    /// What the row kernel needs of `read` and its paired `write`.
    pub(crate) fn offer(
        read: &super::layer_ops::HyperReadOp<'_>,
        write: &super::layer_ops::HyperWriteOp<'_>,
    ) -> Self {
        Self {
            norm_weight: f32_view(read.norm_weight, 0, read.norm_weight.numel()),
            gate_weight: f32_view(write.block_inject.buf, 0, write.block_inject.buf.numel()),
            next_gates: f32_view(write.gates, 0, read.rows * write.branches),
        }
    }
}

impl HcSharedDown {
    /// What the shared down needs of `write`, or `None` when its operands
    /// are not the plain four-branch F32-typed streams / gates it folds into.
    pub(crate) fn offer(write: &super::layer_ops::HyperWriteOp<'_>) -> Option<Self> {
        let wide = write.branches.checked_mul(write.hidden)?;
        let (streams_len, gates_len) = (write.rows.checked_mul(wide)?, write.rows.checked_mul(4)?);
        (write.branches == 4
            && write.input.buf.as_ptr() == write.output.buf.as_ptr()
            && write.input.dtype == DType::F32
            && write.input.numel() >= streams_len
            && write.gates.dtype == DType::F32
            && write.gates.numel() >= gates_len)
            .then(|| Self {
                streams: f32_view(write.input, 0, streams_len),
                gates: f32_view(write.gates, 0, gates_len),
                mixed: write.mixed.buf.as_ptr(),
                rows: write.rows,
                hidden: write.hidden,
                state_bf16: write.state_bf16,
                row_fold: None,
            })
    }
}

/// [`shared_down`] with the paired HC write folded into the GEMM epilogue
/// (`hc`).  Bitwise the shared down followed by `hyper_write`: the BF16 shared
/// down on the Qwen4 F16 WMMA route, the BF16 scaled add, then the write.  The
/// routed target is not rewritten (it is dead after the write).  Returns
/// `false`, having launched nothing, when `hc` does not fit this call; the
/// caller then runs [`shared_down`] and the write unfused.
pub(crate) fn shared_down_hc(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    use_path2: bool,
) -> Result<bool, DispatchError> {
    require_geometry(p)?;
    let Some(shared) = p.prelude.shared.as_ref() else {
        return Ok(false);
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if !hc_shared_down_admits(gpu, p, hc) {
        if hc.row_fold.is_some() {
            return Err(DispatchError::Hip(
                "qt44 shared down: the HC row fold was planned but its route no longer applies".into(),
            ));
        }
        return Ok(false);
    }
    if let Some(next) = &hc.row_fold {
        shared_down_row_fold(gpu, p, hc, next, use_path2)?;
        return Ok(true);
    }
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
        down.buf,
        &hidden,
        down.m,
        down.k,
        p.batch_size,
        target,
        shared.scalar,
        &hc.streams,
        &hc.gates,
        hc.state_bf16,
        false,
    ))?;
    Ok(true)
}

/// Whether the shared down of this call can carry `hc` (the conditions of
/// [`shared_down_hc`]).
fn hc_shared_down_admits(gpu: &Gpu, p: &MoePrefillParams<'_>, hc: &HcSharedDown) -> bool {
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_fuse_level() >= 3
        && down.dtype == DType::BF16
        && down.rotation.is_none()
        && down.awq_scale.is_none()
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && down.m == p.down_m
        && down.k == shared.intermediate
        && hc.rows == p.batch_size
        && hc.hidden == p.down_m
        && hc.mixed == target.buf.as_ptr()
        && target.numel() >= p.batch_size * p.down_m
        && shared.scalar.numel() >= p.batch_size
        && gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(down.buf, down.m, down.k, p.batch_size)
}

/// Whether this call's whole tail (combine, shared fold, `hc`'s HC write, the
/// next read's norm) can run as the HC row fold
/// (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the zero-initialized BF16-row combine of
/// path 2 into the routed target (which, nothing reading it, then holds the
/// BF16 shared-down rows), the HC-write shared down, BF16 streams.
pub(crate) fn row_fold_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    hc: &HcSharedDown,
) -> bool {
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_row_fold_enabled()
        && hc.state_bf16
        && combine_initial_zero_applies(gpu, p, use_path2)
        && hc_shared_down_admits(gpu, p, hc)
        && gpu.hc_row_fold_applies(hc.hidden)
        && p.k_top == 10
        && p.down_expanded.buf.size() >= p.batch_size * 10 * 8
        && target.buf.size() >= p.batch_size * p.down_m * 2
}

/// The row-folded combine stage: only the rank order (the row kernel reads the
/// grouped rows itself).
pub(crate) fn combine_order_only(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    validate_scratch(gpu, p, true, grouped_rows)?;
    hip(gpu.moe_combine_order_top10(
        p.inverse_perm,
        p.topk_indices,
        p.topk_weights,
        p.down_expanded,
        grouped_rows,
        p.batch_size,
    ))
}

/// The row-folded shared down: the shared-down GEMM stores its BF16 rows into
/// the (otherwise unwritten) routed target, then one row kernel per token
/// combines the ten expert rows, folds the shared row, writes the HC streams
/// and runs the next read's norm + gate projection.  Bitwise
/// [`combine`] (`initial_zero`) + [`shared_down_hc`] + the next read's
/// `hyper_norm_gate_outputs`.
fn shared_down_row_fold(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    next: &HcRowFoldNext,
    use_path2: bool,
) -> Result<(), DispatchError> {
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    // The next read's F16 row lives in the shared F16 scratch, which the
    // shared-down GEMM's own F16 input also uses (it is dead by the time the
    // row kernel writes): size it first so nothing reallocates under the GEMM.
    let x16 = hip(gpu.qwen4_f16_x_scratch(hc.rows * 4 * hc.hidden))?;
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(
        down.buf,
        &hidden,
        down.m,
        down.k,
        p.batch_size,
        target,
    ))?;
    hip(gpu.hc_row_fold_norm_gate(&rdna_compute::hc_row_fold::HcRowFold {
        grouped_down: p.y_down_grouped,
        order: p.down_expanded,
        shared_bf16: target,
        selector: shared.scalar,
        streams: &hc.streams,
        gates: &hc.gates,
        norm_weight: &next.norm_weight,
        gate_weight: &next.gate_weight,
        next_gates: &next.next_gates,
        normalized_f16: &x16,
        rows: hc.rows,
        hidden: hc.hidden,
    }))
}

/// Whether path 2 takes the symmetric IU4 arm (fn-moe-sym): the layer's
/// experts were verified symmetric at load (the policy says so), the recipe
/// keeps the BF16 boundaries the kernels implement, the activation is F32,
/// and the device admits it (`HIPFIRE_QWEN4_MOE_SYM_IU4`: default on for
/// gfx1151 and gfx1201; >= 512 rows, C2 producers). It replaces scatter,
/// gate/up, unscatter/rotation and down; the combine reads its BF16 rows like
/// the F16 WMMA arm's.
fn sym_iu4(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2
        && p
            .route_policy
            .is_some_and(|policy| policy.capability == MoeRouteCapability::Qt44Qt53GroupedSymmetric)
        && p.recipe.bf16_round_trip()
        && p.x_norm_batch.dtype == DType::F32
        && p.down_k == p.mi
        && gpu.qwen4_moe_sym_iu4_applies(p.batch_size)
}

/// Path-2 grouping (the stage runs on path 2 only).
pub(crate) fn scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
    if sym_iu4(gpu, p, true) {
        return hip(gpu.moe_scatter_stable_top10(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            total_slots,
            p.n_exp,
            grouped_rows,
        ));
    }
    hip(gpu.moe_scatter_fused_top10(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        total_slots,
        p.n_exp,
        grouped_rows,
        crate::families::moe::MOE_GROUPED_BLOCK_M,
    ))
}

/// Whether the grouped gate/up takes the F16 WMMA arm and hands the unscatter
/// BF16 rows: the recipe rounds gate/up to BF16 before SiLU anyway.  The
/// BF16-output entries exist in the gfx11 source only.
fn gateup_bf16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    p.recipe.bf16_round_trip()
        && gpu.arch_caps.has_wmma_w32()
        && gpu.qwen4_moe_gateup_wmma_applies(2 * p.mi, p.gate_up_k, p.batch_size)
}

/// Whether the grouped down takes the F16 WMMA arm: rotated straight to F16 in
/// `down`, output as BF16 for the combine, which rounds every row to BF16.
fn down_wmma(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    p.down_k == p.mi && gpu.qwen4_moe_down_wmma_applies(p.down_m, p.down_k, p.batch_size * p.k_top)
}


/// The tile ids the grouped symmetric GEMMs read: the scatter's `expert_tile_ids`, or under the
/// owner-routed EP chunk exchange its masked copy (`-1` on every tile of an expert another rank
/// owns).
fn gemm_tile_ids<'p>(p: &'p MoePrefillParams<'_>) -> &'p GpuTensor {
    p.ep_tile_exchange
        .map_or(p.expert_tile_ids, |tx| tx.tile_ids_owned)
}

fn tile_exchange_eager_only(gpu: &Gpu, what: &str) -> Result<(), DispatchError> {
    // The exchange waits on peer devices and a host barrier; neither can be
    // recorded into a graph or retained body. EP is eager-only.
    if gpu.graphs.capture_mode
        || gpu.graphs.verify.capturing.is_some()
        || gpu.graphs.replay.capturing.is_some()
        || gpu.replay.is_recording()
    {
        return Err(DispatchError::Hip(format!(
            "sealed moe: expert-parallel tile {what} refused while a graph capture or replay record is armed"
        )));
    }
    Ok(())
}

/// Padded tile capacity of a grouped layout (the symmetric GEMMs' grid height).
fn grouped_tiles(grouped_rows: usize) -> usize {
    grouped_rows.div_ceil(16)
}

/// Owner-routed EP chunk MoE, after the scatter: write the masked copy of the tile ids (the
/// owner of every expert is the exchange's table input). Valid only on the symmetric IU4 arm.
pub(crate) fn tile_mask(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    tx: MoeTileExchangeParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if !sym_iu4(gpu, p, true) {
        return Err(DispatchError::Hip(
            "qt44 tile mask: the owner-routed chunk exchange needs the symmetric IU4 arm".into(),
        ));
    }
    tile_exchange_eager_only(gpu, "mask")?;
    let (rank, table, n_exp) = tx.exchange.owners();
    if n_exp != p.n_exp {
        return Err(DispatchError::Hip(format!(
            "qt44 tile mask: the owner table covers {n_exp} experts, the layer has {}",
            p.n_exp
        )));
    }
    hip(gpu.ep_tile_mask(
        p.expert_tile_ids,
        tx.tile_ids_owned,
        table,
        grouped_tiles(grouped_rows),
        n_exp,
        rank,
    ))
}

/// Owner-routed EP chunk MoE, between the mutation fence and the combine: fill every
/// peer-owned live tile of `y_down_grouped` from its owner.
pub(crate) fn tile_exchange(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    tx: MoeTileExchangeParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if !sym_iu4(gpu, p, true) {
        return Err(DispatchError::Hip(
            "qt44 tile exchange: the owner-routed chunk exchange needs the symmetric IU4 arm".into(),
        ));
    }
    tile_exchange_eager_only(gpu, "exchange")?;
    tx.exchange
        .exchange(
            gpu,
            p.ep_layer_idx,
            p.expert_tile_ids,
            p.y_down_grouped,
            grouped_tiles(grouped_rows),
        )
        .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))
}

/// Whether gate/up stores the SwiGLU activation (BF16) straight from its
/// epilogue and `down` unscatters it fused with the 128-wide rotation.
fn gateup_silu(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    gateup_bf16(gpu, p) && down_wmma(gpu, p) && p.mi % 128 == 0
}

/// Whether this layer's routed experts live in host-mapped memory (spilled
/// past the VRAM budget); the symmetric IU4 GEMMs then take a wider expert-run
/// tile. A layer's experts share one residency, so expert 0 decides.
fn experts_host_mapped(p: &MoePrefillParams<'_>) -> bool {
    p.expert_stage_ptrs.is_none() && p.routed_experts.host_mapped()
}

pub(crate) fn gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    let gate_up_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_gate_up_ptrs, |stage| stage.gate_up);
    if sym_iu4(gpu, p, use_path2) {
        let xq = hip(gpu.qwen4_moe_rotate256_i4(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
        return hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            gate_up_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            &xq,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
            experts_host_mapped(p),
        ));
    }
    if use_path2 && gateup_bf16(gpu, p) {
        let x_f16 = if gateup_rotates_f16(gpu, p, use_path2) {
            Some(hip(gpu.rotate_x_mq_batched_f16(
                p.x_norm_batch,
                p.gate_up_k,
                p.batch_size,
            ))?)
        } else {
            None
        };
        let gemm = if gateup_silu(gpu, p) {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_silu_bf16out
        } else {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_bf16out
        };
        hip(gemm(
            gpu,
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            x_f16.as_ref().unwrap_or(p.x_rot_batch),
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else if use_path2 {
        hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else {
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            gate_up_ptrs,
            p.topk_indices,
            p.x_rot_batch,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            p.gate_up_k,
            p.batch_size,
        ))?;
        let active = p
            .batch_size
            .checked_mul(p.k_top)
            .and_then(|slots| slots.checked_mul(p.mi))
            .ok_or_else(|| DispatchError::Hip("grouped gate/up extent overflows".into()))?;
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
            hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))?;
        }
        Ok(())
    }
}

/// Path-2 unscatter fused with the SwiGLU activation: the grouped gate/up
/// rows go straight to the rotation input (`rot_batch`) with the recipe's BF16
/// round trips, bitwise the unfused unscatter -> round trip -> silu_mul ->
/// round trip sequence, rotated in the same launch when
/// [`unscatter_rotates`] (else [`activation`] rotates).
pub(crate) fn unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    validate_scratch(gpu, p, true, grouped_rows)?;
    // Symmetric IU4: SiLU in the gate/up epilogue, rotation + A4 in `down`.
    if sym_iu4(gpu, p, true) {
        return Ok(());
    }
    // SiLU in the gate/up epilogue, unscatter + rotation in the down stage.
    if gateup_silu(gpu, p) {
        return Ok(());
    }
    if gateup_bf16(gpu, p) {
        return hip(gpu.moe_gate_up_unscatter_silu_top10_bf16in(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
        ));
    }
    if unscatter_rotates(gpu, p) {
        // The shared expert's activation rides in the same launch; its stage
        // skipped it (the shared down reads it after the combine).
        let shared = p
            .prelude
            .shared
            .as_ref()
            .filter(|_| shared_in_unscatter(gpu, p))
            .map(|shared| {
                let live = p.batch_size * shared.intermediate;
                (
                    f32_view(shared.gate_out, 0, live),
                    f32_view(shared.up_out, 0, live),
                    f32_view(shared.rotated, 0, live),
                    shared.scalar,
                )
            });
        let shared = shared
            .as_ref()
            .map(|(gate, up, out, selector)| SharedExpertActivation {
                gate,
                up,
                out,
                selector,
                selectors: p.batch_size,
            });
        return hip(gpu.moe_gate_up_unscatter_silu_rotate128_top10(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
            shared.as_ref(),
        ));
    }
    hip(gpu.moe_gate_up_unscatter_silu_top10(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.rot_batch,
        p.mi,
        grouped_rows,
        p.recipe.bf16_round_trip(),
    ))
}

/// Whether path 2's F32 unscatter also applies the down's 128-wide rotation
/// (in the same launch), leaving [`activation`] nothing to do.
fn unscatter_rotates(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !sym_iu4(gpu, p, true)
        && !gateup_bf16(gpu, p)
        && !down_wmma(gpu, p)
        && p.mi.is_multiple_of(128)
}

/// Whether path 2's fused unscatter launch also runs the BF16 shared expert
/// activation: its only reader, a natural-basis shared down, runs after the
/// combine (so after the unscatter).
fn shared_in_unscatter(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    unscatter_rotates(gpu, p)
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && p.prelude
            .shared
            .as_ref()
            .is_some_and(|shared| shared.weights.down.dtype != DType::MQ4G128V2)
}

pub(crate) fn activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
    // Path 2's unscatter already produced the activation.
    if !use_path2 {
        hip(gpu.silu_mul_f32(p.gate_batch, p.up_batch, p.rot_batch))?;
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(p.rot_batch))?;
        }
    }
    // The F16 WMMA and symmetric IU4 downs rotate the grouped rows themselves
    // (see `down`); the F32 unscatter may already have rotated.
    if use_path2 && (sym_iu4(gpu, p, use_path2) || down_wmma(gpu, p) || unscatter_rotates(gpu, p)) {
        return Ok(());
    }
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))
}

/// A few-row forward (speculative verify) on grouped path 2 without the WMMA
/// down: the route slots are nearly all distinct experts, so the grouped down
/// tile (16 slot rows per expert) runs almost empty; the decode kernel's
/// per-slot down over the unscattered, rotated rows streams the experts
/// faster.  Its per-slot dot is the grouped kernel's (bitwise), combined in
/// slot order.
fn indexed_down(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2 && gpu.route_rows(p.batch_size) <= 8 && !down_wmma(gpu, p)
}

/// Release the stage immediately after the routed down, before combine.
fn release_stage(gpu: &Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if let Some(stage) = p.expert_stage_ptrs {
        hip(gpu.hip.event_record(stage.free, gpu.active_stream.as_ref()))?;
    }
    Ok(())
}

pub(crate) fn down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    down_stage(gpu, p, use_path2, grouped_rows)
}

/// [`down`] after its scratch preflight; the indexed arm re-enters on path 1
/// with the same (already validated) buffers.
fn down_stage(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let down_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_down_ptrs, |stage| stage.down);
    let total_slots = p.batch_size * p.k_top;
    if sym_iu4(gpu, p, use_path2) {
        // Grouped BF16 SwiGLU rows -> FWHT128 -> compact flat-slot A4 ->
        // grouped IU4 down, BF16 per grouped row for the BF16-input combine.
        let xq = hip(gpu.qwen4_moe_rotate128_i4(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.down_k,
            grouped_rows,
            total_slots,
        ))?;
        if let Some(tx) = p.ep_tile_exchange {
            // The peer reads this rank's `y_down_grouped` between its producer
            // and completion events of the previous layer's exchange: no write
            // before its pull completed.
            tx.exchange
                .before_down(gpu)
                .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))?;
        }
        hip(gpu.gemm_qwen4_moe_down_iu4_sym(
            down_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            &xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            experts_host_mapped(p),
        ))?;
        return release_stage(gpu, p);
    }
    if indexed_down(gpu, p, use_path2) {
        return down_stage(gpu, p, false, grouped_rows);
    }
    if use_path2 && down_wmma(gpu, p) {
        // Rotation and GEMM in one stage: the rotated F16 rows live in the
        // shared FP16 scratch, which another stage's GEMM would overwrite.
        let x_f16 = if gateup_silu(gpu, p) {
            hip(gpu.moe_unscatter_rotate128_f16(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.mi,
                grouped_rows,
                total_slots,
            ))?
        } else {
            hip(gpu.rotate_x_mq_128_v2_f16(p.rot_batch, p.mi, total_slots))?
        };
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &x_f16,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            grouped_rows,
            true,
        ))?;
    } else if use_path2 {
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            p.n_exp,
        ))?;
        // A BF16-source recipe rounds each grouped expert output before
        // weighted combination, just as on the scalar route.  The grouped
        // combine kernel performs that same RNE rounding on every row it
        // reads, so a separate pass over all (padded) grouped rows would only
        // re-round values that are already BF16-exact.
    } else {
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.down_m,
            p.down_k,
            p.batch_size,
            p.n_exp,
        ))?;
        // The combine rounds every expert output it reads to BF16 itself.
    }
    release_stage(gpu, p)
}

/// Whether [`combine`] can start from +0.0 instead of the zero-filled target
/// (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`): the BF16-row combine of path 2, with
/// nothing writing the target between its zero fill and the combine (the
/// shared down, the only other writer, runs after it).
pub(crate) fn combine_initial_zero_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> bool {
    use_path2
        && !indexed_down(gpu, p, use_path2)
        && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2))
        && p.recipe.shared_after_combine()
}

/// `initial_zero`: the combine writes the target from +0.0 rather than
/// accumulating into zeros the caller filled (requires
/// [`combine_initial_zero_applies`]).
pub(crate) fn combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    combine_stage(gpu, p, use_path2, grouped_rows, initial_zero)
}

/// [`combine`] after its scratch preflight; the indexed arm re-enters on
/// path 1 with the same (already validated) buffers.
fn combine_stage(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if initial_zero && !combine_initial_zero_applies(gpu, p, use_path2) {
        return Err(DispatchError::Hip(
            "qt44 combine: zero-initialized target requested off the BF16-row path".into(),
        ));
    }
    if indexed_down(gpu, p, use_path2) {
        return combine_stage(gpu, p, false, grouped_rows, false);
    }
    if use_path2 && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2)) {
        // Both downs store BF16 grouped rows. The rank order goes to
        // `down_expanded`: unused on this route until the shared down, which
        // runs after the combine.
        hip(gpu.moe_down_combine_grouped_top10_bf16in(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_expanded,
            p.down_m,
            grouped_rows,
            p.batch_size,
            initial_zero,
        ))?;
    } else if use_path2 {
        hip(gpu.moe_down_combine_grouped_top10(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            grouped_rows,
            p.batch_size,
        ))?;
    } else {
        let expanded = f32_view(p.down_expanded, 0, p.batch_size * p.k_top * p.down_m);
        hip(gpu.moe_down_combine_top10_batched(
            &expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            p.batch_size,
        ))?;
    }
    // A shared down placed after the combine rounds `target` to BF16 as its
    // first read, so the round trip is owed only when nothing follows.
    if p.recipe.bf16_round_trip() && !p.recipe.shared_after_combine() {
        hip(gpu.bf16_round_trip_f32(target))?;
    }
    Ok(())
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    const MI: usize = 640;
    const HIDDEN: usize = 2560;
    const K_TOP: usize = 10;
    const N_EXP: usize = 512;
    const WORD: usize = 4;

    fn dims(rows: usize) -> ScratchDims {
        let slots = rows * K_TOP;
        ScratchDims {
            rows,
            k_top: K_TOP,
            mi: MI,
            down_m: HIDDEN,
            grouped_rows: grouped_rows(slots),
            shared_down: true,
        }
    }

    /// `checked_grouped_m_total_bound` for 16-row blocks: slots plus up to 15
    /// padding rows per active expert.
    fn grouped_rows(slots: usize) -> usize {
        (slots + 15 * slots.min(N_EXP)).div_ceil(16) * 16
    }

    fn route(path2: bool) -> ScratchRoute {
        ScratchRoute {
            path2,
            sym_iu4: false,
            gateup_bf16: false,
            gateup_silu: false,
            down_wmma: false,
            indexed_down: false,
        }
    }

    fn sym() -> ScratchRoute {
        ScratchRoute { sym_iu4: true, ..route(true) }
    }

    fn wmma() -> ScratchRoute {
        ScratchRoute { gateup_bf16: true, gateup_silu: true, down_wmma: true, ..route(true) }
    }

    fn wmma_unfused() -> ScratchRoute {
        ScratchRoute { gateup_bf16: true, down_wmma: true, ..route(true) }
    }

    fn indexed() -> ScratchRoute {
        ScratchRoute { indexed_down: true, ..route(true) }
    }

    /// What the generic validator demands of a Full-width F32 scratch.
    fn full_scratch(rows: usize) -> ScratchBytes {
        let slots = rows * K_TOP;
        let m_total = grouped_rows(slots);
        ScratchBytes {
            gate_batch: slots * MI * 4,
            up_batch: slots * MI * 4,
            rot_batch: slots * MI * 4,
            down_expanded: slots * HIDDEN * 4,
            y_gate_up_grouped: m_total * 2 * MI * 4,
            y_down_grouped: m_total * HIDDEN * 4,
        }
    }

    /// The frozen compact capacities (`c = min(n, 512)`) for chunk `n`.
    fn compact_scratch(n: usize) -> ScratchBytes {
        let c = n.min(512);
        let (gc, gn) = (grouped_rows(c * K_TOP), grouped_rows(n * K_TOP));
        ScratchBytes {
            gate_batch: c * 10 * 640 * 4,
            up_batch: c * 10 * 640 * 4,
            rot_batch: c * 10 * 640 * 4,
            down_expanded: (c * 10 * 2560 * 4).max(n * 2560 * 4).max(n * 10 * 8),
            y_gate_up_grouped: (gc * 1280 * 4).max(gn * 640 * 2),
            y_down_grouped: (gc * 2560 * 4).max(gn * 2560 * 2),
        }
    }

    #[test]
    fn a_separate_shared_down_output_leaves_down_expanded_to_the_routed_rows() {
        // With `shared_down_out` the F32 shared-down rows are not charged to
        // `down_expanded` (they are checked on that tensor by `validate_scratch`).
        let rows = 8192;
        let shared = ScratchDims { shared_down: true, ..dims(rows) };
        let separate = ScratchDims { shared_down: false, ..dims(rows) };
        let with = sym().requirement(&shared).unwrap();
        let without = sym().requirement(&separate).unwrap();
        assert_eq!(with.down_expanded, shared_down_bytes(rows, HIDDEN).unwrap());
        assert_eq!(without.down_expanded, rows * 80);
        // Everything else is the same.
        assert_eq!(
            ScratchBytes { down_expanded: 0, ..with },
            ScratchBytes { down_expanded: 0, ..without }
        );
        // The few-row indexed route's routed rows are the same either way.
        let few = ScratchDims { shared_down: false, ..dims(4) };
        assert_eq!(indexed().requirement(&few).unwrap().down_expanded, 4 * K_TOP * HIDDEN * 4);
    }

    #[test]
    fn symmetric_route_needs_bf16_grouped_rows_and_the_shared_down_fallback() {
        let rows = 8192;
        let g = grouped_rows(rows * K_TOP);
        assert_eq!(g, 89_600);
        let need = sym().requirement(&dims(rows)).unwrap();
        assert_eq!(need.y_gate_up_grouped, g * MI * 2);
        assert_eq!(need.y_down_grouped, g * HIDDEN * 2);
        // The F32 shared-down fallback dominates the 80-byte-per-row order.
        assert_eq!(need.down_expanded, rows * HIDDEN * 4);
        assert!(need.down_expanded > rows * 80);
        assert_eq!((need.gate_batch, need.up_batch, need.rot_batch), (0, 0, 0));
    }

    #[test]
    fn combine_order_alone_is_eighty_bytes_per_row() {
        let mut no_shared = dims(1024);
        no_shared.shared_down = false;
        assert_eq!(sym().requirement(&no_shared).unwrap().down_expanded, 1024 * 80);
        assert_eq!(wmma().requirement(&no_shared).unwrap().down_expanded, 1024 * 80);
    }

    #[test]
    fn f32_grouped_arms_keep_todays_full_sizes() {
        let rows = 700;
        let g = grouped_rows(rows * K_TOP);
        let need = route(true).requirement(&dims(rows)).unwrap();
        assert_eq!(need.y_gate_up_grouped, g * 2 * MI * 4);
        assert_eq!(need.y_down_grouped, g * HIDDEN * 4);
        assert_eq!(need.rot_batch, rows * K_TOP * MI * 4);
        assert_eq!(need.down_expanded, rows * HIDDEN * 4);
        assert_eq!((need.gate_batch, need.up_batch), (0, 0));
    }

    #[test]
    fn wmma_arms_store_bf16_grouped_rows() {
        let rows = 900;
        let g = grouped_rows(rows * K_TOP);
        let fused = wmma().requirement(&dims(rows)).unwrap();
        assert_eq!(fused.y_gate_up_grouped, g * MI * 2);
        assert_eq!(fused.y_down_grouped, g * HIDDEN * 2);
        assert_eq!(fused.rot_batch, 0);
        let unfused = wmma_unfused().requirement(&dims(rows)).unwrap();
        assert_eq!(unfused.y_gate_up_grouped, g * 2 * MI * 2);
        assert_eq!(unfused.rot_batch, rows * K_TOP * MI * 4);
    }

    #[test]
    fn indexed_down_and_path1_need_the_batch_buffers() {
        let rows = 8;
        let slots = rows * K_TOP;
        let few = indexed().requirement(&dims(rows)).unwrap();
        assert_eq!(few.down_expanded, slots * HIDDEN * 4);
        assert_eq!(few.rot_batch, slots * MI * 4);
        assert_eq!(few.y_down_grouped, 0);
        let path1 = route(false).requirement(&dims(rows)).unwrap();
        assert_eq!(path1.gate_batch, slots * MI * 4);
        assert_eq!(path1.up_batch, slots * MI * 4);
        assert_eq!(path1.rot_batch, slots * MI * 4);
        assert_eq!(path1.down_expanded, slots * HIDDEN * 4);
        assert_eq!((path1.y_gate_up_grouped, path1.y_down_grouped), (0, 0));
    }

    #[test]
    fn full_width_buffers_always_pass() {
        for rows in [1, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            for r in [route(false), route(true), sym(), wmma(), wmma_unfused(), indexed()] {
                let need = r.requirement(&dims(rows)).unwrap();
                need.require_within(&full_scratch(rows))
                    .unwrap_or_else(|e| panic!("rows={rows} route={r:?}: {e:?}"));
            }
        }
    }

    #[test]
    fn buffers_short_by_one_word_are_rejected_naming_the_buffer() {
        for rows in [8, 513, 4096] {
            for r in [route(false), route(true), sym(), wmma(), wmma_unfused(), indexed()] {
                let need = r.requirement(&dims(rows)).unwrap();
                // Exactly the requirement passes.
                need.require_within(&need).unwrap();
                for (name, bytes) in need.fields() {
                    if bytes == 0 {
                        continue;
                    }
                    let mut held = need;
                    match name {
                        "gate_batch" => held.gate_batch -= WORD,
                        "up_batch" => held.up_batch -= WORD,
                        "rot_batch" => held.rot_batch -= WORD,
                        "down_expanded" => held.down_expanded -= WORD,
                        "y_gate_up_grouped" => held.y_gate_up_grouped -= WORD,
                        "y_down_grouped" => held.y_down_grouped -= WORD,
                        other => panic!("unknown buffer {other}"),
                    }
                    let err = need.require_within(&held).expect_err(name);
                    let DispatchError::Hip(message) = err else {
                        panic!("unexpected error kind for {name}");
                    };
                    assert!(message.contains(name), "{message}");
                    assert!(message.contains(&bytes.to_string()), "{message}");
                }
            }
        }
    }

    #[test]
    fn shared_down_fallback_is_owed_beyond_the_combine_order() {
        let rows = 2048;
        let need = sym().requirement(&dims(rows)).unwrap();
        let mut held = need;
        // Room for the 80-byte-per-row order only.
        held.down_expanded = rows * 80;
        let DispatchError::Hip(message) = need.require_within(&held).unwrap_err() else {
            panic!("unexpected error kind");
        };
        assert!(message.contains("down_expanded"), "{message}");
    }

    #[test]
    fn compact_capacities_admit_the_proven_routes() {
        let n = 8192;
        let held = compact_scratch(n);
        // Ordinary large prefill on the symmetric IU4 route.
        sym().requirement(&dims(n)).unwrap().require_within(&held).unwrap();
        // The same route at the 4096 chunk and every smaller row count.
        for rows in [1, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            sym().requirement(&dims(rows)).unwrap().require_within(&held).unwrap();
        }
        // Fallback arms at <= 512 rows keep their F32 sizes inside the cap.
        for rows in [1, 8, 9, 64, 511, 512] {
            for r in [route(false), route(true), wmma(), wmma_unfused(), indexed()] {
                r.requirement(&dims(rows)).unwrap().require_within(&held).unwrap();
            }
        }
    }

    #[test]
    fn compact_scratch_refuses_fallback_arms_above_the_cap() {
        let held = compact_scratch(8192);
        for r in [route(false), route(true)] {
            assert!(
                r.requirement(&dims(513)).unwrap().require_within(&held).is_err(),
                "route {r:?} must be refused above 512 rows on compact scratch"
            );
        }
        // The 4096 plan too: the F32 grouped writer needs more than the cap.
        assert!(route(true).requirement(&dims(1024)).unwrap().require_within(&compact_scratch(4096)).is_err());
    }

    #[test]
    fn tensor_bytes_takes_the_smaller_of_shape_and_allocation() {
        let tensor = |bytes: usize, shape: &[usize]| GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(0x1000 as *mut std::ffi::c_void, bytes) },
            shape: shape.to_vec(),
            dtype: DType::F32,
        };
        assert_eq!(tensor_bytes(&tensor(64, &[16])), 64);
        // Shape claims more than the allocation holds.
        assert_eq!(tensor_bytes(&tensor(60, &[16])), 60);
        // Allocation larger than the logical view.
        assert_eq!(tensor_bytes(&tensor(128, &[16])), 64);
        assert_eq!(tensor_bytes(&tensor(128, &[usize::MAX, 2])), 0);
    }

    #[test]
    fn extents_that_overflow_are_errors() {
        let mut huge = dims(1);
        huge.rows = usize::MAX / 4;
        assert!(sym().requirement(&huge).is_err());
    }
}
