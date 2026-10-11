// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched draft forward over several lanes' blocks (the continuous-batching
//! store's DFlash lanes). See [`draft_forward_lanes`].

use super::{
    draft_dtod_at, draft_ffn_layer, gemm_dispatch, ordered_draft, ring_segments, upload_slice_i32, DflashConfig,
    DflashLayerWeights, DflashScratch, DflashWeights, DraftCtxMode,
};
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// One lane of [`draft_forward_lanes`]: its own draft scratch (context rings,
/// projection watermarks) and the singleton `draft_dflash_forward_rank`
/// geometry (`ctx_slice = None`).
pub struct DraftLane<'a> {
    pub scratch: &'a mut DflashScratch,
    /// Block rows of this lane (`<= scratch.max_block_size`).
    pub b: usize,
    /// Context rows `l` (`min(thlog.abs_positions().len(), position)`).
    pub ctx_len: usize,
    /// `[b]` absolute query positions.
    pub positions_q: &'a [i32],
    /// `[l + b]` absolute key positions.
    pub positions_k: &'a [i32],
}

/// Per-lane geometry of [`draft_forward_lanes`] (the singleton's per-call
/// locals of `draft_forward_opts`).
struct LaneGeo {
    l: usize,
    b: usize,
    /// First row of this lane in the shared row scratch.
    off: usize,
    swa_w: usize,
    full_w: usize,
    windowed: bool,
    all_sliding_windowed: bool,
    pos_base: usize,
    /// `thlog.proj_cached_rows()` at entry.
    cached_rows: usize,
}

/// The key positions a lane uploads: from `l - w_full` on in windowed mode
/// (the windowed `positions_k` holds the full-window span), the whole
/// `[l + b]` array in legacy mode.
fn lane_key_positions<'p>(ls: &DflashScratch, positions_k: &'p [i32], l: usize) -> &'p [i32] {
    match ls.ctx_mode {
        DraftCtxMode::Legacy => positions_k,
        DraftCtxMode::Windowed { w_full, .. } => &positions_k[l.saturating_sub(w_full)..],
    }
}

/// Context fill of one layer of one lane: the incremental K/V projection of
/// the rows past the layer's watermark into the lane's own K/V ring. Verbatim
/// the context block of `draft_forward_opts` (`q_dummy` is the ignored query
/// operand of the K-only RoPE). The FWHT rotation temporaries of the two
/// GEMMs come from `shared` (see [`draft_forward_lanes`]), never from the
/// lane: a lane scratch may be allocated without them.
#[allow(clippy::too_many_arguments)]
fn lane_ctx_fill(
    gpu: &mut Gpu,
    cfg: &DflashConfig,
    layer: &DflashLayerWeights,
    ls: &DflashScratch,
    shared: &DflashScratch,
    q_dummy: &GpuTensor,
    li: usize,
    g: &LaneGeo,
) -> HipResult<()> {
    let h = cfg.hidden;
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;
    let l = g.l;
    let (swa_w, full_w) = (g.swa_w, g.full_w);
    let is_last_layer = li + 1 == cfg.n_layers;
    let is_full_layer = !g.all_sliding_windowed && is_last_layer;
    let layer_w = if is_full_layer { full_w } else { swa_w };
    let (k_cache_layer, v_cache_layer) = if is_full_layer && g.windowed {
        (
            ls.k_full_cached.as_ref().expect("windowed k_full"),
            ls.v_full_cached.as_ref().expect("windowed v_full"),
        )
    } else {
        (&ls.k_ctx_cached[li], &ls.v_ctx_cached[li])
    };
    let span_start = l.saturating_sub(layer_w);
    let wm = if is_full_layer {
        ls.thlog.full_cached_rows()
    } else {
        g.cached_rows
    };
    let fill_start = wm.max(span_start).max(l.saturating_sub(swa_w));
    let mut row = fill_start;
    while row < l {
        let step = (swa_w.saturating_sub(if swa_w == usize::MAX { 0 } else { row % swa_w }))
            .min(layer_w.saturating_sub(if layer_w == usize::MAX {
                0
            } else {
                row % layer_w
            }))
            .min(l - row);
        let p_slot = if swa_w == usize::MAX { row } else { row % swa_w };
        let c_slot = if layer_w == usize::MAX { row } else { row % layer_w };
        let thp_slice = ls.target_hidden_proj.sub_offset(p_slot * h, step * h);
        let k_slot = k_cache_layer.sub_offset(c_slot * kvd, step * kvd);
        let v_slot = v_cache_layer.sub_offset(c_slot * kvd, step * kvd);
        gemm_dispatch(
            gpu,
            &thp_slice,
            &layer.wk,
            &k_slot,
            step,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            &thp_slice,
            &layer.wv,
            &v_slot,
            step,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gpu.rmsnorm_batched(&k_slot, &layer.k_norm, &k_slot, step * cfg.n_kv_heads, hd, eps)?;
        let fill_positions = ls.positions_k.sub_offset(row - g.pos_base, step);
        gpu.rope_batched_f32(q_dummy, &k_slot, &fill_positions, 0, cfg.n_kv_heads, hd, theta, step)?;
        row += step;
    }
    Ok(())
}

/// Context-K/V concat and attention of one layer of one lane: the lane's own
/// K/V ring plus its noise rows (taken from the shared row scratch) into the
/// lane's concat buffers, then attention of its query rows into the shared
/// attention output. Verbatim the concat/attention block of
/// `draft_forward_opts`.
fn lane_attention(
    gpu: &mut Gpu,
    cfg: &DflashConfig,
    ls: &DflashScratch,
    shared: &DflashScratch,
    li: usize,
    g: &LaneGeo,
    ordered: bool,
) -> HipResult<()> {
    let kvd = cfg.kv_dim();
    let qd = cfg.q_dim();
    let hd = cfg.head_dim;
    let (b, l) = (g.b, g.l);
    let (swa_w, full_w) = (g.swa_w, g.full_w);
    let is_last_layer = li + 1 == cfg.n_layers;
    let is_full_layer = !g.all_sliding_windowed && is_last_layer;
    let layer_w = if is_full_layer { full_w } else { swa_w };
    let (k_cache_layer, v_cache_layer) = if is_full_layer && g.windowed {
        (
            ls.k_full_cached.as_ref().expect("windowed k_full"),
            ls.v_full_cached.as_ref().expect("windowed v_full"),
        )
    } else {
        (&ls.k_ctx_cached[li], &ls.v_ctx_cached[li])
    };
    let (k_cat_l, v_cat_l) = if is_full_layer && g.windowed {
        (
            ls.k_cat_full.as_ref().expect("windowed k_cat_full"),
            ls.v_cat_full.as_ref().expect("windowed v_cat_full"),
        )
    } else {
        (&ls.k_cat, &ls.v_cat)
    };
    let span_start = l.saturating_sub(layer_w);
    let span = l - span_start;
    let noise_bytes = (b * kvd) * 4;
    let mut cat_off = 0usize;
    for (_row0, slot0, len) in ring_segments(span_start, l, layer_w) {
        let seg_bytes = len * kvd * 4;
        draft_dtod_at(gpu, ordered, &k_cat_l.buf, cat_off, &k_cache_layer.buf, slot0 * kvd * 4, seg_bytes)?;
        draft_dtod_at(gpu, ordered, &v_cat_l.buf, cat_off, &v_cache_layer.buf, slot0 * kvd * 4, seg_bytes)?;
        cat_off += seg_bytes;
    }
    draft_dtod_at(gpu, ordered, &k_cat_l.buf, cat_off, &shared.k_noise.buf, g.off * kvd * 4, noise_bytes)?;
    draft_dtod_at(gpu, ordered, &v_cat_l.buf, cat_off, &shared.v_noise.buf, g.off * kvd * 4, noise_bytes)?;
    let q_view = shared.q.sub_offset(g.off * qd, b * qd);
    let out_view = shared.attn_out.sub_offset(g.off * qd, b * qd);
    let is_swa_layer = g.windowed && (g.all_sliding_windowed || !is_full_layer);
    if is_swa_layer {
        gpu.attention_dflash_sliding_f32(
            &q_view,
            k_cat_l,
            v_cat_l,
            &out_view,
            b,
            span + b,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
            span,
            swa_w,
        )?;
    } else {
        use crate::llama::{attention_family, DispatchCtx, FullAttnParams, KernelKey};
        let ctx = DispatchCtx::new(gpu);
        let family = attention_family();
        family
            .run_full_attention(
                &ctx,
                gpu,
                &FullAttnParams {
                    key: KernelKey::AttnFullF32,
                    q: &q_view,
                    k: k_cat_l,
                    v: v_cat_l,
                    out: &out_view,
                    n: b,
                    seq_len: span + b,
                    n_heads: cfg.n_heads,
                    n_kv_heads: cfg.n_kv_heads,
                    head_dim: hd,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}

/// Batched draft forward over several lanes' blocks (one forward of the
/// draft model over all lanes' rows). Row-local stages — every draft weight
/// GEMM (qkv, wo, FFN, conv kernel projections), the norms, the query/noise-K
/// RoPE (per-row positions) — run ONCE over the concatenated rows in
/// `shared`; lane-state stages — the context fill into each lane's own K/V
/// ring, the K/V concat, the attention, and the DFlash2 convolutions (causal
/// across the rows of ONE block) — run per lane on that lane's rows.
///
/// `shared.x[off_i .. off_i + b_i)` MUST hold lane `i`'s noise embeddings on
/// entry (rows packed in lane order); the final-normed hidden rows of every
/// lane are left in `shared.x` at the same offsets. Each lane's draft scratch
/// (K/V rings, `target_hidden_proj`, thlog watermarks) ends bit-identical to
/// a singleton `draft_forward_opts` of that lane iff every shared GEMM is
/// row-count independent for the row counts used (`sum b_i`); the caller owns
/// that proof (exact gfx1201 MQ4 v2 below 64 rows). Requires the non-fused
/// (non-gfx1100) draft route, `ctx_slice = None` geometry and no FFN graph.
///
/// FWHT rotation scratch: the per-lane GEMMs (the `fc` projection and the
/// context-fill `wk`/`wv`) rotate their activation into `shared.mq_x_rot`
/// / `shared.mq_x_rot_f16`, NOT into the lane's own buffers (a lane may be
/// allocated without them). This is byte-exact: the rotation buffer is a
/// per-`gemm_dispatch`-call temporary (each call writes the rotated rows,
/// the GEMM reads them, in the same call and on the single stream, and
/// nothing reads it afterwards), and `gemm_dispatch` chunks by the element
/// count of the buffer, which is identical for every `DflashScratch`
/// (`MQ_X_ROT_CHUNK_ROWS × max(inter, q_dim, num_extract × hidden)`,
/// independent of block/context size) — so chunking, kernels and row order
/// are unchanged. A lane that does own a rotation buffer must match the
/// shared one in shape (checked), and MQ draft weights with no shared
/// rotation buffer are an error, not a panic.
pub fn draft_forward_lanes(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    shared: &mut DflashScratch,
    lanes: &mut [DraftLane<'_>],
) -> HipResult<()> {
    if lanes.is_empty() {
        return Ok(());
    }
    // Checked before any launch: the per-lane GEMMs borrow the shared
    // rotation scratch (see the doc comment).
    if weights.has_mq && shared.mq_x_rot.is_none() {
        return Err(hip_bridge::HipError::new(
            0,
            "draft_forward_lanes: MQ draft weights need the shared FWHT rotation scratch (shared.mq_x_rot is None)",
        ));
    }
    if let Some(sh) = shared.mq_x_rot.as_ref() {
        if lanes
            .iter()
            .any(|ln| ln.scratch.mq_x_rot.as_ref().is_some_and(|t| t.shape != sh.shape))
        {
            return Err(hip_bridge::HipError::new(
                0,
                "draft_forward_lanes: a lane's own FWHT rotation scratch differs in size from the shared one",
            ));
        }
    }
    if gpu.draft_collapse_fused_enabled() {
        return Err(hip_bridge::HipError::new(
            0,
            "draft_forward_lanes: the fused gfx1100 draft route is row-count dependent",
        ));
    }
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;
    let inter = cfg.intermediate;
    let total: usize = lanes.iter().map(|ln| ln.b).sum();
    assert!(total <= shared.max_block_size, "lane rows > shared scratch max");

    // ── Per-lane geometry, key positions, context projection ─────────────
    let ordered = ordered_draft();
    // Owned staging (capacity `max_block_size >= total`, never grown).
    shared.lane_positions_q.clear();
    for lane in lanes.iter() {
        shared.lane_positions_q.extend_from_slice(lane.positions_q);
    }
    if ordered {
        // Every host position upload precedes the draft body, so no
        // synchronous H2D drains the queue between the lane fc GEMMs. The fc
        // GEMMs and hidden norms read no positions; every position reader
        // (context RoPE, query/noise RoPE) still runs after its upload.
        for lane in lanes.iter() {
            assert_eq!(lane.positions_k.len(), lane.ctx_len + lane.b, "positions_k size");
            let ls = &*lane.scratch;
            upload_slice_i32(gpu, &ls.positions_k, lane_key_positions(ls, lane.positions_k, lane.ctx_len))?;
        }
        upload_slice_i32(gpu, &shared.positions_q, &shared.lane_positions_q)?;
    }
    let mut geos: Vec<LaneGeo> = Vec::with_capacity(lanes.len());
    let mut off = 0usize;
    for lane in lanes.iter_mut() {
        let (b, l) = (lane.b, lane.ctx_len);
        let ls = &mut *lane.scratch;
        assert!(b <= ls.max_block_size, "block_size > scratch max");
        assert!(l <= ls.max_ctx_len, "ctx_len > scratch max");
        assert_eq!(lane.positions_q.len(), b, "positions_q size");
        assert_eq!(lane.positions_k.len(), l + b, "positions_k size");
        let (swa_w, full_w) = match ls.ctx_mode {
            DraftCtxMode::Legacy => (usize::MAX, usize::MAX),
            DraftCtxMode::Windowed { w, w_full } => (w, w_full),
        };
        let windowed = !matches!(ls.ctx_mode, DraftCtxMode::Legacy);
        let pos_base = if windowed { l.saturating_sub(full_w) } else { 0 };
        if windowed && full_w != swa_w && l > swa_w && ls.thlog.full_cached_rows() < l.saturating_sub(swa_w) {
            static BACKFILL_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !BACKFILL_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[dflash] windowed: last-layer backfill watermark {} < l−W {} — stale \
                     out-of-window K/V (acceptance degraded; output still verify-exact). \
                     Call draft_seed_backfill after cold seeds.",
                    ls.thlog.full_cached_rows(),
                    l.saturating_sub(swa_w)
                );
            }
        }
        if !ordered {
            upload_slice_i32(gpu, &ls.positions_k, lane_key_positions(ls, lane.positions_k, l))?;
        }
        let cached_rows = ls.thlog.proj_cached_rows();
        let fc_start = cached_rows.max(l.saturating_sub(swa_w));
        if l.saturating_sub(fc_start) > 0 {
            for (_row0, slot0, len) in ring_segments(fc_start, l, swa_w) {
                let th_slice = ls.target_hidden.sub_offset(slot0 * ne * h, len * ne * h);
                let thp_slice = ls.target_hidden_proj.sub_offset(slot0 * h, len * h);
                // Per-lane fc projection (`len` rows): singleton route, outside
                // the batched 64-row scope.
                let scoped = gpu.draft_rows64_enter(false);
                let fc = gemm_dispatch(
                    gpu,
                    &th_slice,
                    &weights.fc,
                    &thp_slice,
                    len,
                    shared.mq_x_rot.as_ref(),
                    shared.mq_x_rot_f16.as_ref(),
                );
                gpu.draft_rows64_exit(scoped);
                fc?;
                gpu.rmsnorm_batched(&thp_slice, &weights.hidden_norm, &thp_slice, len, h, eps)?;
            }
        }
        geos.push(LaneGeo {
            l,
            b,
            off,
            swa_w,
            full_w,
            windowed,
            all_sliding_windowed: windowed && cfg.all_layers_sliding,
            pos_base,
            cached_rows,
        });
        off += b;
    }
    if !ordered {
        upload_slice_i32(gpu, &shared.positions_q, &shared.lane_positions_q)?;
    }

    // ── Per-layer decoder ────────────────────────────────────────────────
    let rows = total;
    let conv_kgs = || {
        let k = cfg.conv_kernel_size.unwrap_or(2);
        let g = cfg.conv_group_size.unwrap_or(16);
        (k, g, h / g, 2 * k * (h / g))
    };
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];
        draft_dtod_at(gpu, ordered, &shared.residual.buf, 0, &shared.x.buf, 0, (rows * h) * 4)?;
        gpu.rmsnorm_batched(&shared.x, &layer.attn_norm, &shared.x_norm, rows, h, eps)?;

        // DFlash2 prepare convolution (causal across the rows of ONE block ⇒
        // one launch per lane; the kernel-coefficient GEMM is row-local).
        let conv = match (&layer.attn_conv_base, &layer.attn_conv_proj) {
            (Some(base), Some(proj)) => match (&shared.conv_dynamic, &shared.conv_temp) {
                (Some(dyn_buf), Some(tmp)) => Some((base, proj, dyn_buf, tmp)),
                _ => None,
            },
            _ => None,
        };
        let qkv_src: &GpuTensor = if let Some((base, proj, dyn_buf, tmp)) = conv {
            let (k, g, _groups, stride) = conv_kgs();
            let dyn_all = dyn_buf.sub_offset(0, rows * stride);
            gemm_dispatch(
                gpu,
                &shared.x_norm,
                proj,
                &dyn_all,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            for g_ in &geos {
                let inp = shared.x_norm.sub_offset(g_.off * h, g_.b * h);
                let out = tmp.sub_offset(g_.off * h, g_.b * h);
                let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                gpu.dynamic_causal_conv_f32(&inp, base, &dy, &out, g_.b, h, k, g, stride, 0)?;
            }
            tmp
        } else {
            &shared.x_norm
        };
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wq,
            &shared.q,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wk,
            &shared.k_noise,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wv,
            &shared.v_noise,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;

        // Context K/V of every lane into its own ring.
        // Per-lane context fill is a lane-local GEMM of arbitrary row count: it
        // keeps the singleton's route even inside the batched 64-row scope.
        let scoped = gpu.draft_rows64_enter(false);
        let mut filled = Ok(());
        for (lane, g_) in lanes.iter().zip(&geos) {
            filled = lane_ctx_fill(gpu, cfg, layer, lane.scratch, shared, &shared.q, li, g_);
            if filled.is_err() {
                break;
            }
        }
        gpu.draft_rows64_exit(scoped);
        filled?;

        gpu.rmsnorm_batched(&shared.q, &layer.q_norm, &shared.q, rows * cfg.n_heads, hd, eps)?;
        gpu.rmsnorm_batched(&shared.k_noise, &layer.k_norm, &shared.k_noise, rows * cfg.n_kv_heads, hd, eps)?;
        gpu.rope_batched_f32(
            &shared.q,
            &shared.k_noise,
            &shared.positions_q,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
            theta,
            rows,
        )?;

        for (lane, g_) in lanes.iter().zip(&geos) {
            lane_attention(gpu, cfg, lane.scratch, shared, li, g_, ordered)?;
        }

        // wo lands in x (the non-fused route), then the DFlash2 finish
        // convolution per lane and the attention residual add.
        gemm_dispatch(
            gpu,
            &shared.attn_out,
            &layer.wo,
            &shared.x,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        let res_v = shared.residual.sub_offset(0, rows * h);
        let x_v = shared.x.sub_offset(0, rows * h);
        if let Some((base, _proj, dyn_buf, tmp)) = conv {
            let (k, g, groups, stride) = conv_kgs();
            let base_phase1 = base.sub_offset(k * h, k * h);
            for g_ in &geos {
                let inp = shared.x.sub_offset(g_.off * h, g_.b * h);
                let out = tmp.sub_offset(g_.off * h, g_.b * h);
                let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                gpu.dynamic_causal_conv_f32(&inp, &base_phase1, &dy, &out, g_.b, h, k, g, stride, k * groups)?;
            }
            let tmp_v = tmp.sub_offset(0, rows * h);
            gpu.add_f32(&res_v, &tmp_v, &x_v)?;
        } else {
            gpu.add_f32(&res_v, &x_v, &x_v)?;
        }

        // FFN tail.
        if let (Some(base), Some(proj), Some(dyn_buf), Some(tmp)) = (
            &layer.mlp_conv_base,
            &layer.mlp_conv_proj,
            &shared.conv_dynamic,
            &shared.conv_temp,
        ) {
            draft_dtod_at(gpu, ordered, &shared.residual.buf, 0, &shared.x.buf, 0, (rows * h) * 4)?;
            gpu.rmsnorm_batched(&shared.x, &layer.ffn_norm, &shared.x_norm, rows, h, eps)?;
            let (k, g, groups, stride) = conv_kgs();
            let dyn_all = dyn_buf.sub_offset(0, rows * stride);
            gemm_dispatch(
                gpu,
                &shared.x_norm,
                proj,
                &dyn_all,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            for g_ in &geos {
                let inp = shared.x_norm.sub_offset(g_.off * h, g_.b * h);
                let out = tmp.sub_offset(g_.off * h, g_.b * h);
                let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                gpu.dynamic_causal_conv_f32(&inp, base, &dy, &out, g_.b, h, k, g, stride, 0)?;
            }
            gemm_dispatch(
                gpu,
                tmp,
                &layer.w_gate,
                &shared.gate,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            gemm_dispatch(
                gpu,
                tmp,
                &layer.w_up,
                &shared.up,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            let gate_v = shared.gate.sub_offset(0, rows * inter);
            let up_v = shared.up.sub_offset(0, rows * inter);
            let gate_up_v = shared.gate_up.sub_offset(0, rows * inter);
            gpu.silu_mul_f32(&gate_v, &up_v, &gate_up_v)?;
            gemm_dispatch(
                gpu,
                &shared.gate_up,
                &layer.w_down,
                &shared.x,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            let base_phase1 = base.sub_offset(k * h, k * h);
            for g_ in &geos {
                let inp = shared.x.sub_offset(g_.off * h, g_.b * h);
                let out = tmp.sub_offset(g_.off * h, g_.b * h);
                let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                gpu.dynamic_causal_conv_f32(&inp, &base_phase1, &dy, &out, g_.b, h, k, g, stride, k * groups)?;
            }
            let tmp_v = tmp.sub_offset(0, rows * h);
            gpu.add_f32(&res_v, &tmp_v, &x_v)?;
        } else {
            draft_ffn_layer(gpu, layer, shared, rows, h, eps, false)?;
        }
    }

    // ── Final norm and projection-cache advance ──────────────────────────
    gpu.rmsnorm_batched(&shared.x, &weights.norm, &shared.x, rows, h, eps)?;
    for (lane, g_) in lanes.iter_mut().zip(&geos) {
        lane.scratch.thlog.mark_proj_cached(g_.l);
    }
    Ok(())
}
