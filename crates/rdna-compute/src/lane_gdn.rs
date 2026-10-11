// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launchers for the lane-batched DeltaNet (GDN) verify preamble of the
//! DFlash continuous-batching trunk (exact gfx1201).
//!
//! The multi-request verify chunk keeps every lane's rows in ONE shared
//! scratch (lane `i` owns rows `[row0_i, row0_i + n_i)`), so the per-lane
//! `sigmoid/alpha-gate + conv1d + QK-norm + GDN` sequence (4 launches per lane
//! per DeltaNet layer) collapses to one launch per stage per layer:
//!
//! * the two row-local kernels (`fused_sigmoid_alpha_gate_f32_batched`,
//!   `fused_qk_l2_norm_scale_interleave_f32_batched`) run once over all rows
//!   (existing launchers, unchanged);
//! * [`Gpu::conv1d_silu_split_f32_lanes`] — the per-lane conv rings ride in a
//!   by-value kernarg lane table (`kernels/src/conv1d_silu_split_lanes.hip`);
//! * `Gpu::gated_delta_net_q8_fast_ml` over a device [`GdnLayerTable`]
//!   (one row per lane, uploaded by [`Gpu::upload_gdn_lane_table`] only when
//!   it changes).
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::conv1d_silu_split_f32_lanes`].
pub const CONV1D_SILU_SPLIT_LANES_SRC: &str =
    include_str!("../../../kernels/src/conv1d_silu_split_lanes.hip");
/// Module key and device symbol.
pub const CONV1D_SILU_SPLIT_LANES_SYMBOL: &str = "conv1d_silu_split_f32_lanes";
/// Lanes per launch (`CONV_LANES_MAX` in the kernel); also the row padding of
/// the per-layer device GDN lane table.
pub const CONV_LANES_MAX: usize = 16;

/// `#[repr(C)]` mirror of `struct ConvLaneTable` (the by-value kernel
/// argument): 16 x u64 state pointers, 16 x i32 row0, 16 x i32 n_tokens
/// (128 + 64 + 64 = 256 B, 8-aligned).
#[repr(C)]
struct ConvLaneTableArgs {
    state: [u64; CONV_LANES_MAX],
    row0: [i32; CONV_LANES_MAX],
    n_tokens: [i32; CONV_LANES_MAX],
}

impl Gpu {
    /// Conv1d + SiLU + Q/K/V split of `lanes.len()` (1..=16) lanes in ONE
    /// launch. `lanes[i] = (conv ring device pointer, first row, rows)`; the
    /// buffers are the FULL shared row-major scratch buffers. Byte-identical
    /// to one `conv1d_silu_split_f32_n` per lane over that lane's row view
    /// (see the kernel header).
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_f32_lanes(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        lanes: &[(u64, usize, usize)],
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        assert!(
            (1..=CONV_LANES_MAX).contains(&lanes.len()),
            "conv1d_silu_split_f32_lanes: {} lanes outside 1..={CONV_LANES_MAX}",
            lanes.len()
        );
        self.bind_thread()?;
        self.ensure_kernel(
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
            CONV1D_SILU_SPLIT_LANES_SRC,
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
        )?;
        let mut args = ConvLaneTableArgs {
            state: [0; CONV_LANES_MAX],
            row0: [0; CONV_LANES_MAX],
            n_tokens: [0; CONV_LANES_MAX],
        };
        for (i, &(state, row0, n)) in lanes.iter().enumerate() {
            args.state[i] = state;
            args.row0[i] = row0 as i32;
            args.n_tokens[i] = n as i32;
        }
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut args as *mut _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        // Same block / channel grid as `conv1d_silu_split_f32_n`; grid.y = lane.
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let blob_args = &args;
        self.launch_maybe_blob(
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
            [grid, lanes.len() as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                // The 256 B struct is 8-aligned: it starts at offset 40, right
                // after the five pointers.
                for v in &blob_args.state {
                    b.push_u64(*v);
                }
                for v in blob_args.row0.iter().chain(&blob_args.n_tokens) {
                    b.push_i32(*v);
                }
                b.push_i32(kd);
                b.push_i32(vd);
                b
            },
        )
    }

    /// May the lane-batched GDN verify route run for a chunk of DeltaNet
    /// layers with these head counts / dims on this device and process state?
    /// Exact gfx1201; the fast (single-end requant) GDN kernel; head dims 128;
    /// a GQA layout (`n_key_heads < n_v_heads`, exact division); no graph
    /// capture, replay recording or G0 observation (the multi-lane GDN kernel
    /// reads its pointers from device memory, which a recording could not
    /// attribute). The caller adds the per-chunk conditions (every lane
    /// carries a tape, ChainVerify fusion, equal row counts, ...).
    pub fn lane_gdn_batch_admitted(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> bool {
        self.arch_caps.is_gfx1201()
            && !crate::norm::dn_requant_per_token()
            && key_head_dim == 128
            && value_head_dim == 128
            && n_key_heads > 0
            && n_key_heads < n_v_heads
            && n_v_heads % n_key_heads == 0
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.replay.is_g0_observing()
    }

    /// Stage `bytes` into the device GDN lane table `dst` on the active
    /// stream (stream-ordered after every kernel already queued, so the
    /// previous step's recurrence launches finish reading the old rows
    /// first). The copy is followed by a stream synchronize so the host
    /// source may be dropped or rewritten immediately (a pageable-source
    /// async copy gives no earlier guarantee); callers upload only when the
    /// table contents change. Refused during graph capture.
    pub fn upload_gdn_lane_table(&mut self, dst: &GpuTensor, bytes: &[u8]) -> HipResult<()> {
        self.bind_thread()?;
        if self.graphs.capture_mode {
            return Err(HipError::new(0, "upload_gdn_lane_table: table staging during graph capture"));
        }
        if bytes.len() > dst.buf.size() {
            return Err(HipError::new(
                0,
                &format!(
                    "upload_gdn_lane_table: table holds {} B, needs {} B",
                    dst.buf.size(),
                    bytes.len()
                ),
            ));
        }
        match self.active_stream.as_ref() {
            Some(stream) => {
                self.hip.memcpy_htod_async(&dst.buf, bytes, stream)?;
                self.hip.stream_synchronize(stream)
            }
            None => self.hip.memcpy_htod(&dst.buf, bytes),
        }
    }
}
