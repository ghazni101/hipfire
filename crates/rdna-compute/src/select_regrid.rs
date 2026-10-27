// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6d: re-gridded row selectors (gfx1201, gfx1151).
//!
//! `topk_values_batched_f32` (DFlash2 selector, raw top-K) and
//! `argmax_f32_batched` (verify argmax) launch one 256-thread block per row;
//! the top-K merges 256 x K candidates on a single thread. On exact gfx1201
//! and gfx1151 (opt out with `HIPFIRE_SELECT_REGRID_OFF=1`) both route through
//! `kernels/src/select_regrid.hip` instead: 1024 threads per row with wave32
//! butterflies. Outputs are byte-identical to the shipping kernels for every
//! input (see the kernel header for the argument; `test_select_regrid`
//! checks it, ties and NaNs included). Top-K rows whose leading K+1 values
//! tie are recomputed by `topk_values_fixup_f32`, the shipping kernel body
//! behind a marker check.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

/// Re-grid kernel source (both entries).
pub const SELECT_REGRID_SRC: &str = include_str!("../../../kernels/src/select_regrid.hip");
/// Compiled-module key for [`SELECT_REGRID_SRC`].
pub const SELECT_REGRID_MODULE: &str = "select_regrid";
/// Top-K re-grid entry.
pub const TOPK_VALUES_REGRID_SYMBOL: &str = "topk_values_regrid_f32";
/// Argmax re-grid entry.
pub const ARGMAX_REGRID_SYMBOL: &str = "argmax_f32_batched_regrid";
/// Shipping top-K body entered only for rows the re-grid marked (tie).
pub const TOPK_VALUES_FIXUP_SRC: &str = concat!(
    "#define HIPFIRE_TOPK_FIXUP_MARKED 1\n",
    "#define topk_logsumexp_batched_f32 topk_values_fixup_f32\n",
    include_str!("../../../kernels/src/topk_logsumexp_batched.hip")
);
/// Module key and symbol of [`TOPK_VALUES_FIXUP_SRC`].
pub const TOPK_VALUES_FIXUP_SYMBOL: &str = "topk_values_fixup_f32";
/// Threads per row block for both re-grid kernels.
pub const SELECT_REGRID_BLOCK: u32 = 1024;

impl Gpu {
    /// Exact gfx1201 or gfx1151 (the archs `test_select_regrid` byte-checks),
    /// not opted out, and no open Redline recording (the re-grid kernels are
    /// not in Redline's kernel tables).
    pub fn select_regrid_enabled(&self) -> bool {
        (self.arch_caps.is_gfx1201() || self.arch_caps.is_gfx1151())
            && !self.flags.select_regrid_off
            && !self.replay.is_recording()
    }

    /// Raw top-K over `[b x vocab]` rows: re-grid launch, then the fixup
    /// launch (a no-op for unmarked rows). Same stream, arguments and output
    /// contract as the shipping raw launch; `k` in `1..=16`.
    pub(crate) fn topk_values_regrid_f32(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_values: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            SELECT_REGRID_MODULE,
            SELECT_REGRID_SRC,
            TOPK_VALUES_REGRID_SYMBOL,
        )?;
        self.ensure_kernel(
            TOPK_VALUES_FIXUP_SYMBOL,
            TOPK_VALUES_FIXUP_SRC,
            TOPK_VALUES_FIXUP_SYMBOL,
        )?;
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut to = top_values.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut raw = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut to as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        self.launch_params_or_plan(
            TOPK_VALUES_REGRID_SYMBOL,
            [b as u32, 1, 1],
            [SELECT_REGRID_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut bl = KernargBlob::new();
                bl.push_ptr(lp);
                bl.push_ptr(ti);
                bl.push_ptr(to);
                bl.push_i32(vs);
                bl.push_i32(kk);
                bl
            },
        )?;
        // Shipping launch geometry and LDS for the fixup body.
        const MAX_K: u32 = 16;
        let nth: u32 = 256;
        let lds = (32 + nth * MAX_K * 2) * 4;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut to as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut raw as *mut _ as *mut c_void,
        ];
        self.launch_params_or_plan(
            TOPK_VALUES_FIXUP_SYMBOL,
            [b as u32, 1, 1],
            [nth, 1, 1],
            lds,
            &mut params,
            || {
                let mut bl = KernargBlob::new();
                bl.push_ptr(lp);
                bl.push_ptr(ti);
                bl.push_ptr(to);
                bl.push_i32(vs);
                bl.push_i32(kk);
                bl.push_i32(raw);
                bl
            },
        )
    }

    /// Batched argmax over `[batch x n]` rows with the shipping kernel's
    /// exact tie order, through `launch_maybe_blob` like the shipping launch.
    pub(crate) fn argmax_f32_batched_regrid(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            SELECT_REGRID_MODULE,
            SELECT_REGRID_SRC,
            ARGMAX_REGRID_SYMBOL,
        )?;
        let dp = data.buf.as_ptr();
        let rp = result.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ARGMAX_REGRID_SYMBOL,
            [batch_size as u32, 1, 1],
            [SELECT_REGRID_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(rp);
                b.push_i32(nn);
                b
            },
        )
    }

    /// The shipping raw top-K launch (kept callable for the parity gate).
    pub fn topk_values_batched_f32_shipping(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_values: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.launch_topk_batched_f32_shipping(logits, top_idx, top_values, vocab, k, b, true)
    }

    /// The shipping batched argmax launch (kept callable for the parity gate).
    pub fn argmax_f32_batched_shipping(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "argmax_f32_batched",
            kernels::ARGMAX_BATCHED_SRC,
            "argmax_f32_batched",
        )?;
        let dp = data.buf.as_ptr();
        let rp = result.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared = block_size * 8; // f32 + i32 per thread
        self.launch_maybe_blob(
            "argmax_f32_batched",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(rp);
                b.push_i32(nn);
                b
            },
        )
    }
}
