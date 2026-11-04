// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode GEMV for [`crate::DType::IU8G128`] (qt 59), the lifted-role tier of
//! an `iu4` / `iu8` file.
//!
//! The grid is the Q8 grid — signed int8 codes, zero point 0, no FWHT — with
//! **one fp16 scale per 128** moved to the row tail:
//!
//! ```text
//! [K signed codes][K/128 f16 scales]       (K + K/64 bytes, 8.125 bpw)
//! ```
//!
//! The kernel (`gemv_iu8g32`, and the fused `gemv_iu8g32_residual`) folds
//! element for element like `gemv_q8_0`: one wave per row, thread `t` takes
//! element `t` of every 32-block (four blocks share one scale) and the warp
//! reduces by shuffle. The numeric contract is
//! `hipfire_runtime::iu_quant::dequant_q8` (the dot of the dequantized row
//! with `x`), which is also the CPU oracle.
//!
//! No arch gate: the math is f16-scale × int8 × f32 `x`, available everywhere.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// `y = W x` for a planar IU8G128 weight. `x` is the natural-basis
    /// activation (qt 59 is not rotated). `k` must be a positive multiple of
    /// 128 (one f16 scale per 128 at the row tail).
    pub fn gemv_iu8g32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu8g32_launch("gemv_iu8g32", a, x, y, m, k)
    }

    /// `y += W x` for a planar IU8G128 weight (fused residual arm; `y` is both
    /// accumulated into and stored).
    pub fn gemv_iu8g32_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu8g32_launch("gemv_iu8g32_residual", a, x, y, m, k)
    }

    /// Test/A-B entry: launch one named IU8G128 GEMV kernel (v1 or a v2
    /// `_rN` variant) with no automatic selection.
    pub fn gemv_iu8g32_named(
        &mut self,
        func: &'static str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu8g32_launch_exact(func, a, x, y, m, k)
    }

    /// `y[b] = W·x[b]` for `batch` activation columns sharing one IU8G128
    /// weight pass (`1..=4`). Multi-slot decode: a per-slot GEMV re-reads the
    /// whole weight matrix once per slot; here each weight row is read once
    /// for every column. Column `b` matches [`Self::gemv_iu8g32`] on that
    /// column to f32 rounding (different module → contraction may differ);
    /// two runs are bit-identical.
    ///
    /// `x` is `[batch × k]`, `y` is `[batch × m]`, both row-major. Requires a
    /// 16 B-aligned row stride (`(k + k/64) % 16 == 0`) for the `uint4`
    /// weight and activation chunks.
    pub fn gemv_iu8g32_xbatch(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        if m == 0 || k == 0 || k % 128 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu8g32_xbatch: M > 0 and K % 128 == 0",
            ));
        }
        if batch == 0 || batch > 4 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu8g32_xbatch: batch must be 1..=4",
            ));
        }
        let row_bytes = k + k / 64;
        if row_bytes % 16 != 0
            || (a.buf.as_ptr() as usize) % 16 != 0
            || (x.buf.as_ptr() as usize) % 16 != 0
        {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu8g32_xbatch: needs a 16 B-aligned row and activation base",
            ));
        }
        self.bind_thread()?;
        // Same module key as the v1/v2 kernels: one source, one compiled
        // module, and the registry entry that declares it.
        self.ensure_kernel(
            "gemv_iu8g32",
            kernels::GEMV_IU8G32_SRC,
            "gemv_iu8g32_xbatch",
        )?;
        let waves = m.div_ceil(4);
        let grid = [waves.div_ceil(4) as u32, 1, 1];
        let mut ap = a.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut bv = batch as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut bv as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_iu8g32_xbatch",
            grid,
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(bv);
                b
            },
        )
    }

    fn gemv_iu8g32_launch(
        &mut self,
        func: &'static str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        if m == 0 || k == 0 || k % 128 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu8g32: M > 0 and K % 128 == 0",
            ));
        }
        let func = iu8g32_pick(func, x, a, m, k);
        self.gemv_iu8g32_launch_exact(func, a, x, y, m, k)
    }

    fn gemv_iu8g32_launch_exact(
        &mut self,
        func: &'static str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_iu8g32", kernels::GEMV_IU8G32_SRC, func)?;
        let (grid, block) = iu8g32_geometry(func, m);
        let mut ap = a.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(func, grid, block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ap);
            b.push_ptr(xp);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b
        })
    }
}

/// `HIPFIRE_IU8_GEMV=v1` keeps the original one-byte-per-lane kernel (A/B);
/// anything else = v2 (f32 x). Read once.
fn iu8g32_v1_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_IU8_GEMV")
                .ok()
                .as_deref(),
            Some("v1")
        )
    })
}

/// v2 (R rows/wave, dwordx4 code loads) when rows and `x` are 16B aligned.
fn iu8g32_pick(
    func: &'static str,
    x: &GpuTensor,
    a: &GpuTensor,
    m: usize,
    k: usize,
) -> &'static str {
    let row_bytes = k + k / 64;
    let aligned = row_bytes % 16 == 0
        && (x.buf.as_ptr() as usize) % 16 == 0
        && (a.buf.as_ptr() as usize) % 16 == 0;
    if iu8g32_v1_forced() || !aligned {
        return func;
    }
    let add = func == "gemv_iu8g32_residual";
    let r = if m >= 32768 {
        4
    } else if m >= 4096 {
        2
    } else {
        1
    };
    match (add, r) {
        (false, 4) => "gemv_iu8g32_v2_r4",
        (false, 2) => "gemv_iu8g32_v2_r2",
        (false, _) => "gemv_iu8g32_v2_r1",
        (true, 4) => "gemv_iu8g32_v2_residual_r4",
        (true, 2) => "gemv_iu8g32_v2_residual_r2",
        (true, _) => "gemv_iu8g32_v2_residual_r1",
    }
}

fn iu8g32_geometry(func: &str, m: usize) -> ([u32; 3], [u32; 3]) {
    let r = if func.ends_with("_r4") {
        4
    } else if func.ends_with("_r2") {
        2
    } else if func.ends_with("_r1") {
        1
    } else {
        0
    };
    if r == 0 {
        return ([m as u32, 1, 1], [32, 1, 1]);
    }
    let waves = m.div_ceil(r);
    ([waves.div_ceil(4) as u32, 1, 1], [128, 1, 1])
}
