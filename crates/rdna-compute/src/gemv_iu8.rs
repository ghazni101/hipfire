// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode GEMV for [`crate::DType::IU8G32`] (qt 59), the lifted-role tier of
//! an `iu4` / `iu8` file.
//!
//! The grid is Q8F16's — signed int8 codes, one fp16 scale per 32, zero point
//! 0, no FWHT — with the scales moved to the row tail:
//!
//! ```text
//! [K signed codes][f16 scale] * (K / 32)
//! ```
//!
//! The kernel (`gemv_iu8g32`, and the fused `gemv_iu8g32_residual`) folds
//! element for element like `gemv_q8_0`: one wave per row, thread `t` takes
//! element `t` of every 32-block and the warp reduces by shuffle. The numeric
//! contract is `hipfire_runtime::iu_quant::dequant_q8` (the dot of the
//! dequantized row with `x`), which is also the CPU oracle.
//!
//! No arch gate: the math is f16-scale × int8 × f32 `x`, available everywhere.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// `y = W x` for a planar IU8G32 weight. `x` is the natural-basis
    /// activation (qt 59 is not rotated). `k` must be a positive multiple of
    /// 32.
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

    /// `y += W x` for a planar IU8G32 weight (fused residual arm; `y` is both
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

    /// Test/A-B entry: launch one named IU8G32 GEMV kernel (v1 or a v2
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

    fn gemv_iu8g32_launch(
        &mut self,
        func: &'static str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        if m == 0 || k == 0 || k % 32 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu8g32: M > 0 and K % 32 == 0",
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

/// `HIPFIRE_IU8_GEMV=v1` keeps the original one-byte-per-lane kernel (A/B).
fn iu8g32_v1_forced() -> bool {
    iu8g32_mode() == 1
}

/// `HIPFIRE_IU8_GEMV`: `v1` (original wave-per-row), `dot4` (W8A8 int8
/// activation per 32, `v_dot4_i32_iu8`), anything else = v2 (f32 x).
/// 0 = v2, 1 = v1, 2 = dot4. Read once.
fn iu8g32_mode() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| match hipfire_config::developer_var("HIPFIRE_IU8_GEMV").ok().as_deref() {
        Some("v1") => 1,
        Some("dot4") => 2,
        _ => 0,
    })
}

/// v2 (R rows/wave, dwordx4 code loads) when rows and `x` are 16B aligned.
fn iu8g32_pick(func: &'static str, x: &GpuTensor, a: &GpuTensor, m: usize, k: usize) -> &'static str {
    let row_bytes = k + k / 16;
    let aligned = row_bytes % 16 == 0
        && (x.buf.as_ptr() as usize) % 16 == 0
        && (a.buf.as_ptr() as usize) % 16 == 0;
    if iu8g32_v1_forced() || !aligned {
        return func;
    }
    let add = func == "gemv_iu8g32_residual";
    let r = if m >= 32768 { 4 } else if m >= 4096 { 2 } else { 1 };
    if iu8g32_mode() == 2 {
        return match (add, r) {
            (false, 4) => "gemv_iu8g32_a8_r4",
            (false, 2) => "gemv_iu8g32_a8_r2",
            (false, _) => "gemv_iu8g32_a8_r1",
            (true, 4) => "gemv_iu8g32_a8_residual_r4",
            (true, 2) => "gemv_iu8g32_a8_residual_r2",
            (true, _) => "gemv_iu8g32_a8_residual_r1",
        };
    }
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
    let r = if func.ends_with("_r4") { 4 } else if func.ends_with("_r2") { 2 } else if func.ends_with("_r1") { 1 } else { 0 };
    if r == 0 {
        return ([m as u32, 1, 1], [32, 1, 1]);
    }
    let waves = m.div_ceil(r);
    ([waves.div_ceil(4) as u32, 1, 1], [128, 1, 1])
}
