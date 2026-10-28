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
        self.bind_thread()?;
        self.ensure_kernel("gemv_iu8g32", kernels::GEMV_IU8G32_SRC, func)?;
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
        self.launch_maybe_blob(func, [m as u32, 1, 1], [32, 1, 1], 0, &mut params, || {
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
