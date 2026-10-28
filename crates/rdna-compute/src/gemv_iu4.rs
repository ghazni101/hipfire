// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode GEMV for [`crate::DType::IU4G256`].
//!
//! One wave per output row. The weight dword is issued as an unsigned
//! `v_dot8_i32_iu4` operand. The numeric contract, including the signed-int4
//! activation quant and the per-eight-weight fold, is
//! `hipfire_runtime::iu_quant::gemv_iu4_dot8`.
//!
//! `gemv_iu4g256_residual` is the fused `y += W·x` twin for a body-dtype
//! `ssm_out` decode step.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// `y = W x` for a planar IU4G256 weight. `x` is one FWHT-256-rotated
    /// f32 vector of length `k`. `k` must be a positive multiple of 256.
    /// RDNA3 and RDNA4 only.
    pub fn gemv_iu4g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu4g256_launch("gemv_iu4g256", a, x, y, m, k)
    }

    /// `y += W x` for a planar IU4G256 weight (fused residual arm; `y` is
    /// both accumulated into and stored). RDNA3 and RDNA4 only.
    pub fn gemv_iu4g256_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu4g256_launch("gemv_iu4g256_residual", a, x, y, m, k)
    }

    fn gemv_iu4g256_launch(
        &mut self,
        func: &'static str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu4g256: M > 0 and K % 256 == 0",
            ));
        }
        if !self.arch_caps.has_iu4_wmma() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu4g256: v_dot8_i32_iu4 is the RDNA3/RDNA4 integer-4 dot",
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel("gemv_iu4g256", kernels::GEMV_IU4G256_SRC, func)?;
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
