// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode GEMV for [`crate::DType::IU8G256`] (qt 58).
//!
//! The MQ4V2 grid (FWHT-256, unsigned codes `0..=15`, one fp16
//! `(scale, zero)` pair per 128 weights) stored one byte per weight:
//!
//! ```text
//! [K code bytes][f16 scale, f16 zero] * (K / 128)
//! ```
//!
//! The kernel folds the stored affine grid in f32 — `(scale * code + zero) * x`
//! with `x` the already-rotated activation. That is the dot of the dequantized
//! row with `x`, the same contract as `iu_quant::dequant(Layout::Iu8)` ·
//! `gemv_iu8_dot`; the byte plane only removes the nibble expand. No arch gate:
//! there is no ISA-specific intrinsic.
//!
//! `gemv_iu8g256_residual` is the fused `y += W·x` twin (`y` is both in and
//! out), used by a body-dtype `ssm_out` decode step.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// `y = W x` for a planar IU8G256 weight. `x` is one FWHT-256-rotated
    /// f32 vector of length `k`. `k` must be a positive multiple of 256.
    pub fn gemv_iu8g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu8g256_launch("gemv_iu8g256", a, x, y, m, k)
    }

    /// `y += W x` for a planar IU8G256 weight (fused residual arm; `y` is
    /// both accumulated into and stored).
    pub fn gemv_iu8g256_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_iu8g256_launch("gemv_iu8g256_residual", a, x, y, m, k)
    }

    fn gemv_iu8g256_launch(
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
                "gemv_iu8g256: M > 0 and K % 256 == 0",
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel("gemv_iu8g256", kernels::GEMV_IU8G256_SRC, func)?;
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
