// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill) GEMM for [`crate::DType::IU4G256`].
//!
//! On gfx1100/gfx1151 every call is the MQ4V2 Q8_1 MMQ kernel built with
//! `HIPFIRE_MMQ_IU4L` (IU4 row addressing, same arithmetic): `X` is
//! quantized once to `block_q8_1_mmq` and the weight tiles are read straight
//! from the planar IU4 rows. A qt57 tensor that is the relayout of a qt44
//! tensor therefore gives the same `Y` bits as `gemm_mq4g256v2_mmq_*` on the
//! twin. Elsewhere, and while a graph is being captured or recorded, each
//! token is one f32 IU4 GEMV (the decode kernel), i.e. the per-token result.
//!
//! `X` is `[N, K]` row-major f32 (FWHT-256 rotated by the caller), `Y` is
//! `[N, M]` f32.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const MODULE: &str = "gemm_iu4g256_mmq";

/// One matrix of a shared-`X` batched call: weight, output, output rows.
pub type Iu4Proj<'a> = (&'a GpuTensor, &'a GpuTensor, usize);

/// `HIPFIRE_IU4_PREFILL=gemv`: skip the MMQ path (per-token f32 GEMV).
fn iu4_prefill_gemv() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_IU4_PREFILL").ok().as_deref(),
            Some("gemv")
        )
    })
}

impl Gpu {
    /// True when the IU4 MMQ path serves this call.
    pub fn iu4g256_mmq_eligible(&self, k: usize, n: usize) -> bool {
        n > 0
            && k % 256 == 0
            && matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
            && !self.graphs.capture_mode
            && !self.replay.is_recording()
            && !iu4_prefill_gemv()
    }

    /// `Y = W X` (or `Y += W X` when `add`) for one IU4G256 matrix.
    pub fn gemm_iu4g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.gemm_iu4g256_multi(&[(a, y, m)], x, k, n, add)
    }

    /// Several IU4G256 matrices over one `X` (gate/up, q/k/v,
    /// qkv/z/beta/alpha). `X` is quantized once.
    pub fn gemm_iu4g256_multi(
        &mut self,
        projs: &[Iu4Proj<'_>],
        x: &GpuTensor,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        if n == 0 || projs.is_empty() {
            return Ok(());
        }
        if self.iu4g256_mmq_eligible(k, n) {
            self.bind_thread()?;
            let xq = if self.flags.gfx11_mmq_x128_enabled() {
                self.ensure_q8_1_mmq_x128(x, n, k)?
            } else {
                self.ensure_q8_1_mmq_x(x, n, k)?
            };
            for &(a, y, m) in projs {
                self.gemm_iu4g256_mmq_prequant(a, xq, y, m, k, n, add)?;
            }
            return Ok(());
        }
        self.gemm_iu4g256_pertoken(projs, x, k, n, add)
    }

    /// The fallback: one f32 IU4 GEMV per token per matrix. Public for the
    /// parity test and `HIPFIRE_IU4_PREFILL=gemv` A/B runs.
    pub fn gemm_iu4g256_pertoken(
        &mut self,
        projs: &[Iu4Proj<'_>],
        x: &GpuTensor,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        let func = if add { "gemv_iu4g256_residual" } else { "gemv_iu4g256" };
        let xp = x.buf.as_ptr() as *mut u8;
        for &(a, y, m) in projs {
            let yp = y.buf.as_ptr() as *mut u8;
            for t in 0..n {
                // SAFETY: device pointer arithmetic inside [N,K] / [N,M] buffers.
                let (xt, yt) = unsafe { (xp.add(t * k * 4), yp.add(t * m * 4)) };
                self.gemv_iu4g256_launch_raw(
                    func,
                    a.buf.as_ptr(),
                    xt as *mut c_void,
                    yt as *mut c_void,
                    m,
                    k,
                )?;
            }
        }
        Ok(())
    }

    /// The IU4 MMQ launch over a pre-quantized `X` (`block_q8_1_mmq`, the
    /// layout of `ensure_q8_1_mmq_x{,128}` matching the x128 flag).
    pub fn gemm_iu4g256_mmq_prequant(
        &mut self,
        a: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        if !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") {
            return Err(hip_bridge::HipError::new(1, "IU4 Q8_1 MMQ requires exact gfx1100/gfx1151"));
        }
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(1, "IU4 Q8_1 MMQ requires M > 0 and K % 256 == 0"));
        }
        self.bind_thread()?;
        let full = m % 128 == 0 && batch_size % 128 == 0;
        let x128 = self.flags.gfx11_mmq_x128_enabled();
        let kernel_name = match (full, add, x128) {
            (true, true, false) => "gemm_iu4g256_mmq_full_add",
            (true, false, false) => "gemm_iu4g256_mmq_full_set",
            (false, _, false) => "gemm_iu4g256_mmq",
            (true, true, true) => "gemm_iu4g256_mmq_full_add_x128",
            (true, false, true) => "gemm_iu4g256_mmq_full_set_x128",
            (false, _, true) => "gemm_iu4g256_mmq_x128",
        };
        self.ensure_kernel(MODULE, kernels::GEMM_IU4G256_MMQ_SRC, kernel_name)?;
        let mut a_ptr = a.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = i32::from(add);
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];
        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 36;
        const MMQ_TILE_X_K: usize = 76;
        let row_tiles = m.div_ceil(MMQ_Y);
        let batch_tiles = batch_size.div_ceil(MMQ_X);
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
        let bytes = m * (k / 256) * crate::dispatch::MQ4V2_GROUP_BYTES + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 8, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
