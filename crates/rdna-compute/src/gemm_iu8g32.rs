// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill) residual GEMM for [`crate::DType::IU8G32`] (qt59) on gfx11.
//!
//! `Y[N, M] += W X`: the Q8_0 f16-WMMA residual kernel built with
//! `HIPFIRE_IU8G32L` (tail-scale row addressing). `X` is `[N, K]` f32 in the
//! natural basis (qt59 is not rotated); it is converted to f16 once.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const SYM: &str = "gemm_iu8g32_residual_wmma";

impl Gpu {
    /// Whether [`Gpu::gemm_iu8g32_residual_wmma`] can run here (gfx11 WMMA).
    pub fn iu8g32_wmma_eligible(&self, k: usize) -> bool {
        k > 0 && k % 32 == 0 && matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
    }

    /// `Y += W X` for one IU8G32 matrix (`m` rows, `k` cols, `n` tokens).
    pub fn gemm_iu8g32_residual_wmma(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.iu8g32_wmma_eligible(k) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("gemm_iu8g32_residual_wmma: needs gfx1100/gfx1151 and K%32==0 (arch {}, K={k})", self.arch),
            ));
        }
        self.ensure_kernel(SYM, kernels::GEMM_IU8G32_RESIDUAL_WMMA_SRC, SYM)?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;
        let mut a_p = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let bytes = m * (k / 32) * 34 + n * k * 2 + n * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", SYM, bytes);
        let result = self.launch_maybe_blob(
            SYM,
            [((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
