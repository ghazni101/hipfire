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
const SYM_MB4: &str = "gemm_iu8g32_residual_wmma_mb4";
const SYM_MB4M2: &str = "gemm_iu8g32_residual_wmma_mb4m2";

/// `HIPFIRE_IU8G32_MB4=0` keeps the 16-token-tile kernel (A/B control).
/// Unset keeps the 64-token fanout for every batch of 64+ rows.
fn iu8g32_mb4_enabled() -> bool {
    static V: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(
            hipfire_config::developer_var("HIPFIRE_IU8G32_MB4").as_deref(),
            Ok("0") | Ok("false")
        )
    });
    *V
}

/// 32-row-per-wave variant (two 16-row M tiles sharing one X fragment
/// stream). `HIPFIRE_IU8G32_M2=0` restores the 16-row fanout (A/B control).
fn iu8g32_m2_enabled() -> bool {
    static V: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(
            hipfire_config::developer_var("HIPFIRE_IU8G32_M2").as_deref(),
            Ok("0") | Ok("false")
        )
    });
    *V
}

impl Gpu {
    /// Whether [`Gpu::gemm_iu8g32_residual_wmma`] can run here (gfx11 WMMA).
    pub fn iu8g32_wmma_eligible(&self, k: usize) -> bool {
        k > 0 && k % 128 == 0 && matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
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
                &format!(
                    "gemm_iu8g32_residual_wmma: needs gfx1100/gfx1151 and K%128==0 (arch {}, K={k})",
                    self.arch
                ),
            ));
        }
        let mb4 = n >= 64 && iu8g32_mb4_enabled();
        let m2 = mb4 && m >= 32 && iu8g32_m2_enabled();
        let sym = if m2 {
            SYM_MB4M2
        } else if mb4 {
            SYM_MB4
        } else {
            SYM
        };
        if m2 {
            self.ensure_kernel(
                SYM_MB4M2,
                kernels::GEMM_IU8G32_RESIDUAL_WMMA_MB4M2_SRC,
                SYM_MB4M2,
            )?;
        } else if mb4 {
            self.ensure_kernel(SYM_MB4, kernels::GEMM_IU8G32_RESIDUAL_WMMA_MB4_SRC, SYM_MB4)?;
        } else {
            self.ensure_kernel(SYM, kernels::GEMM_IU8G32_RESIDUAL_WMMA_SRC, SYM)?;
        }
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
        let bytes = m * (k + k / 64) + n * k * 2 + n * m * 4 * 2;
        let grid_y = if mb4 { (n + 63) / 64 } else { (n + 15) / 16 };
        let grid_x = if m2 { (m + 31) / 32 } else { (m + 15) / 16 };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", sym, bytes);
        let result = self.launch_maybe_blob(
            sym,
            [grid_x as u32, grid_y as u32, 1],
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

    /// Minimal LDS-tiled W8A8 launch for a planar int8-per-128 weight row
    /// `[K codes][K/128 f16 scales]` and `block_i8_128` activations. The
    /// caller supplies the pre-quantized X pointer. This is the qt59 prefill
    /// path: the singleton batched body and the multi-slot body both route
    /// residual roles here when K % 256 == 0.
    pub fn gemm_iu8g32_w8a8_min(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if m == 0 || n == 0 || k % 128 != 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "gemm_iu8g32_w8a8_min needs M>0, N>0, K%128==0 and K%256==0",
            ));
        }
        let module = "gemm_iu8g32_w8a8_min";
        let entry = if add {
            "gemm_iu8g32_w8a8_min_add"
        } else {
            "gemm_iu8g32_w8a8_min_set"
        };
        self.ensure_kernel(module, kernels::GEMM_IU8G32_W8A8_MIN_SRC, entry)?;
        let mut a_p = a.buf.as_ptr();
        let mut xq_p = xq;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut add_val = i32::from(add);
        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xq_p as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];
        let grid = [m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1];
        let lds = 33_792u32;
        let bytes = m * (k + k / 64) + n * k + n * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", entry, bytes);
        let result = self.launch_maybe_blob(entry, grid, [512, 1, 1], lds, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_p);
            b.push_ptr(xq_p);
            b.push_ptr(y_p);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b.push_i32(n_val);
            b.push_i32(add_val);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
