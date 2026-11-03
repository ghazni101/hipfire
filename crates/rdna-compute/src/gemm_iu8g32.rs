// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill) GEMMs for [`crate::DType::IU8G32`] (qt59) on gfx11.
//!
//! Two kernels compute `Y[N, M] += W X` for the compact `[K i8][K/128 f16]`
//! row. `X` is `[N, K]` f32 in the natural basis (qt59 is not rotated).
//!
//! - **f16 twin** (`gemm_iu8g32_residual_wmma` + mb4/mb4m2): the Q8_0
//!   f16-WMMA residual kernel built with `HIPFIRE_IU8G32L` (tail-scale row
//!   addressing); `X` is converted to f16. Needs `K % 128 == 0`.
//! - **W8A8** (`gemm_iu8g32_w8a8_min`): a 128x128 LDS tile on
//!   `v_wmma_i32_16x16x16_iu8`; `X` is quantized to int8 per 128. Needs
//!   `K % 256 == 0`.
//!
//! [`Gpu::iu8g32_gemm_route`] is the one selector between them and
//! [`Gpu::gemm_iu8g32_residual`] the one entry point, so the singleton
//! batched body and the multi-slot body cannot disagree on which arithmetic
//! a prompt chunk gets. Arch admission is
//! [`crate::arch_caps::supports_iu8g32_gemm`].

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const SYM: &str = "gemm_iu8g32_residual_wmma";
const SYM_MB4: &str = "gemm_iu8g32_residual_wmma_mb4";
const SYM_MB4M2: &str = "gemm_iu8g32_residual_wmma_mb4m2";

/// Fewest token rows that take the W8A8 route. Measured on gfx1100 with
/// `iu8g32_route_crossover` (`hipfire-runtime/tests/iu8g32_wmma_gemm_gpu.rs`,
/// 27B `wo` M=5120/K=6144 and `w_down` M=5120/K=17408, activation conversion
/// charged per call, two passes): the f16 twin is 1.3-2.4x faster at n<=32,
/// the two are at parity at n=40-48, and W8A8 is 1.26-1.36x faster at n=56-63
/// and 1.9-2.8x from n=64 (where the twin's mb4 fanout starts). Below the
/// threshold the 128-row W8A8 tile idles and its activation quant buys
/// nothing, so short chunks and DFlash-width batches keep the f16 twin.
pub const IU8G32_W8A8_MIN_ROWS: usize = 48;

/// Which qt59 batched GEMM a shape takes ([`Gpu::iu8g32_gemm_route`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iu8g32GemmRoute {
    /// `gemm_iu8g32_w8a8_min`: int8 activations, IU8 WMMA.
    W8A8,
    /// `gemm_iu8g32_residual_wmma*`: f16 activations, f16 WMMA.
    F16Twin,
}

/// `HIPFIRE_IU8G32_W8A8=0` keeps every qt59 GEMM on the f16 twin (A/B
/// control for both the singleton and the multi-slot prefill).
fn iu8g32_w8a8_enabled() -> bool {
    static V: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(
            hipfire_config::developer_var("HIPFIRE_IU8G32_W8A8").as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        )
    });
    *V
}

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

/// Whether a qt59 matrix of width `k` has a batched GEMM on `arch` (the f16
/// twin, which every other qt59 GEMM route falls back to). Admission
/// predicates that hold only an arch name call this; launchers call
/// [`Gpu::iu8g32_wmma_eligible`], which is the same rule.
pub fn iu8g32_gemm_admits(arch: &str, k: usize) -> bool {
    k > 0 && k % 128 == 0 && crate::arch_caps::supports_iu8g32_gemm(arch)
}

impl Gpu {
    /// Whether [`Gpu::gemm_iu8g32_residual_wmma`] (the f16 twin) can run
    /// here. Any qt59 batched GEMM is admissible exactly when this holds.
    pub fn iu8g32_wmma_eligible(&self, k: usize) -> bool {
        iu8g32_gemm_admits(&self.arch, k)
    }

    /// Whether [`Gpu::gemm_iu8g32_w8a8_min`] can run here.
    pub fn iu8g32_w8a8_eligible(&self, k: usize) -> bool {
        k > 0 && k % 256 == 0 && crate::arch_caps::supports_iu8g32_gemm(&self.arch)
    }

    /// The qt59 batched GEMM for `n` token rows of a `K`-wide matrix, or
    /// `None` when no qt59 GEMM runs here (the caller must refuse the
    /// layer). W8A8 needs its own eligibility and at least
    /// [`IU8G32_W8A8_MIN_ROWS`] rows; everything else takes the f16 twin.
    pub fn iu8g32_gemm_route(&self, k: usize, n: usize) -> Option<Iu8g32GemmRoute> {
        if !self.iu8g32_wmma_eligible(k) {
            return None;
        }
        if n >= IU8G32_W8A8_MIN_ROWS && self.iu8g32_w8a8_eligible(k) && iu8g32_w8a8_enabled() {
            Some(Iu8g32GemmRoute::W8A8)
        } else {
            Some(Iu8g32GemmRoute::F16Twin)
        }
    }

    /// `Y += W X` for one qt59 matrix through [`Gpu::iu8g32_gemm_route`].
    /// `x` is `[n, k]` f32; the W8A8 route quantizes it here.
    pub fn gemm_iu8g32_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        match self.iu8g32_gemm_route(k, n) {
            Some(Iu8g32GemmRoute::W8A8) => {
                let xq = self.ensure_int8_mmq_linear_x(x, n, k)?;
                self.gemm_iu8g32_w8a8_min(a, xq, y, m, k, n, true)
            }
            // The twin's own guard names the arch/K refusal.
            Some(Iu8g32GemmRoute::F16Twin) | None => {
                self.gemm_iu8g32_residual_wmma(a, x, y, m, k, n)
            }
        }
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
    /// caller supplies the pre-quantized X pointer. Prefill bodies reach it
    /// through [`Gpu::gemm_iu8g32_residual`], not directly.
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
        if m == 0 || n == 0 || !self.iu8g32_w8a8_eligible(k) {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "gemm_iu8g32_w8a8_min needs gfx1100/gfx1151, M>0, N>0 and K%256==0 (arch {}, M={m}, N={n}, K={k})",
                    self.arch
                ),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qt59_gemm_admission_is_one_rule() {
        for arch in ["gfx1100", "gfx1151"] {
            assert!(iu8g32_gemm_admits(arch, 128), "{arch}");
            assert!(iu8g32_gemm_admits(arch, 6144), "{arch}");
            assert!(!iu8g32_gemm_admits(arch, 0), "{arch}");
            assert!(
                !iu8g32_gemm_admits(arch, 32),
                "{arch}: retired per-32 width"
            );
            assert!(!iu8g32_gemm_admits(arch, 192), "{arch}");
        }
        for arch in ["gfx1201", "gfx1101", "gfx1030", ""] {
            assert!(!iu8g32_gemm_admits(arch, 256), "{arch:?}");
        }
    }
}
