// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode GEMV for [`crate::DType::IU4G256`].
//!
//! One wave per output row, f32 activation. The arithmetic is the MQ4G256V2
//! decode kernel's, with IU4 address math, so a qt 57 tensor that is the
//! `repack_mq4v2` relayout of a qt 44 tensor decodes bit-identically to it.
//!
//! `gemv_iu4g256_residual` is the fused `y += W·x` twin. `fused_iu4g256_rows`
//! runs up to four matrices that share `x` in one launch (gate/up, q/k/v,
//! qkv/z/beta/alpha); each output equals the plain kernel on that matrix.
//!
//! `HIPFIRE_IU4_DECODE=dot8` selects the original W4A4 kernels (signed int4
//! activation per 32, `v_dot8_i32_iu4`, RDNA3/RDNA4 only), whose numeric
//! contract is `hipfire_runtime::iu_quant::gemv_iu4_dot8`.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const MODULE: &str = "gemv_iu4g256";

/// Integer-activation decode mode for IU4G256 (`HIPFIRE_IU4_DECODE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iu4DecodeMode {
    /// Default: f32 activation (MQ4G256V2's algorithm, bit-exact with the twin).
    F32,
    /// `dot4`: W4A8, int8 activation per 32, `v_dot4_i32_iu8`.
    Dot4,
    /// `dot8`: W4A4, int4 activation per 32, `v_dot8_i32_iu4` (branch original).
    Dot8,
}

/// `HIPFIRE_IU4_DECODE=f32|dot4|dot8` (unset or unknown = f32). Read once.
pub fn iu4_decode_mode() -> Iu4DecodeMode {
    static V: std::sync::OnceLock<Iu4DecodeMode> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_IU4_DECODE")
            .ok()
            .as_deref()
        {
            Some("dot8") => Iu4DecodeMode::Dot8,
            Some("dot4") => Iu4DecodeMode::Dot4,
            _ => Iu4DecodeMode::F32,
        }
    })
}

/// True for either integer-activation mode.
pub fn iu4_decode_dot8() -> bool {
    iu4_decode_mode() != Iu4DecodeMode::F32
}

/// True only for `dot8`, the original W4A4 route: the dispatcher skips the
/// IU4 fused-row keys and the fused SwiGLU-down arm and prefill runs per
/// token, as on the branch before review. `dot4` keeps the fused keys (with
/// the `_dot4q` kernels) and the MMQ prefill.
pub fn iu4_decode_w4a4() -> bool {
    iu4_decode_mode() == Iu4DecodeMode::Dot8
}

/// Fixed size of `scratch.iu4_a8_x_scratch` (2 B per K element: K <= 131072).
const IU4_A8_SCRATCH_BYTES: usize = 256 * 1024;

fn check_shape(what: &str, m: usize, k: usize) -> HipResult<()> {
    if m == 0 || k == 0 || k % 256 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("{what}: M > 0 and K % 256 == 0"),
        ));
    }
    Ok(())
}

impl Gpu {
    /// `y = W x` for a planar IU4G256 weight. `x` is one FWHT-256-rotated
    /// f32 vector of length `k`. `k` must be a positive multiple of 256.
    pub fn gemv_iu4g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let func = match iu4_decode_mode() {
            Iu4DecodeMode::Dot8 => "gemv_iu4g256_dot8",
            Iu4DecodeMode::Dot4 => return self.gemv_iu4g256_dot4q(a, x, y, m, k, false),
            Iu4DecodeMode::F32 => "gemv_iu4g256",
        };
        self.gemv_iu4g256_launch(func, a, x, y, m, k)
    }

    /// `y += W x` for a planar IU4G256 weight (`y` is both accumulated into
    /// and stored).
    pub fn gemv_iu4g256_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let func = match iu4_decode_mode() {
            Iu4DecodeMode::Dot8 => "gemv_iu4g256_dot8_residual",
            Iu4DecodeMode::Dot4 => return self.gemv_iu4g256_dot4q(a, x, y, m, k, true),
            Iu4DecodeMode::F32 => "gemv_iu4g256_residual",
        };
        self.gemv_iu4g256_launch(func, a, x, y, m, k)
    }

    /// The W4A8 `v_dot4` kernel, regardless of `HIPFIRE_IU4_DECODE`
    /// (parity tests and A/B). `residual` selects `y += W x`.
    pub fn gemv_iu4g256_dot4(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        residual: bool,
    ) -> HipResult<()> {
        let func = if residual {
            "gemv_iu4g256_dot4_residual"
        } else {
            "gemv_iu4g256_dot4"
        };
        self.gemv_iu4g256_launch(func, a, x, y, m, k)
    }

    /// W4A8 v2: quantize `x` once (`iu4_quant_x_a8`) into the fixed scratch,
    /// then `gemv_iu4g256_dot4q[_residual]`. Same per-group arithmetic as
    /// [`Self::gemv_iu4g256_dot4`]. Used by `HIPFIRE_IU4_DECODE=dot4`.
    pub fn gemv_iu4g256_dot4q(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        residual: bool,
    ) -> HipResult<()> {
        check_shape("gemv_iu4g256_dot4q", m, k)?;
        let qp = self.iu4_quant_x_a8(x.buf.as_ptr(), k)?;
        let func = if residual {
            "gemv_iu4g256_dot4q_residual"
        } else {
            "gemv_iu4g256_dot4q"
        };
        self.gemv_iu4g256_launch_raw(func, a.buf.as_ptr(), qp, y.buf.as_ptr(), m, k)
    }

    /// Quantize a length-`k` f32 `x` into `scratch.iu4_a8_x_scratch` and
    /// return its pointer (valid until the next call on this stream).
    fn iu4_quant_x_a8(&mut self, xp: *mut c_void, k: usize) -> HipResult<*mut c_void> {
        if !self.arch_caps.has_iu4_wmma() {
            return Err(hip_bridge::HipError::new(
                0,
                "iu4 dot4: RDNA3/RDNA4 integer dot only",
            ));
        }
        if k * 2 > IU4_A8_SCRATCH_BYTES {
            return Err(hip_bridge::HipError::new(
                0,
                "iu4 dot4: K exceeds the fixed x scratch",
            ));
        }
        self.bind_thread()?;
        if self.scratch.iu4_a8_x_scratch.is_none() {
            let b = self.hip.malloc(IU4_A8_SCRATCH_BYTES)?;
            self.scratch.iu4_a8_x_scratch = Some(b);
        }
        let qp = self.scratch.iu4_a8_x_scratch.as_ref().unwrap().as_ptr();
        self.ensure_kernel(MODULE, kernels::GEMV_IU4G256_SRC, "iu4_quant_x_a8")?;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "iu4_quant_x_a8", k * 6);
        let result = self.launch_maybe_blob(
            "iu4_quant_x_a8",
            [(k / 256) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(qp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(qp)
    }

    /// The W4A4 `v_dot8` kernel, regardless of `HIPFIRE_IU4_DECODE`
    /// (parity tests and A/B). `residual` selects `y += W x`.
    pub fn gemv_iu4g256_dot8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        residual: bool,
    ) -> HipResult<()> {
        let func = if residual {
            "gemv_iu4g256_dot8_residual"
        } else {
            "gemv_iu4g256_dot8"
        };
        self.gemv_iu4g256_launch(func, a, x, y, m, k)
    }

    /// `y_i = W_i x` for up to four IU4G256 weights sharing `x` and `k`, in
    /// one launch. Each `(weight, output, rows)`; an empty slice is an error.
    pub fn fused_iu4g256_rows(
        &mut self,
        mats: &[(&GpuTensor, &GpuTensor, usize)],
        x: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        if mats.is_empty() || mats.len() > 4 {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_iu4g256_rows: 1..=4 matrices",
            ));
        }
        let total: usize = mats.iter().map(|m| m.2).sum();
        check_shape("fused_iu4g256_rows", total, k)?;
        if total > u32::MAX as usize {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_iu4g256_rows: grid overflow",
            ));
        }
        let dot4 = iu4_decode_mode() == Iu4DecodeMode::Dot4;
        self.fused_iu4g256_rows_impl(mats, x, k, dot4)
    }

    /// [`Self::fused_iu4g256_rows`] with the kernel chosen explicitly:
    /// `dot4` = quantize x once, then `fused_iu4g256_dot4q_rows` (W4A8).
    pub fn fused_iu4g256_rows_impl(
        &mut self,
        mats: &[(&GpuTensor, &GpuTensor, usize)],
        x: &GpuTensor,
        k: usize,
        dot4: bool,
    ) -> HipResult<()> {
        if mats.is_empty() || mats.len() > 4 {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_iu4g256_rows: 1..=4 matrices",
            ));
        }
        let total: usize = mats.iter().map(|m| m.2).sum();
        check_shape("fused_iu4g256_rows", total, k)?;
        let fname: &'static str = if dot4 {
            "fused_iu4g256_dot4q_rows"
        } else {
            "fused_iu4g256_rows"
        };
        let xp = if dot4 {
            self.iu4_quant_x_a8(x.buf.as_ptr(), k)?
        } else {
            x.buf.as_ptr()
        };
        self.bind_thread()?;
        self.ensure_kernel(MODULE, kernels::GEMV_IU4G256_SRC, fname)?;
        let mut ap = [std::ptr::null_mut::<c_void>(); 4];
        let mut yp = [std::ptr::null_mut::<c_void>(); 4];
        let mut mv = [0i32; 4];
        for (i, (a, y, m)) in mats.iter().enumerate() {
            ap[i] = a.buf.as_ptr();
            yp[i] = y.buf.as_ptr();
            mv[i] = *m as i32;
        }
        // Unused slots alias slot 0 and carry zero rows.
        for i in mats.len()..4 {
            ap[i] = ap[0];
            yp[i] = yp[0];
        }
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = Vec::with_capacity(13);
        for p in &ap {
            params.push(p as *const _ as *mut c_void);
        }
        params.push(&xp as *const _ as *mut c_void);
        for p in &yp {
            params.push(p as *const _ as *mut c_void);
        }
        for m in &mv {
            params.push(m as *const _ as *mut c_void);
        }
        params.push(&kv as *const _ as *mut c_void);
        let bytes = mats
            .iter()
            .map(|m| crate::profile::gemv_hfq4g256_bytes(m.2, k))
            .sum();
        let timer = crate::profile::begin_timer(&self.hip, "fused", fname, bytes);
        let result = self.launch_maybe_blob(
            fname,
            [total as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in ap {
                    b.push_ptr(p);
                }
                b.push_ptr(xp);
                for p in yp {
                    b.push_ptr(p);
                }
                for m in mv {
                    b.push_i32(m);
                }
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
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
        check_shape("gemv_iu4g256", m, k)?;
        if (func.contains("dot8") || func.contains("dot4")) && !self.arch_caps.has_iu4_wmma() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_iu4g256_dot8: v_dot8_i32_iu4 is the RDNA3/RDNA4 integer-4 dot",
            ));
        }
        self.gemv_iu4g256_launch_raw(func, a.buf.as_ptr(), x.buf.as_ptr(), y.buf.as_ptr(), m, k)
    }

    /// Raw-pointer form of [`Self::gemv_iu4g256_launch`] (the per-token
    /// prefill fallback walks `x`/`y` rows of a batch).
    pub(crate) fn gemv_iu4g256_launch_raw(
        &mut self,
        func: &'static str,
        ap: *mut c_void,
        xp: *mut c_void,
        yp: *mut c_void,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        check_shape("gemv_iu4g256", m, k)?;
        self.bind_thread()?;
        self.ensure_kernel(MODULE, kernels::GEMV_IU4G256_SRC, func)?;
        let mv = m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        // The f32 residual kernel is the two-rows-per-block MQ4V2 residual port.
        let grid = if func == "gemv_iu4g256_residual" {
            (m as u32).div_ceil(2)
        } else {
            m as u32
        };
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", func, bytes);
        let result = self.launch_maybe_blob(func, [grid, 1, 1], [32, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ap);
            b.push_ptr(xp);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
