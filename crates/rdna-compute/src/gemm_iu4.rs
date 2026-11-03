// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill) GEMM for [`crate::DType::IU4G256`].
//!
//! On exact gfx1100/gfx1151 ([`crate::arch_caps::supports_iu4g256_gemm`])
//! the activation format is `HIPFIRE_IU4_PREFILL_ACT`: `a4` (default) is the
//! W4A4 int4 MMQ on `v_wmma_i32_16x16x16_iu4`, `a8` the W4A8 int8 MMQ on
//! `v_wmma_i32_16x16x16_iu8`. Each is bit-identical to its MQ4V2 twin kernel
//! on a qt57 tensor that is the relayout of a qt44 tensor. Projections
//! narrower than one 128-row tile take the f16 MW-LDS WMMA.
//! Elsewhere, and while a graph is being captured or recorded, each token is
//! one f32 IU4 GEMV (the decode kernel), i.e. the per-token result.
//!
//! `X` is `[N, K]` row-major f32 (FWHT-256 rotated by the caller), `Y` is
//! `[N, M]` f32.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// One matrix of a shared-`X` batched call: weight, output, output rows.
pub type Iu4Proj<'a> = (&'a GpuTensor, &'a GpuTensor, usize);

/// `HIPFIRE_IU4_PREFILL=gemv`: skip the MMQ path (per-token f32 GEMV).
fn iu4_prefill_gemv() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_IU4_PREFILL")
                .ok()
                .as_deref(),
            Some("gemv")
        )
    })
}

/// `HIPFIRE_IU4_PREFILL_ACT`: activation format of the MMQ prefill path.
/// `a4` (default): `block_i4_128` int4 X (FWHT-rotated, the MQ4P W4A4 recipe)
/// into the planar-addressed lf16 iu4 WMMA kernel. `a8`: `block_i8_128` X
/// into the iu8 WMMA W4A8 MMQ (C11; KL-exact to decode, ~17-25% slower).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Iu4PrefillAct {
    A4,
    A8,
}

fn iu4_prefill_act() -> Iu4PrefillAct {
    static V: std::sync::OnceLock<Iu4PrefillAct> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_IU4_PREFILL_ACT")
            .ok()
            .as_deref()
        {
            Some("a8") => Iu4PrefillAct::A8,
            None | Some("") | Some("a4") => Iu4PrefillAct::A4,
            // A typo (or the removed `a8q`) must not take down a serving
            // daemon on its first prefill.
            Some(v) => {
                eprintln!("[iu4] HIPFIRE_IU4_PREFILL_ACT: ignoring unknown value {v:?}, using a4");
                Iu4PrefillAct::A4
            }
        }
    })
}

/// C11b/C13 `HIPFIRE_IU4_FUSED_X`: `0` = f32 producers + the standalone
/// quantizer; `1`/`a8` = fuse only the `a8` sidecar (bit-exact to the
/// standalone `block_i8_128` quantizer); unset/`all` (C13 default) = also fuse
/// `a4`, whose producer emit uses MQ4P's reduced MSE candidate set (bit-exact
/// to MQ4P prefill math, KL 3e-13; not bit-exact to `ensure_int4_mmq_x`).
/// Returns (fuse a8, fuse a4).
fn iu4_fused_x() -> (bool, bool) {
    use std::sync::OnceLock;
    static V: OnceLock<(bool, bool)> = OnceLock::new();
    *V.get_or_init(
        || match hipfire_config::developer_var("HIPFIRE_IU4_FUSED_X").as_deref() {
            Ok("0") => (false, false),
            Ok("1") | Ok("a8") => (true, false),
            _ => (true, true),
        },
    )
}

/// Sidecar a fused producer emitted for one `X` (C11b).
#[derive(Debug)]
pub enum Iu4PendingXKind {
    I4(crate::scratch::Int4MmqPrepared),
    I8(crate::scratch::Int8MmqPrepared),
}

/// [`Iu4PendingXKind`] keyed by the f32 `x_rot` it mirrors.
#[derive(Debug)]
pub struct Iu4PendingX {
    x: usize,
    n: usize,
    k: usize,
    kind: Iu4PendingXKind,
}

const W4A4_MODULE: &str = "gemm_iu4g256_w4a4";
const MWL_MODULE: &str = "gemm_iu4g256_mwl";

impl Gpu {
    /// True when the IU4 MMQ path serves this call.
    pub fn iu4g256_mmq_eligible(&self, k: usize, n: usize) -> bool {
        n > 0
            && k % 256 == 0
            && crate::arch_caps::supports_iu4g256_gemm(&self.arch)
            && !self.graphs.capture_mode
            && !self.replay.is_recording()
            && !iu4_prefill_gemv()
    }

    /// C11b: `Some(true)` = fuse an int8 (`a8`) sidecar into the producer
    /// for an IU4G256 consumer, `Some(false)` = int4 (`a4`), `None` = the
    /// f32-only producer (route ineligible or the knob off).
    fn iu4_fused_x_act(&self, k: usize, n: usize) -> Option<bool> {
        let (f8, f4) = iu4_fused_x();
        if !self.iu4g256_mmq_eligible(k, n) {
            return None;
        }
        match iu4_prefill_act() {
            Iu4PrefillAct::A4 if f4 => Some(false),
            Iu4PrefillAct::A8 if f8 => Some(true),
            _ => None,
        }
    }

    fn iu4_set_pending(&mut self, x: &GpuTensor, n: usize, k: usize, kind: Iu4PendingXKind) {
        self.iu4_pending_x = Some(Iu4PendingX {
            x: x.buf.as_ptr() as usize,
            n,
            k,
            kind,
        });
    }

    /// RMSNorm+FWHT for an IU4G256 consumer, also emitting the MMQ X
    /// sidecar when eligible. `Ok(false)`: nothing launched; run the f32
    /// producer. Always writes the f32 `x_rot` (the small-M f16 route reads it).
    pub fn iu4_fused_rmsnorm_rotate(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq: Option<&GpuTensor>,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        n: usize,
    ) -> HipResult<bool> {
        // The x_rot-storing RMSNorm producers stage the row in LDS:
        // (k + 256) * 4 bytes must fit the 64 KiB workgroup limit.
        if (k + 256) * 4 > 65536 {
            return Ok(false);
        }
        match self.iu4_fused_x_act(k, n) {
            Some(true) => {
                let res = self.reserve_int8_mmq(k, n)?;
                let p = self.fused_rmsnorm_rotate_mq_i8_batched(
                    x,
                    weight,
                    awq,
                    Some(x_rot),
                    res,
                    k,
                    eps,
                    n,
                )?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I8(p));
                Ok(true)
            }
            Some(false) => {
                let res = self.reserve_int4_mmq(k, n)?;
                let p = self.fused_rmsnorm_rotate_mq_i4_batched(
                    x,
                    weight,
                    awq,
                    Some(x_rot),
                    res,
                    k,
                    eps,
                    n,
                )?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I4(p));
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// [`Self::iu4_fused_rmsnorm_rotate`] for `silu(gate)*up` + FWHT.
    pub fn iu4_fused_silu_mul_rotate(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq: Option<&GpuTensor>,
        x_rot: &GpuTensor,
        k: usize,
        n: usize,
    ) -> HipResult<bool> {
        match self.iu4_fused_x_act(k, n) {
            Some(true) if awq.is_none() => {
                let res = self.reserve_int8_mmq(k, n)?;
                let p =
                    self.fused_silu_mul_rotate_mq_i8_batched(gate, up, Some(x_rot), res, k, n)?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I8(p));
                Ok(true)
            }
            Some(false) => {
                let res = self.reserve_int4_mmq(k, n)?;
                let p = self.fused_silu_mul_rotate_mq_i4_batched(
                    gate,
                    up,
                    awq,
                    Some(x_rot),
                    res,
                    k,
                    n,
                )?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I4(p));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// [`Self::iu4_fused_rmsnorm_rotate`] for a plain FWHT of `x`.
    pub fn iu4_fused_rotate_x(
        &mut self,
        x: &GpuTensor,
        awq: Option<&GpuTensor>,
        x_rot: &GpuTensor,
        k: usize,
        n: usize,
    ) -> HipResult<bool> {
        match self.iu4_fused_x_act(k, n) {
            Some(true) if awq.is_none() => {
                let res = self.reserve_int8_mmq(k, n)?;
                let p = self.rotate_x_mq_i8_batched(x, Some(x_rot), res, k, n)?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I8(p));
                Ok(true)
            }
            Some(false) => {
                let res = self.reserve_int4_mmq(k, n)?;
                let p = self.rotate_x_mq_i4_batched(x, awq, Some(x_rot), res, k, n)?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I4(p));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Take the pending sidecar if it mirrors `x` at `(n, k)` and is still
    /// live (generation-checked); any other call drops it.
    fn iu4_take_pending(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
        a8: bool,
    ) -> Option<*mut c_void> {
        let p = self.iu4_pending_x.take()?;
        if p.x != x.buf.as_ptr() as usize || p.n != n || p.k != k {
            return None;
        }
        match (&p.kind, a8) {
            (Iu4PendingXKind::I8(h), true) => self.int8_mmq_prepared_ptr(h, k, n).ok(),
            (Iu4PendingXKind::I4(h), false) => self.int4_mmq_prepared_ptr(h, k, n).ok(),
            _ => None,
        }
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
        let pending = self.iu4_pending_x.take();
        if n == 0 || projs.is_empty() {
            return Ok(());
        }
        if self.iu4g256_mmq_eligible(k, n) {
            self.iu4_pending_x = pending;
            self.bind_thread()?;
            let mut big: Vec<Iu4Proj<'_>> = Vec::with_capacity(projs.len());
            // f16 X converted once per call, uncached: the pointer-keyed
            // fp16 cache can serve stale data when a buffer is reused.
            let mut x16: Option<*mut c_void> = None;
            for &(a, y, m) in projs {
                if m < 128 {
                    if !add {
                        self.iu4_zero_y(y, n * m)?;
                    }
                    let xp = match x16 {
                        Some(p) => p,
                        None => {
                            let p = self.iu4_fp16_x(x, n * k)?;
                            x16 = Some(p);
                            p
                        }
                    };
                    self.gemm_iu4g256_mwl_residual_f16(a, xp, y, m, k, n)?;
                } else {
                    big.push((a, y, m));
                }
            }
            if big.is_empty() {
                self.iu4_pending_x = None;
                return Ok(());
            }
            match iu4_prefill_act() {
                Iu4PrefillAct::A4 => {
                    let xq = match self.iu4_take_pending(x, n, k, false) {
                        Some(p) => p,
                        None => self.ensure_int4_mmq_x(x, n, k)?,
                    };
                    for &(a, y, m) in &big {
                        self.gemm_iu4g256_w4a4_prequant(a, xq, y, m, k, n, add)?;
                    }
                }
                Iu4PrefillAct::A8 => {
                    let xq = match self.iu4_take_pending(x, n, k, true) {
                        Some(p) => p,
                        None => self.ensure_int8_mmq_x(x, n, k)?,
                    };
                    for &(a, y, m) in &big {
                        self.gemm_iu4g256_w4a8_prequant(a, xq, y, m, k, n, add)?;
                    }
                }
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
        let func = if add {
            "gemv_iu4g256_residual"
        } else {
            "gemv_iu4g256"
        };
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

    fn iu4_zero_y(&mut self, y: &GpuTensor, elems: usize) -> HipResult<()> {
        match self.active_stream.as_ref() {
            Some(stream) => self.hip.memset_async(&y.buf, 0, elems * 4, stream),
            None => self.hip.memset(&y.buf, 0, elems * 4),
        }
    }

    /// `Y += W X` for a small-M IU4G256 matrix: f16 X, the planar-addressed
    /// MW4-LDS f16 WMMA residual (the MQ4V2 small-M prefill route).
    pub fn gemm_iu4g256_mwl_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k % 256 != 0 || !crate::arch_caps::supports_iu4g256_gemm(&self.arch) {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 MW-LDS requires gfx1100/gfx1151 and K % 256 == 0",
            ));
        }
        self.bind_thread()?;
        let xp = self.iu4_fp16_x(x, n * k)?;
        self.gemm_iu4g256_mwl_residual_f16(a, xp, y, m, k, n)
    }

    /// f32 -> f16 X conversion that never trusts the pointer-keyed cache,
    /// then drops the cache key (the scratch it describes was overwritten).
    fn iu4_fp16_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        let p = self.convert_fp16_x_uncached(x, n_elems)?;
        self.invalidate_fp16_cache();
        Ok(p)
    }

    /// MW4-LDS residual on an already-converted f16 X (`n * k` halves).
    pub fn gemm_iu4g256_mwl_residual_f16(
        &mut self,
        a: &GpuTensor,
        x_f16_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k % 256 != 0 || !crate::arch_caps::supports_iu4g256_gemm(&self.arch) {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 MW-LDS requires gfx1100/gfx1151 and K % 256 == 0",
            ));
        }
        const NAME: &str = "gemm_iu4g256_residual_wmma_gfx11_mw4_lds";
        self.ensure_kernel(MWL_MODULE, kernels::GEMM_IU4G256_MWL_SRC, NAME)?;
        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        const WAVES: usize = 4;
        let grid = [m.div_ceil(16) as u32, n.div_ceil(16 * WAVES) as u32, 1];
        let bytes = m * (k / 256) * crate::dispatch::MQ4V2_GROUP_BYTES + n * k * 2 + n * m * 8;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", NAME, bytes);
        let result = self.launch_maybe_blob(
            NAME,
            grid,
            [(32 * WAVES) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
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

    /// The IU4 W4A4 launch over a pre-quantized `block_i4_128` X (the layout
    /// of `ensure_int4_mmq_x`). Full tiles: the lf16 16-wave column entry
    /// (grid [N/128, M/128]); otherwise the guarded generic entry.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_iu4g256_w4a4_prequant(
        &mut self,
        a: &GpuTensor,
        x_i4_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        if !crate::arch_caps::supports_iu4g256_gemm(&self.arch) {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 W4A4 MMQ requires exact gfx1100/gfx1151",
            ));
        }
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 W4A4 MMQ requires M > 0 and K % 256 == 0",
            ));
        }
        self.bind_thread()?;
        let full = m % 128 == 0 && batch_size % 128 == 0;
        let kernel_name = match (full, add) {
            (true, false) => "gemm_iu4g256_w4a4_full_set_lf16_col",
            (true, true) => "gemm_iu4g256_w4a4_full_add_lf16_col",
            (false, _) => "gemm_iu4g256_w4a4",
        };
        self.ensure_kernel(W4A4_MODULE, kernels::GEMM_IU4G256_W4A4_SRC, kernel_name)?;
        let mut a_ptr = a.buf.as_ptr();
        let mut xq_ptr = x_i4_ptr;
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
        // Same tile geometry as the MQ4V2 IU4 family: LDS = (128*18 + 128*42)*4.
        const MMQ_X: usize = 128;
        const MMQ_Y: usize = 128;
        const MMQ_TILE_Y_K: usize = 18;
        const MMQ_TILE_X_K: usize = 42;
        let row_tiles = m.div_ceil(MMQ_Y);
        let batch_tiles = batch_size.div_ceil(MMQ_X);
        let shared_mem =
            ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
        let (grid, block) = if full {
            ([batch_tiles as u32, row_tiles as u32, 1], [32, 16, 1])
        } else {
            ([row_tiles as u32, batch_tiles as u32, 1], [32, 8, 1])
        };
        let bytes = m * (k / 256) * crate::dispatch::MQ4V2_GROUP_BYTES + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result =
            self.launch_maybe_blob(kernel_name, grid, block, shared_mem, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
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

/// `HIPFIRE_IU4_I8_TILE=128|256` pins the W4A8 i8 tile (default: 256 when
/// the 256-tile grid fills the CUs, as the MQ4V2 A8 route).
fn iu4_i8_tile() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_IU4_I8_TILE")
            .ok()
            .as_deref()
        {
            Some("128") => Some(128),
            Some("256") => Some(256),
            _ => None,
        }
    })
}

impl Gpu {
    /// IU4G256 W4A8 (`block_i8_128` X, iu8 WMMA): `Y = W X` / `Y += W X`.
    pub fn gemm_iu4g256_w4a8_prequant(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.i8_asym_prequant(true, a, xq, y, m, k, n, add)
    }

    /// Interleaved-MQ4V2 twin of [`Self::gemm_iu4g256_w4a8_prequant`] (tests).
    pub fn gemm_mq4g256v2_asym_i8_prequant(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        self.i8_asym_prequant(false, a, xq, y, m, k, n, add)
    }

    #[allow(clippy::too_many_arguments)]
    fn i8_asym_prequant(
        &mut self,
        planar: bool,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        add: bool,
    ) -> HipResult<()> {
        if !crate::arch_caps::supports_iu4g256_gemm(&self.arch) {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 W4A8 i8 MMQ requires gfx1100/gfx1151",
            ));
        }
        if m == 0 || n == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "IU4 W4A8 i8 MMQ requires M, N > 0 and K % 256 == 0",
            ));
        }
        self.bind_thread()?;
        let tile = iu4_i8_tile().unwrap_or_else(|| {
            if m.div_ceil(256) * n.div_ceil(256) >= self.cu_count_or_default() {
                256
            } else {
                128
            }
        });
        let (module, source, threads, lds): (&str, &str, u32, u32) = match (planar, tile) {
            (true, 256) => (
                "gemm_iu4g256_w4a8_t256",
                kernels::GEMM_IU4G256_W4A8_GFX11_T256_SRC,
                1024,
                57344,
            ),
            (true, _) => (
                "gemm_iu4g256_w4a8",
                kernels::GEMM_IU4G256_W4A8_GFX11_SRC,
                512,
                28672,
            ),
            (false, 256) => (
                "gemm_mq4g256v2_asym_mmq_i8_t256",
                kernels::GEMM_MQ4G256V2_ASYM_MMQ_I8_GFX11_T256_SRC,
                1024,
                57344,
            ),
            (false, _) => (
                "gemm_mq4g256v2_asym_mmq_i8",
                kernels::GEMM_MQ4G256V2_ASYM_MMQ_I8_GFX11_SRC,
                512,
                28672,
            ),
        };
        let kernel: &'static str = match (planar, add, tile) {
            (true, true, 256) => "gemm_iu4g256_w4a8_full_add_t256",
            (true, false, 256) => "gemm_iu4g256_w4a8_full_set_t256",
            (true, true, _) => "gemm_iu4g256_w4a8_full_add",
            (true, false, _) => "gemm_iu4g256_w4a8_full_set",
            (false, true, 256) => "gemm_mq4g256v2_asym_mmq_i8_full_add_t256",
            (false, false, 256) => "gemm_mq4g256v2_asym_mmq_i8_full_set_t256",
            (false, true, _) => "gemm_mq4g256v2_asym_mmq_i8_full_add",
            (false, false, _) => "gemm_mq4g256v2_asym_mmq_i8_full_set",
        };
        self.ensure_kernel(module, source, kernel)?;
        let mut ap = a.buf.as_ptr();
        let mut xp = xq;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut nv = n as i32;
        let mut av = i32::from(add);
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
        ];
        let bytes = m * (k / 256) * crate::dispatch::MQ4V2_GROUP_BYTES + n * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [m.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1],
            [threads, 1, 1],
            lds,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b.push_i32(av);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
