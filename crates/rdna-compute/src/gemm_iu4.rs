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

/// `HIPFIRE_IU4_PREFILL_ACT`: activation format of the MMQ prefill path.
/// `a4` (default): `block_i4_128` int4 X (FWHT-rotated, the MQ4P W4A4 recipe)
/// into the planar-addressed lf16 iu4 WMMA kernel. `a8`: `block_q8_1_mmq` X
/// into the W4A8 MMQ kernel (C2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Iu4PrefillAct {
    A4,
    /// `block_i8_128` X into the iu8 WMMA W4A8 MMQ (C11).
    A8,
    /// `block_q8_1_mmq` X into the C2 W4A8 MMQ (`a8q`, the pre-C11 a8).
    A8Q,
}

fn iu4_prefill_act() -> Iu4PrefillAct {
    static V: std::sync::OnceLock<Iu4PrefillAct> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_IU4_PREFILL_ACT").ok().as_deref() {
            Some("a8") => Iu4PrefillAct::A8,
            Some("a8q") => Iu4PrefillAct::A8Q,
            _ => Iu4PrefillAct::A4,
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
    *V.get_or_init(|| match std::env::var("HIPFIRE_IU4_FUSED_X").as_deref() {
        Ok("0") => (false, false),
        Ok("1") | Ok("a8") => (true, false),
        _ => (true, true),
    })
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

fn iu4_prefill_a4() -> bool {
    iu4_prefill_act() == Iu4PrefillAct::A4
}

/// `HIPFIRE_IU4_SMALLM=mmq`: keep M < 128 projections on the MMQ path
/// instead of the f16 MW4-LDS WMMA residual (default `wmma`).
fn iu4_smallm_wmma() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            hipfire_config::developer_var("HIPFIRE_IU4_SMALLM").ok().as_deref(),
            Some("mmq")
        )
    })
}

/// C14 per-role a8 override of the a4 default. `HIPFIRE_IU4_A8_ROLES`: comma
/// list of `aqkv` (full-attn q/k/v), `ao` (full-attn o), `dqkv` (DeltaNet
/// in_proj qkv/z/beta/alpha), `do` (DeltaNet out), `gu` (FFN gate/up), `dn`
/// (FFN down), `all`. `HIPFIRE_IU4_A8_LAYERS`: comma list of `firstN`,
/// `lastN`, `a-b`, `a`. Either unset = every role / every layer; both unset
/// = off. Only applies when `HIPFIRE_IU4_PREFILL_ACT` is `a4`.
fn iu4_a8_sel() -> &'static Option<(u8, Vec<(u8, usize, usize)>)> {
    static V: std::sync::OnceLock<Option<(u8, Vec<(u8, usize, usize)>)>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let roles = std::env::var("HIPFIRE_IU4_A8_ROLES").ok().filter(|v| !v.is_empty());
        let layers = std::env::var("HIPFIRE_IU4_A8_LAYERS").ok().filter(|v| !v.is_empty());
        if roles.is_none() && layers.is_none() {
            return None;
        }
        let mut mask = 0u8;
        match roles {
            None => mask = 0x3f,
            Some(r) => {
                for t in r.split(',').map(str::trim) {
                    mask |= match t {
                        "aqkv" => 1 << Gpu::IU4_ROLE_AQKV,
                        "ao" => 1 << Gpu::IU4_ROLE_AO,
                        "dqkv" => 1 << Gpu::IU4_ROLE_DQKV,
                        "do" => 1 << Gpu::IU4_ROLE_DO,
                        "gu" => 1 << Gpu::IU4_ROLE_GU,
                        "dn" => 1 << Gpu::IU4_ROLE_DN,
                        "all" => 0x3f,
                        _ => panic!("HIPFIRE_IU4_A8_ROLES: unknown role {t:?}"),
                    };
                }
            }
        }
        let mut sel = Vec::new();
        for t in layers.as_deref().unwrap_or("").split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let bad = || panic!("HIPFIRE_IU4_A8_LAYERS: bad item {t:?}");
            if let Some(v) = t.strip_prefix("first") {
                sel.push((1, v.parse().unwrap_or_else(|_| bad()), 0));
            } else if let Some(v) = t.strip_prefix("last") {
                sel.push((2, v.parse().unwrap_or_else(|_| bad()), 0));
            } else if let Some((a, b)) = t.split_once('-') {
                sel.push((0, a.parse().unwrap_or_else(|_| bad()), b.parse().unwrap_or_else(|_| bad())));
            } else {
                let a = t.parse().unwrap_or_else(|_| bad());
                sel.push((0, a, a));
            }
        }
        Some((mask, sel))
    })
}

impl Gpu {
    pub const IU4_ROLE_AQKV: u8 = 0;
    pub const IU4_ROLE_AO: u8 = 1;
    pub const IU4_ROLE_DQKV: u8 = 2;
    pub const IU4_ROLE_DO: u8 = 3;
    pub const IU4_ROLE_GU: u8 = 4;
    pub const IU4_ROLE_DN: u8 = 5;

    /// The prefill activation format for the GEMM being issued: the global
    /// `HIPFIRE_IU4_PREFILL_ACT`, with the C14 per-role/layer a8 override.
    fn iu4_act_eff(&self) -> Iu4PrefillAct {
        let base = iu4_prefill_act();
        if base != Iu4PrefillAct::A4 || self.iu4_role >= 6 {
            return base;
        }
        let Some((mask, layers)) = iu4_a8_sel() else { return base };
        if mask & (1u8 << self.iu4_role) == 0 {
            return base;
        }
        let (li, nl) = self.iu4_layer;
        let hit = layers.is_empty()
            || layers.iter().any(|&(kind, a, b)| match kind {
                0 => li >= a && li <= b,
                1 => li < a,
                _ => nl > 0 && li + a >= nl,
            });
        if hit { Iu4PrefillAct::A8 } else { base }
    }
}

const W4A4_MODULE: &str = "gemm_iu4g256_w4a4";
const MWL_MODULE: &str = "gemm_iu4g256_mwl";

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

    /// C11b: `Some(true)` = fuse an int8 (`a8`) sidecar into the producer
    /// for an IU4G256 consumer, `Some(false)` = int4 (`a4`), `None` = the
    /// f32-only producer (route ineligible, `a8q`, or the knob off).
    fn iu4_fused_x_act(&self, k: usize, n: usize) -> Option<bool> {
        let (f8, f4) = iu4_fused_x();
        if !self.iu4g256_mmq_eligible(k, n) {
            return None;
        }
        match self.iu4_act_eff() {
            Iu4PrefillAct::A4 if f4 => Some(false),
            Iu4PrefillAct::A8 if f8 => Some(true),
            _ => None,
        }
    }

    fn iu4_set_pending(&mut self, x: &GpuTensor, n: usize, k: usize, kind: Iu4PendingXKind) {
        self.iu4_pending_x = Some(Iu4PendingX { x: x.buf.as_ptr() as usize, n, k, kind });
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
                let p = self.fused_rmsnorm_rotate_mq_i8_batched(x, weight, awq, Some(x_rot), res, k, eps, n)?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I8(p));
                Ok(true)
            }
            Some(false) => {
                let res = self.reserve_int4_mmq(k, n)?;
                let p = self.fused_rmsnorm_rotate_mq_i4_batched(x, weight, awq, Some(x_rot), res, k, eps, n)?;
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
                let p = self.fused_silu_mul_rotate_mq_i8_batched(gate, up, Some(x_rot), res, k, n)?;
                self.iu4_set_pending(x_rot, n, k, Iu4PendingXKind::I8(p));
                Ok(true)
            }
            Some(false) => {
                let res = self.reserve_int4_mmq(k, n)?;
                let p = self.fused_silu_mul_rotate_mq_i4_batched(gate, up, awq, Some(x_rot), res, k, n)?;
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
    fn iu4_take_pending(&mut self, x: &GpuTensor, n: usize, k: usize, a8: bool) -> Option<*mut c_void> {
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
            let smallm = iu4_smallm_wmma();
            let mut big: Vec<Iu4Proj<'_>> = Vec::with_capacity(projs.len());
            // f16 X converted once per call, uncached: the pointer-keyed
            // fp16 cache can serve stale data when a buffer is reused.
            let mut x16: Option<*mut c_void> = None;
            for &(a, y, m) in projs {
                if smallm && m < 128 {
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
            let act = self.iu4_act_eff();
            if act == Iu4PrefillAct::A4 {
                let xq = match self.iu4_take_pending(x, n, k, false) {
                    Some(p) => p,
                    None => self.ensure_int4_mmq_x(x, n, k)?,
                };
                for &(a, y, m) in &big {
                    self.gemm_iu4g256_w4a4_prequant(a, xq, y, m, k, n, add)?;
                }
                return Ok(());
            }
            if act == Iu4PrefillAct::A8 {
                let xq = match self.iu4_take_pending(x, n, k, true) {
                    Some(p) => p,
                    None => self.ensure_int8_mmq_x(x, n, k)?,
                };
                for &(a, y, m) in &big {
                    self.gemm_iu4g256_w4a8_prequant(a, xq, y, m, k, n, add)?;
                }
                return Ok(());
            }
            self.iu4_pending_x = None;
            let xq = if self.flags.gfx11_mmq_x128_enabled() {
                self.ensure_q8_1_mmq_x128(x, n, k)?
            } else {
                self.ensure_q8_1_mmq_x(x, n, k)?
            };
            for &(a, y, m) in &big {
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
        if k % 256 != 0 || !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") {
            return Err(hip_bridge::HipError::new(1, "IU4 MW-LDS requires gfx1100/gfx1151 and K % 256 == 0"));
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
        if k % 256 != 0 || !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") {
            return Err(hip_bridge::HipError::new(1, "IU4 MW-LDS requires gfx1100/gfx1151 and K % 256 == 0"));
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
        let result = self.launch_maybe_blob(NAME, grid, [(32 * WAVES) as u32, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b.push_i32(n_val);
            b
        });
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
        if !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") {
            return Err(hip_bridge::HipError::new(1, "IU4 W4A4 MMQ requires exact gfx1100/gfx1151"));
        }
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(1, "IU4 W4A4 MMQ requires M > 0 and K % 256 == 0"));
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
        let result = self.launch_maybe_blob(kernel_name, grid, block, shared_mem, &mut params, || {
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
        match hipfire_config::developer_var("HIPFIRE_IU4_I8_TILE").ok().as_deref() {
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
        if !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") {
            return Err(hip_bridge::HipError::new(1, "IU4 W4A8 i8 MMQ requires gfx1100/gfx1151"));
        }
        if m == 0 || n == 0 || k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(1, "IU4 W4A8 i8 MMQ requires M, N > 0 and K % 256 == 0"));
        }
        self.bind_thread()?;
        let tile = iu4_i8_tile().unwrap_or_else(|| {
            if m.div_ceil(256) * n.div_ceil(256) >= self.cu_count_or_default() { 256 } else { 128 }
        });
        let (module, source, threads, lds): (&str, &str, u32, u32) = match (planar, tile) {
            (true, 256) => ("gemm_iu4g256_w4a8_t256", kernels::GEMM_IU4G256_W4A8_GFX11_T256_SRC, 1024, 57344),
            (true, _) => ("gemm_iu4g256_w4a8", kernels::GEMM_IU4G256_W4A8_GFX11_SRC, 512, 28672),
            (false, 256) => ("gemm_mq4g256v2_asym_mmq_i8_t256", kernels::GEMM_MQ4G256V2_ASYM_MMQ_I8_GFX11_T256_SRC, 1024, 57344),
            (false, _) => ("gemm_mq4g256v2_asym_mmq_i8", kernels::GEMM_MQ4G256V2_ASYM_MMQ_I8_GFX11_SRC, 512, 28672),
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
