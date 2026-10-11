// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 Flash-Next lane-step row kernels as one PeaceMaker module
//! (`qwen4_lane_rows_pm`): four multi-column (1..=8 activation columns)
//! GEMV exports whose every column keeps the arithmetic DAG of its 1-row HIP
//! incumbent byte for byte while the activation loads are shared across
//! several weight rows.
//!
//! Exports, in frozen table order: `gemv_mq6g256v2_xbatch_pm` and
//! `gemv_mq6g256v2_x4_xbatch_pm` (authored by [`super::qwen4_lane_rows_mq6`]),
//! `gemv_q8_0_k8_xbatch_pm` and `gemv_q8_0_k320_xbatch_pm` (authored by
//! [`super::qwen4_lane_rows_q8`]). [`emit_module`] combines both authors with
//! [`super::iu4_gemm::module`] and refuses any other export list or any arch
//! but exact gfx1151.
use crate::{Arch, Emitted};
use super::iu4_gemm::ModuleProof;
use super::{qwen4_lane_rows_mq6, qwen4_lane_rows_q8};

/// Module name; the embedded image is `kernels/qwen4_lane_rows_pm_gfx1151.hxaco`.
pub const MODULE: &str = "qwen4_lane_rows_pm";

/// The frozen exports, in module order.
pub const SYMBOLS: [&str; 4] = [
    "gemv_mq6g256v2_xbatch_pm",
    "gemv_mq6g256v2_x4_xbatch_pm",
    "gemv_q8_0_k8_xbatch_pm",
    "gemv_q8_0_k320_xbatch_pm",
];

/// Kernarg block size in bytes of each export, in [`SYMBOLS`] order.
pub const KERNARG_BYTES: [u32; 4] = [36, 96, 44, 32];

/// Workgroup size (threads) of each export, in [`SYMBOLS`] order.
pub const WORKGROUP_SIZE: [u32; 4] = [32, 32, 256, 64];

/// `(offset, size)` of every kernarg of each export, in [`SYMBOLS`] order.
pub const KERNARGS: [&[(u32, u32)]; 4] = [
    &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)],
    &[
        (0, 8), (8, 8), (16, 8), (24, 8), (32, 8), (40, 8), (48, 8), (56, 8), (64, 8),
        (72, 4), (76, 4), (80, 4), (84, 4), (88, 4), (92, 4),
    ],
    &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4), (40, 4)],
    &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4)],
];

/// Committed image file name.
pub fn bundle_name() -> String { format!("{MODULE}_{}.hxaco", Arch::Gfx1151.name()) }

fn exported(e: &Emitted) -> Option<&str> {
    e.s_text.lines().find_map(|l| l.trim().strip_prefix(".amdhsa_kernel ")).map(str::trim)
}

/// Build both authors' exports and combine them into the one module source and
/// proof. The exports must be exactly [`SYMBOLS`] in order and built for exact
/// gfx1151.
pub fn emit_module() -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let mut emitted = qwen4_lane_rows_mq6::build_gfx1151().map_err(|e| format!("{MODULE}: mq6: {e}"))?;
    emitted.extend(qwen4_lane_rows_q8::build_gfx1151().map_err(|e| format!("{MODULE}: q8: {e}"))?);
    let got: Vec<&str> = emitted.iter().map(|e| exported(e).unwrap_or("")).collect();
    if got != SYMBOLS {
        return Err(format!("{MODULE}: builders export {got:?}, the frozen exports are {SYMBOLS:?}"));
    }
    if let Some(e) = emitted.iter().find(|e| e.proof.arch != Arch::Gfx1151) {
        return Err(format!("{MODULE}: {} built for {:?}, not exact gfx1151", exported(e).unwrap_or(""), e.proof.arch));
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}
