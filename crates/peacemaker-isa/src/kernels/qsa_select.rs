// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1151 QSA selector pair, one module
//! `qsa_select_pm_gfx1151` with the score symbol then the select symbol:
//! `indexed_attention_select_scores_rows16_f32_pm_gfx1151` (`qsa_score`) and
//! `indexed_attention_select_from_scores_pm_gfx1151` (`qsa_topk`), with the
//! kernarg ABI, grid and output bytes of the hipcc pair in
//! `kernels/src/tensor_ops.hip`. Every other target is refused.
//! `qsa_select_bf16_pm_gfx1151` contains only the BF16-pooled score variant
//! (exact widening, identical score arithmetic); its route reuses the F32 select image.
use crate::{Arch, Emitted};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind { Score, ScoreBf16, Select }
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::Score, Kind::Select];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub kind: Kind }

pub fn module(arch: Arch) -> String { format!("qsa_select_pm_{}", arch.name()) }
pub fn module_bf16(arch: Arch) -> String { format!("qsa_select_bf16_pm_{}", arch.name()) }

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    if spec.arch != Arch::Gfx1151 { return Err("qsa_select: built for gfx1151 only".into()) }
    match spec.kind {
        Kind::Score => super::qsa_score::emit(spec.arch),
        Kind::ScoreBf16 => super::qsa_score::emit_bf16(spec.arch),
        Kind::Select => super::qsa_topk::emit(spec.arch),
    }
}

/// Both symbols as one module, score first.
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Kind::ALL.into_iter().map(|kind| emit(Spec { arch, kind })).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

/// BF16-pooled score only; the select symbol is shipped in the F32 module.
pub fn emit_module_bf16(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(Spec { arch, kind: Kind::ScoreBf16 })?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module_bf16(arch))?;
    Ok((emitted, text, proof))
}
