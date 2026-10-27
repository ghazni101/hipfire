// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `peacemaker-receipt` schema (receipt v1 = radiowave `PeacemakerRecord`
//! schema 6; the `railgun` section makes it 7; the conditional indexed-argument
//! amendment makes it 8 / section 2). `railgun-cert` writes these
//! records offline; the runtime reads them through this crate, which depends
//! on no peacemaker crate (railgun design D12).
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA: &str = "peacemaker-receipt";
/// Receipt v1 (architecture.md §3) is schema 6; the `railgun` section is 7;
/// guarded/indexed facts (design A §A9.2) are 8. A v7 receipt is never
/// reinterpreted: unknown versions fail admission.
pub const RECEIPT_SCHEMA_VERSION: u32 = 8;
pub const RAILGUN_SECTION_VERSION: u32 = 2;

/// `KernelRecord.status` values.
pub const STATUS_PROVEN: &str = "proven";
pub const STATUS_UNKNOWN: &str = "unknown";
/// Conditional: valid only for the guarded case, and only once every
/// `required_witnesses` kind has a matching runtime bind witness.
pub const STATUS_CONDITIONAL: &str = "conditional";

/// One launch/argument guard of a guarded case (§A9.1). `kind` is one of
/// `kernarg_eq` (kernarg bytes `[arg_offset, arg_offset+size)` == `value`,
/// decimal; a null pointer is `size 8 value 0`), `grid_dim` / `block_dim`
/// (`arg_offset` = axis 0/1/2, `size` 0, `value` = work-group count / size).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GuardRecord {
    pub kind: String,
    pub arg_offset: u32,
    pub size: u32,
    pub value: String,
}

/// `IndexLoad`: an immutable `I32` index loaded from a pointer argument.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct IndexLoadRecord {
    pub pc: u32,
    pub source_arg_offset: u32,
    /// `i32`.
    pub element: String,
    /// Byte-offset expression, e.g. `4*workgroup_id_y`.
    pub offset_expr: String,
    /// Extension/arithmetic semantics of the consuming address
    /// (`sext32`/`zext32`/...).
    pub extension: String,
    pub class: String,
    pub path: String,
    pub cache: String,
    pub width: u32,
}

/// `IndexedArgEffect`: one access whose address is `direct base + expr`.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct IndexedEffectRecord {
    pub pc: u32,
    pub base_arg_offset: u32,
    /// `read`, `write`, `read_write`.
    pub mode: String,
    pub class: String,
    pub path: String,
    /// Raw cache bits, `th=.. scope=.. glc=.. slc=.. dlc=.. nv=..`.
    pub cache: String,
    /// Access width in bytes.
    pub width: u32,
    /// Checked 64-bit address expression, canonical text.
    pub address_expr: String,
    pub index_loads: Vec<IndexLoadRecord>,
    /// Guard `kind@arg_offset` names this effect depends on.
    pub guards: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub crate_version: String,
    pub git_sha: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelIdentity {
    pub symbol: String,
    /// `[entry_va, size]` of the kernel's `.text` range.
    pub text_range: [u64; 2],
    pub text_sha256: String,
    pub kd_sha256: String,
    pub metadata_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Check {
    pub id: String,
    pub status: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub schema: String,
    pub schema_version: u32,
    /// SHA-256 of the input file (HIP bundle or ELF): the lookup key.
    pub object_sha256: String,
    pub device_elf_sha256: String,
    pub arch: String,
    /// Tier D (design D13): diagnostic lift + byte identity, never strict.
    pub strict: bool,
    pub tier: String,
    pub lifter: Tool,
    pub certifier: Tool,
    pub round_trip_proof_sha256: String,
    pub kernels: Vec<KernelIdentity>,
    pub checks: Vec<Check>,
    pub obligations: Vec<String>,
    pub railgun: RailgunSection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RailgunSection {
    pub section_version: u32,
    pub analysis: String,
    pub assumptions: Vec<String>,
    pub kernels: Vec<KernelRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnresolvedRecord {
    pub pc: Option<u32>,
    pub rule: String,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SideRecord {
    pub kernarg_segment: u32,
    pub module_global: u32,
    pub abi_packet: u32,
    pub local_flat: u32,
    pub scratch: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImplicitRecord {
    pub hidden_loaded: Vec<String>,
    pub hidden_grid: Vec<String>,
    pub hidden_grid_live: Vec<String>,
    /// The kernel loads a `hidden_block_count_{x,y,z}` dword (the words
    /// `replay.rs` writes from the recorded grid and never repatches).
    pub loads_block_count: bool,
    pub loads_group_size: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgRecord {
    pub offset: u32,
    pub size: u32,
    pub index: Option<usize>,
    pub name: String,
    pub value_kind: String,
    /// A4: `pointer` (a resolved access base), `integer` (loaded, never a
    /// base) or `not_loaded`.
    pub role: String,
    /// A1: `read`, `write`, `read_write`, or absent when never accessed.
    pub mode: Option<String>,
    pub bound: Option<String>,
    pub classes: Vec<String>,
    pub paths: Vec<String>,
    pub cache_bits: Vec<String>,
    pub accesses: usize,
    /// Some access resolved to more than one argument (e.g. `s_cselect_b64`).
    pub shared: bool,
    /// A3: read caches that can hold data this kernel read from the argument.
    pub read_cache: String,
    /// A4 sinks over the argument's dwords.
    pub sinks: Vec<String>,
    pub loaded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DwordRecord {
    pub offset: u32,
    pub loaded: bool,
    pub sinks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArgCheck {
    pub offset: u32,
    pub name: String,
    pub a1: Option<String>,
    /// radiowave-conservative access (`read_only`/`write_only`/`read_write`).
    pub metadata: String,
    /// `.actual_access` was present (not the conservative default).
    pub explicit: bool,
    pub status: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Differential {
    /// `sidecar` (runtime `*.radiowave.json`), `object-metadata` (the code
    /// object's own note, through the same radiowave rule) or `none`.
    pub source: String,
    pub status: String,
    pub args: Vec<ArgCheck>,
    /// Radiowave's text-derived read-cache class (sidecar only), next to A3.
    pub radiowave_mutable_read_cache: Option<String>,
    /// The guarded case this differential was computed under (empty for an
    /// unconditional summary). A passing differential with guards is a check
    /// of the guarded summary only, never an unconditional proof.
    pub guards: Vec<GuardRecord>,
    /// The summary compared was the guarded (Conditional) one.
    pub guarded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRecord {
    pub symbol: String,
    pub status: String,
    pub unresolved: Vec<UnresolvedRecord>,
    pub kernarg_size: u32,
    pub read_cache: String,
    pub dynamic_kernarg_reads: bool,
    pub taint_overflow: bool,
    pub side: SideRecord,
    pub implicit: ImplicitRecord,
    pub args: Vec<ArgRecord>,
    pub dwords: Vec<DwordRecord>,
    pub differential: Differential,
    /// Open peacemaker analysis obligations by rule id (diagnostic tier).
    pub pm_obligations: BTreeMap<String, usize>,
    /// `edit::analyze` failed; facts come from the kernargs pass alone.
    pub pm_analyze_error: Option<String>,
    /// Diagnostic only, conditional records: the unguarded analysis' refusals
    /// (what the guarded case replaced). Empty otherwise; never a fact source.
    pub unguarded_unresolved: Vec<UnresolvedRecord>,
    /// Guards of the guarded case; non-empty iff `status == "conditional"`.
    pub guards: Vec<GuardRecord>,
    /// Indexed argument effects of the guarded case.
    pub indexed_effects: Vec<IndexedEffectRecord>,
    /// Witness kinds the runtime must hold before the executable may use the
    /// effects (`indexed_bind`); non-empty iff conditional.
    pub required_witnesses: Vec<String>,
    /// SHA-256 binding object, kernel, status, guards, effects and witnesses
    /// ([`case_digest`]); empty unless conditional.
    pub case_digest: String,
    /// #867 kernel guard: the kernarg offset of the ControlWord pointer whose
    /// clear flag (`u32` zero) ends every wave before any other instruction
    /// that touches memory (`railgun_cert::guard_flag`). Absent from receipts
    /// written before the field and from unguarded kernels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_flag: Option<u32>,
}

/// The guarded-case digest under the exact object receipt (§A9.2).
pub fn case_digest(
    object_sha256: &str,
    device_elf_sha256: &str,
    arch: &str,
    k: &KernelRecord,
) -> String {
    let bytes = serde_json::to_vec(&(
        "railgun-case-v1",
        object_sha256,
        device_elf_sha256,
        arch,
        &k.symbol,
        &k.status,
        k.kernarg_size,
        &k.guards,
        &k.indexed_effects,
        &k.required_witnesses,
    ))
    .expect("case digest input serialises");
    crate::sha256_hex(&bytes)
}

impl KernelRecord {
    /// Fail-closed structural check of the guarded fields against `status`.
    /// A conditional record needs guards, effects, witnesses and a matching
    /// digest; a proven/unknown record must carry none of them (a v1 row is
    /// never promoted); any other status is rejected.
    pub fn validate_case(
        &self,
        object_sha256: &str,
        device_elf_sha256: &str,
        arch: &str,
    ) -> Result<(), String> {
        let empty = self.guards.is_empty()
            && self.indexed_effects.is_empty()
            && self.required_witnesses.is_empty()
            && self.case_digest.is_empty();
        match self.status.as_str() {
            STATUS_PROVEN | STATUS_UNKNOWN => {
                if !empty {
                    return Err(format!(
                        "{}: status {} carries guarded-case fields",
                        self.symbol, self.status
                    ));
                }
                Ok(())
            }
            STATUS_CONDITIONAL => {
                if self.guards.is_empty()
                    || self.indexed_effects.is_empty()
                    || self.required_witnesses.is_empty()
                {
                    return Err(format!(
                        "{}: conditional record lacks guards, indexed effects or required witnesses",
                        self.symbol
                    ));
                }
                if !self.differential.guarded || self.differential.guards != self.guards {
                    return Err(format!(
                        "{}: differential does not report the record's guards",
                        self.symbol
                    ));
                }
                if self.case_digest != case_digest(object_sha256, device_elf_sha256, arch, self) {
                    return Err(format!("{}: case digest does not match", self.symbol));
                }
                Ok(())
            }
            other => Err(format!("{}: unknown A1 status {other:?}", self.symbol)),
        }
    }
}
