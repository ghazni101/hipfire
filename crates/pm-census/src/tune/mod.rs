// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4/R5/R6 developer-only autotuner: bounded legal-space search over existing
//! PM generator knobs (never edited source), an evidence-keyed result store with
//! a deterministic exported table, and the promotion gate pipeline.
//!
//! * [`domain`] (R4): `enumerate(&WorkloadCaseV1, node, tokens)` -> the ordered
//!   candidate domain of one corpus node; no GPU probing.
//! * [`run`] (R4): the phase driver (offline legality screen -> A/A noise floor ->
//!   discovery -> held-out confirmation in fresh processes) over `pm_loop`
//!   subprocesses, with a resumable ledger and work/time budgets.
//! * [`store`] (R5): append-only JSONL store of [`TuneRecordV1`], deterministic
//!   [`TableV1`] export and exact-key `lookup`. Only promoted rows are exported
//!   as selectable; the product reads them only via `HIPFIRE_PM_TUNE_TABLE`.
//! * [`promote`] (R6/R7): `can_promote(&TuneEvidenceV1, &ClaimPolicyV1)`; the
//!   `kld_state` stage judges model-level KLD / state-equality evidence.
//! * [`kld`] (R7): HFKSEQ reader, paired per-chunk KLD statistics and the
//!   `pm-census tune kld|promote|trial-table` CLI.
//!
//! The types below are the frozen contract between those modules.

pub mod domain;
pub mod kld;
pub mod promote;
pub mod run;
pub mod store;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// sha256 (hex) of [`crate::noise::canonical`] JSON (sorted keys, no whitespace).
pub fn canonical_sha256(v: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(crate::noise::canonical(v).as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub const DOMAIN_SCHEMA: &str = "hipfire.tune_domain.v1";
pub const RECORD_SCHEMA: &str = "hipfire.tune_record.v1";
pub const TABLE_SCHEMA: &str = "hipfire.tune_table.v1";

/// One point of the legal knob space, exactly the `pm_loop` arguments that
/// select it (`--kernel K --arch A --shape S --<knob> V ...`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CandidateSpecV1 {
    pub kernel: String,
    pub arch: String,
    pub shape: String,
    pub knobs: BTreeMap<String, String>,
}

impl CandidateSpecV1 {
    /// First 16 hex of sha256 over the canonical JSON of the spec.
    pub fn id(&self) -> String {
        let v = serde_json::to_value(self).expect("spec serializes");
        canonical_sha256(&v)[..16].to_owned()
    }
    /// `pm_loop` arguments selecting this candidate (driver flags excluded).
    pub fn pm_loop_args(&self) -> Vec<String> {
        let mut a = vec![
            "--kernel".into(),
            self.kernel.clone(),
            "--arch".into(),
            self.arch.clone(),
            "--shape".into(),
            self.shape.clone(),
        ];
        for (k, v) in &self.knobs {
            a.push(format!("--{k}"));
            a.push(v.clone());
        }
        a
    }
}

/// The enumerated domain of one corpus node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainV1 {
    pub schema: String,
    pub case_id: String,
    pub node_index: usize,
    pub node_symbol: String,
    pub kernel: String,
    pub arch: String,
    pub shape: String,
    /// Knob value set of the shipped incumbent selection (the corpus node's symbol).
    pub incumbent: CandidateSpecV1,
    /// Deterministic order; includes the incumbent's generator twin.
    pub candidates: Vec<CandidateSpecV1>,
    /// How the shape was derived from the node (params checked, presets used).
    pub derivation: Vec<String>,
    /// sha256 of the canonical JSON of `(kernel, arch, shape, incumbent, candidates)`.
    pub domain_hash: String,
}

/// The fields of one `pm_loop` record the gates read (parsed, not trusted blindly:
/// the raw record path is kept so the evidence can be re-read).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptV1 {
    pub candidate_id: String,
    /// `offline`, `aa`, `discovery` or `confirm`.
    pub phase: String,
    pub replicate: u32,
    pub record_path: String,
    pub exit_code: i32,
    /// `error.stage` of the record, if any.
    pub error_stage: Option<String>,
    pub error_message: Option<String>,
    pub m7_status: Option<String>,
    pub correctness_pass: Option<bool>,
    pub identical_to_baseline: Option<bool>,
    pub baseline_launch_symbol: Option<String>,
    pub candidate_launch_symbol: Option<String>,
    pub baseline_co_sha256: Option<String>,
    pub candidate_co_sha256: Option<String>,
    /// `resources[].code_sha256` of the launch symbol of each arm.
    pub baseline_code_sha256: Option<String>,
    pub candidate_code_sha256: Option<String>,
    pub separable: Option<bool>,
    pub direction: Option<String>,
    pub sign_p: Option<f64>,
    pub block_ratio: Option<f64>,
    pub material: Option<bool>,
    pub rounds_run: Option<u64>,
    /// `timing.vs_noise_floor.floor` / `.exceeds` (None when no matching A/A group).
    pub noise_floor: Option<f64>,
    pub exceeds_noise_floor: Option<bool>,
    pub device_uuid: Option<String>,
    pub fabric_job: Option<String>,
}

/// Everything `can_promote` judges for one candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneEvidenceV1 {
    pub candidate: CandidateSpecV1,
    pub discovery: AttemptV1,
    pub confirms: Vec<AttemptV1>,
    /// A/A floor (q95 |block_ratio - 1|) of the incumbent at this shape.
    pub noise_floor: Option<f64>,
    pub aa_runs: usize,
    /// Model-level KLD / state-equality evidence (see [`kld`]).
    #[serde(default)]
    pub kld_state: Option<KldStateEvidenceV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimPolicyV1 {
    pub sign_p_max: f64,
    pub min_effect: f64,
    /// Fresh-process confirmations required after discovery.
    pub confirm_processes: usize,
    /// A/A processes required for the noise floor.
    pub aa_processes: usize,
    /// Admit candidates whose output differs from the incumbent (numeric), subject
    /// to the `kld_state` stage.
    #[serde(default = "default_admit_numeric")]
    pub admit_numeric: bool,
    /// Minimum `max_chunks` (and paired n) of a promotion-grade KLD run.
    #[serde(default = "default_kld_min_chunks")]
    pub kld_promotion_min_chunks: usize,
    /// Max allowed `ci95_hi` of the mean per-chunk KLD increase (candidate - incumbent).
    #[serde(default = "default_kld_margin")]
    pub kld_margin: f64,
}

impl Default for ClaimPolicyV1 {
    fn default() -> Self {
        Self {
            sign_p_max: 1e-3,
            min_effect: 0.005,
            confirm_processes: 1,
            aa_processes: 3,
            admit_numeric: true,
            kld_promotion_min_chunks: 24,
            kld_margin: 0.0005,
        }
    }
}

/// A refused promotion: the first failing stage, its reason, and every stage
/// that passed before it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    pub stage: String,
    pub reason: String,
    pub passed: Vec<String>,
}

/// A granted promotion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionV1 {
    pub passed: Vec<String>,
    pub speedup: f64,
}

/// Store key: any change of any field is a different key (a miss).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TuneKeyV1 {
    pub arch: String,
    pub kernel: String,
    pub shape: String,
    /// `exact` for code whose output is bit-identical to the incumbent.
    pub numeric_contract: String,
    /// `eager` today.
    pub mode: String,
    pub baseline_code_sha256: String,
    pub candidate_co_sha256: String,
    pub candidate_code_sha256: String,
    /// hipfire commit of the generator/checker that produced the candidate.
    pub commit: String,
    /// sha256 of the `pm_loop` binary that measured it (toolchain identity).
    pub pm_loop_sha256: String,
}

/// One search outcome for one key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneRecordV1 {
    pub schema: String,
    pub key: TuneKeyV1,
    pub domain_hash: String,
    pub candidate: CandidateSpecV1,
    /// `winner` (separable, confirmed) or `not_winner`.
    pub verdict: String,
    pub evidence: TuneEvidenceV1,
    pub promotion: Result<PromotionV1, Refusal>,
}

/// A row the runtime would be allowed to select (only promoted records).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotedVariantV1 {
    pub key: TuneKeyV1,
    pub candidate: CandidateSpecV1,
    pub speedup: f64,
    /// What the product reader needs to select this row; `None` in tables built
    /// without a code-object directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<ProductSelectV1>,
}

/// Deterministic export of a store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableV1 {
    pub schema: String,
    /// Sorted by key; promoted rows only.
    pub rows: Vec<PromotedVariantV1>,
    /// Sorted `(key, refusal)` of winner records that were refused promotion.
    pub refused: Vec<(TuneKeyV1, Refusal)>,
    /// Store lines that did not parse as a `TuneRecordV1` (`line N: error`); never rows.
    pub malformed: Vec<String>,
    /// sha256 of the canonical JSON of `rows`.
    pub rows_digest: String,
    /// `kld_trial` for a single-row trial table used only by a KLD run; exported
    /// store tables leave it `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
}

fn default_admit_numeric() -> bool {
    true
}
fn default_kld_min_chunks() -> usize {
    24
}
fn default_kld_margin() -> f64 {
    0.0005
}

/// One `eval_hipfire` run (JSON written by the driver script).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KldRunV1 {
    /// `incumbent` or `candidate`.
    pub role: String,
    pub model_path: String,
    pub model_sha256: String,
    pub reference_path: String,
    pub reference_sha256: String,
    pub evaluator_path: String,
    pub evaluator_sha256: String,
    pub hipfire_commit: String,
    /// Must be `prefill`.
    pub scoring_mode: String,
    pub max_chunks: usize,
    pub kv_mode: String,
    pub kv_v: String,
    pub env: BTreeMap<String, String>,
    pub fresh_home: bool,
    pub tune_table_path: Option<String>,
    pub tune_table_sha256: Option<String>,
    /// Candidate role: must equal the attempt's `candidate_co_sha256`.
    pub selected_code_object_sha256: Option<String>,
    pub kldseq_path: String,
    pub kldseq_sha256: String,
    pub device_uuid: String,
    /// Fabric job id when known (the hold's child does not see it).
    #[serde(default)]
    pub fabric_job: Option<String>,
    /// Slice mean KLD parsed from the evaluator's stderr.
    pub slice_mean_kld: f64,
}

/// Paired per-chunk mean-KLD statistics, candidate - incumbent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedKldV1 {
    pub n: usize,
    pub mean_diff: f64,
    pub sd_diff: f64,
    pub ci95_lo: f64,
    pub ci95_hi: f64,
}

/// Model-level KLD / state-equality evidence for one candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KldStateEvidenceV1 {
    /// `screen` or `promotion`.
    pub grade: String,
    pub incumbent: KldRunV1,
    pub candidate: KldRunV1,
    pub paired: PairedKldV1,
    /// kldseq bytes of both runs are identical.
    pub exact: bool,
}

/// How the product reader selects a promoted row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProductSelectV1 {
    pub symbol: String,
    pub arch: String,
    pub m: u64,
    pub k: u64,
    pub incumbent_image_sha256: String,
    /// Relative to the table file's directory (absolute also accepted).
    pub code_object: String,
    pub code_object_sha256: String,
}
