// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The contract between `pm_loop` (driver: certify, load, ABBA, JSON) and a
//! production-generator kernel module (`iu4.rs`, `moe.rs`): knob parsing, shape
//! grammar, emit/link of the candidate, the shipped baseline, deterministic
//! synthetic inputs, kernarg packing, and the CPU reference.
//!
//! Included with `#[path = "pm_loop/workload.rs"] mod workload;`.
#![allow(dead_code)]

use rdna_compute::Gpu;
use serde_json::Value;
use std::collections::BTreeMap;

pub type Result<T, E = String> = std::result::Result<T, E>;

/// Which kernel a launch/output belongs to. `Baseline` is timing arm A,
/// `Candidate` arm B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Baseline,
    Candidate,
}

/// One kernel dispatch exactly as the production launcher issues it.
#[derive(Clone, Debug)]
pub struct Launch {
    /// Complete kernarg blob (explicit args; the driver passes it to
    /// `hip.launch_kernel_blob`, which fills the hidden block).
    pub kernarg: Vec<u8>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic LDS bytes.
    pub lds: u32,
}

/// A loadable code object (ELF `.co` or clang offload bundle `.hxaco`) and
/// the symbol the workload launches.
#[derive(Clone, Debug)]
pub struct Object {
    pub bytes: Vec<u8>,
    /// Kernel symbol launched by timing/correctness.
    pub launch_symbol: String,
    /// Every export of `bytes` (all certified for the candidate; resources
    /// are reported for each).
    pub exports: Vec<String>,
    /// JSON description: `{knobs:{..}}` for an emitted object, or
    /// `{artifact:"<path>"}` for a committed object.
    pub desc: Value,
    /// JSON provenance of the selection: `{source:"file:line", ..}`.
    pub source: Value,
}

/// Candidate-vs-CPU-reference result.
#[derive(Clone, Debug)]
pub struct RefReport {
    /// Human name of the reference (e.g. `cpu_iu4_fold_rn`).
    pub reference: String,
    pub max_abs_err: f64,
    pub tolerance: f64,
    /// `max_abs_err <= tolerance` and no NaN/Inf mismatch.
    pub pass: bool,
}

/// Raw `--key value` flags the driver did not consume. A module `take`s the
/// flags it understands and the driver errors on whatever is left.
#[derive(Clone, Debug, Default)]
pub struct Knobs(pub BTreeMap<String, String>);

impl Knobs {
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
    /// `take`, or `default`.
    pub fn take_or(&mut self, key: &str, default: &str) -> String {
        self.take(key).unwrap_or_else(|| default.to_owned())
    }
    /// Error naming every flag nobody consumed.
    pub fn finish(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unknown option(s) for this kernel: {}",
                self.0
                    .keys()
                    .map(|k| format!("--{k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }
}

/// One production-generator kernel under test. Constructed GPU-free (knob
/// parsing + shape grammar + candidate/baseline selection); `setup` then
/// allocates and fills the device inputs.
pub trait Workload {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    /// Target arch (`--arch`; `gfx1201` unless the workload supports more).
    fn arch(&self) -> &'static str;
    /// Parsed shape as JSON (every dimension the launch derives from).
    fn shape_json(&self) -> Value;
    /// Candidate knobs as JSON (the resolved Spec axes).
    fn candidate_knobs(&self) -> Value;
    /// Extra top-level record fields (e.g. `sparse`), merged by the driver.
    fn record_extra(&self) -> Vec<(String, Value)> {
        Vec::new()
    }

    /// Emit + link the candidate with the native writer (the path of
    /// `peacemaker-isa emit --co`). Errors (e.g. a closed fold axis) are
    /// reported verbatim by the driver.
    fn candidate(&self) -> Result<Object>;
    /// The shipped baseline (what production selects for `arch()`), or the
    /// `--base-*` knob variant. `source` names file:line / artifact + sha256.
    fn baseline(&self) -> Result<Object>;

    /// Allocate device buffers and upload deterministic synthetic inputs.
    fn setup(&mut self, gpu: &Gpu) -> Result<()>;
    /// Launch parameters for `arm` (kernarg points at that arm's buffers).
    fn launch(&self, arm: Arm) -> Launch;
    /// Restore `arm`'s accumulating state (e.g. `Y` for epi add; zero `Y`
    /// otherwise) to its pre-launch bytes. Called outside event intervals.
    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()>;
    /// Download `arm`'s output (the payload the correctness check covers).
    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>>;
    /// Compare a downloaded output to the CPU reference.
    fn check(&self, out: &[u8]) -> Result<RefReport>;
    /// Max absolute element difference between two outputs (interpreting the
    /// bytes in the kernel's output dtype); `inf` if any element differs in
    /// NaN-ness. 0.0 iff element-wise equal.
    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64>;

    /// `true` when correctness requires the candidate's output to be bitwise
    /// equal to the shipped baseline in addition to passing the CPU reference.
    fn requires_baseline_identity(&self) -> bool {
        false
    }

    /// `--emu`: run the candidate on the peacemaker emulator at a small
    /// shape if every instruction is qualified; otherwise
    /// `Value::String("refused (<reason>)")`. Never fabricate a result.
    fn emu(&self) -> Value;
}

/// Construction context the driver hands a [`ChainWorkload`].
#[derive(Clone, Copy, Debug)]
pub struct ChainCtx {
    /// `--arch` (`gfx1201` default): the target the chain is selected for; the
    /// driver refuses a device of another arch.
    pub arch: &'static str,
    /// `--null-control aa`: the candidate arm runs the BASELINE chain (same
    /// launchers, same selector, same loaded modules) into the candidate arm's
    /// own buffers, so the record is an A/A null of the whole chain.
    pub null_control: bool,
}

/// R3 complete-chain workload. Each arm is an ordered multi-kernel chain
/// enqueued on the default stream through `rdna_compute::Gpu`'s public
/// launchers (the baseline arm: the chain the shipped selector issues at this
/// shape/arch/residency), timed as ONE unit per launch slot (hipEvents around
/// the whole chain). Constructed GPU-free (shape grammar, knob parsing, arm
/// resolution); `setup` allocates and fills deterministic inputs.
///
/// Every chain workload is eager today (`mode() == "eager"`); graph replay is
/// a later additive mode.
pub trait ChainWorkload {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    /// Target arch (from [`ChainCtx::arch`]).
    fn arch(&self) -> &'static str;
    /// Execution mode of both arms (`"eager"`).
    fn mode(&self) -> &'static str {
        "eager"
    }
    /// Parsed shape as JSON (every dimension the chain derives from,
    /// including residency).
    fn shape_json(&self) -> Value;
    /// Candidate knobs as JSON.
    fn candidate_knobs(&self) -> Value;
    /// `{baseline: {...}, candidate: {...}}`: per arm the ordered stages
    /// (`[{stage, launcher, symbols, origin: "hip"|"pm", source}]`), the
    /// selector provenance (file:line of every predicate that picked a
    /// stage), residency, scratch bytes. Called after `setup` (selection can
    /// depend on the device).
    fn chain_json(&self) -> Value;
    /// Declared numeric contract: `{baseline, candidate, relation}` where
    /// `relation` is `"exact"` (the candidate must reproduce the baseline's
    /// bytes) or `"numeric"` (a different recipe: `identical_to_baseline` is
    /// reported honestly and each arm passes on its own reference).
    fn numeric_contract(&self) -> Value;
    /// PM code objects `arm` launches (every export is M7-certified by the
    /// driver for the candidate arm); empty when the arm launches no PM object.
    fn pm_objects(&self, arm: Arm) -> Result<Vec<Object>>;

    /// Allocate device buffers (two complete buffer sets, outputs and
    /// scratch, one per arm under the current binding) and upload
    /// deterministic synthetic inputs.
    fn setup(&mut self, gpu: &mut Gpu) -> Result<()>;
    /// Crossover binding of arms to buffer sets: `false` binds the baseline to
    /// set 0 and the candidate to set 1, `true` swaps them. `encode_chain`,
    /// `reset` and `outputs` follow the current binding. The driver flips it
    /// every ABBA round so a buffer-placement effect alternates sign instead of
    /// biasing one arm (an A/A with a fixed binding measured up to 0.9%).
    fn set_buffer_swap(&mut self, swapped: bool);
    /// Enqueue `arm`'s complete chain on the default stream. No host
    /// synchronization, allocation or upload inside (the driver brackets it
    /// with events).
    fn encode_chain(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()>;
    /// Restore `arm`'s accumulating state (e.g. the residual the combine adds
    /// into) to its pre-chain bytes. Called outside event intervals.
    fn reset(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()>;
    /// Download `arm`'s outputs (every byte the correctness check covers,
    /// including poisoned guard/tail regions).
    fn outputs(&self, gpu: &mut Gpu, arm: Arm) -> Result<Vec<u8>>;
    /// Compare `arm`'s downloaded outputs to that arm's CPU reference.
    fn reference(&self, arm: Arm, out: &[u8]) -> Result<RefReport>;
    /// Max absolute element difference between two output payloads; `inf`
    /// on a NaN-ness mismatch; 0.0 iff element-wise equal.
    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64>;
    /// `true` iff `numeric_contract().relation == "exact"`.
    fn requires_baseline_identity(&self) -> bool;
}
