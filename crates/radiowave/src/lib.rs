// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Radiowave owns the policy boundary between HIP source and LLVM/AMDGPU.
//! It injects reviewed source-level lowering rules, invokes hipcc, inspects the
//! emitted code object, and records enough evidence to reproduce the build.

mod arch;
pub mod atomics;
mod campaign;
mod contracts;
pub mod oracle;
pub mod partition;
pub mod recipes;
pub mod recipes_fp8;
pub mod toolchain;

pub use arch::{ArchProfile, CodeObjectIdentity, IsaVersion};
pub use campaign::{
    CAMPAIGN_SCHEMA_VERSION, CampaignError, CampaignEvent, CampaignLedger, CampaignPolicy,
    CampaignResult, CampaignStarted, CandidateRecord, CandidateSubmission, CandidateVerdict,
    DEFAULT_MAX_COMPLETED_GPU_BATTERIES_PER_TARGET, PromotionRecord, RecordDisposition,
};
pub use contracts::{
    RESOURCE_CONTRACT_SCHEMA_VERSION, ResourceAssessment, ResourceContract, ResourceRejection,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub const HIP_SUPPORT_HEADER: &str = include_str!("../include/radiowave/hip.h");
static SUPPORT_HEADER_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("source file does not exist: {0}")]
    MissingSource(PathBuf),
    #[error("{tool} exited with {status}\nstdout:\n{stdout}\nstderr:\n{stderr}")]
    ToolFailed {
        tool: String,
        status: String,
        stdout: String,
        stderr: String,
    },
    #[error("no HIP offload bundle target containing {arch} was found in {input}")]
    MissingBundleTarget { arch: String, input: PathBuf },
    #[error("Radiowave code-object certification failed: {0}")]
    InvalidCertification(String),
    #[error("invalid Radiowave oracle input: {0}")]
    InvalidOracle(String),
    /// Toolchain does not meet the ROCm 10.0 floor (HIP version gate).
    #[error("HIP version {found} is below required >= {required} (ROCm 10.0 floor)")]
    UnsupportedHipVersion { found: String, required: String },
    /// Version banner lacked the AMD clang marker (generic upstream clang).
    /// 10.0-only policy: radiowave requires amdclang / ROCm >= 10.0 (HIP >= 7.15).
    #[error("non-AMD clang toolchain (requires ROCm >= 10.0 / amdclang): {0}")]
    NonAmdClang(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wavefront {
    Wave32,
    Wave64,
}

impl Wavefront {
    pub const fn width(self) -> u32 {
        match self {
            Self::Wave32 => 32,
            Self::Wave64 => 64,
        }
    }
}

/// AMDGPU machine-scheduler policies which Radiowave can reproduce and
/// correctness-gate per code object. These are deliberately explicit rather
/// than hidden in an arbitrary `-mllvm` string.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerProfile {
    #[default]
    Default,
    MaxIlp,
    IterativeIlp,
    MemoryClause,
    PipelineIlp,
}

impl SchedulerProfile {
    pub const ALL: [Self; 5] = [
        Self::Default,
        Self::MaxIlp,
        Self::IterativeIlp,
        Self::MemoryClause,
        Self::PipelineIlp,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::MaxIlp => "max_ilp",
            Self::IterativeIlp => "iterative_ilp",
            Self::MemoryClause => "memory_clause",
            Self::PipelineIlp => "pipeline_ilp",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "default" => Some(Self::Default),
            "max-ilp" | "max_ilp" => Some(Self::MaxIlp),
            "iterative-ilp" | "iterative_ilp" => Some(Self::IterativeIlp),
            "memory-clause" | "memory_clause" => Some(Self::MemoryClause),
            "pipeline-ilp" | "pipeline_ilp" => Some(Self::PipelineIlp),
            _ => None,
        }
    }

    pub const fn llvm_args(self) -> &'static [&'static str] {
        match self {
            Self::Default => &[],
            Self::MaxIlp => &["-mllvm", "-misched=gcn-max-ilp"],
            Self::IterativeIlp => &["-mllvm", "-misched=gcn-iterative-ilp"],
            Self::MemoryClause => &[
                "-mllvm",
                "-misched=gcn-max-memory-clause",
                "-mllvm",
                "-amdgpu-max-memory-clause=4",
            ],
            Self::PipelineIlp => &[
                "-mllvm",
                "-misched=gcn-max-ilp",
                "-mllvm",
                "-amdgpu-schedule-relaxed-occupancy",
                "-mllvm",
                "-amdgpu-igrouplp-exact-solver",
                "-mllvm",
                "-enable-pipeliner",
            ],
        }
    }
}

/// Resolve the hipcc binary for ROCm 10.0+ layouts.
///
/// Order: non-empty `$HIPCC`, `/opt/rocm/core/bin/hipcc`,
/// `/opt/rocm/core-10.0/bin/hipcc`, then bare `"hipcc"` (PATH fallback).
pub fn resolve_hipcc() -> PathBuf {
    if let Some(value) = env::var_os("HIPCC") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    for candidate in ["/opt/rocm/core/bin/hipcc", "/opt/rocm/core-10.0/bin/hipcc"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("hipcc")
}

#[derive(Clone, Debug)]
pub struct CompileRequest {
    pub source: PathBuf,
    pub output: PathBuf,
    pub arch: String,
    pub wavefront: Wavefront,
    pub hipcc: PathBuf,
    pub working_directory: Option<PathBuf>,
    pub optimization_level: u8,
    pub fast_math: bool,
    pub scheduler_profile: SchedulerProfile,
    pub defines: Vec<String>,
    pub extra_args: Vec<OsString>,
    pub manifest: Option<PathBuf>,
    pub inspect: bool,
    /// Environment applied only to the hipcc invocation.
    ///
    /// This lets callers pin a non-default ROCm root without mutating the
    /// process environment used by inspection and replay.
    pub envs: Vec<(OsString, OsString)>,
}

impl CompileRequest {
    pub fn new(
        source: impl Into<PathBuf>,
        output: impl Into<PathBuf>,
        arch: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            output: output.into(),
            arch: arch.into(),
            wavefront: Wavefront::Wave32,
            hipcc: resolve_hipcc(),
            working_directory: None,
            optimization_level: 3,
            fast_math: true,
            scheduler_profile: SchedulerProfile::Default,
            defines: Vec::new(),
            extra_args: Vec::new(),
            manifest: None,
            inspect: true,
            envs: Vec::new(),
        }
    }

    pub fn wavefront(mut self, wavefront: Wavefront) -> Self {
        self.wavefront = wavefront;
        self
    }

    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }

    pub fn hipcc(mut self, hipcc: impl Into<PathBuf>) -> Self {
        self.hipcc = hipcc.into();
        self
    }

    pub fn scheduler_profile(mut self, profile: SchedulerProfile) -> Self {
        self.scheduler_profile = profile;
        self
    }

    pub fn define(mut self, define: impl Into<String>) -> Self {
        self.defines.push(define.into());
        self
    }

    pub fn manifest(mut self, path: impl Into<PathBuf>) -> Self {
        self.manifest = Some(path.into());
        self
    }
}

/// Describes a code object emitted by an external compiler driver which
/// Radiowave should inspect and bind to a certification manifest.
///
/// This is the runtime/JIT integration path: the owner keeps its existing
/// compiler and cache, while Radiowave owns the exact-object inspection and
/// hash binding consumed by replay.
#[derive(Clone, Debug)]
pub struct ExistingCodeObjectRequest {
    pub source: PathBuf,
    pub output: PathBuf,
    pub arch: String,
    pub wavefront: Wavefront,
    pub hipcc: PathBuf,
    pub command: Vec<String>,
    pub manifest: Option<PathBuf>,
    pub scheduler_profile: SchedulerProfile,
}

impl ExistingCodeObjectRequest {
    pub fn new(
        source: impl Into<PathBuf>,
        output: impl Into<PathBuf>,
        arch: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            output: output.into(),
            arch: arch.into(),
            wavefront: Wavefront::Wave32,
            hipcc: resolve_hipcc(),
            command: Vec::new(),
            manifest: None,
            scheduler_profile: SchedulerProfile::Default,
        }
    }

    pub fn wavefront(mut self, wavefront: Wavefront) -> Self {
        self.wavefront = wavefront;
        self
    }

    pub fn hipcc(mut self, hipcc: impl Into<PathBuf>) -> Self {
        self.hipcc = hipcc.into();
        self

[Showing lines 1-300 of 1780. Use :301 to continue]