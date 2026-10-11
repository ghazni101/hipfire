// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 / Qwen3.8-Flash-Next architecture foundations.
//!
//! Product registration remains fail-closed until the model's lifecycle
//! milestones are accepted. This crate owns strict configuration and
//! admission, typed weights and GPU state, bounded PLE storage, and native MTP
//! transactions. CPU reference equations are opt-in parity fixtures.
#![recursion_limit = "256"]

pub mod admission;
pub mod artifact;
pub mod bundle;
#[cfg(any(test, feature = "reference-parity"))]
pub mod cache_quality;
pub mod config;
pub mod expert_residency;
pub mod gpu_forward;
pub mod gpu_forward_lanes;
pub mod kv_backend;
pub mod lane;
pub mod lane_exec;
pub mod mtp_gpu;
pub mod mtp_lanes;
pub mod mtp_spec;
pub mod ops;
pub mod ple;
pub(crate) mod ple_stage;
pub mod program;
pub(crate) mod projection;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_forward;
#[cfg(any(test, feature = "reference-parity"))]
pub mod reference_mtp;
pub mod state;
#[cfg(any(test, feature = "reference-parity"))]
pub mod state_parity;
pub mod weights;

pub use admission::{
    AdmissionError, ArtifactFormat, EffectiveMesh, InputModality, MeshShape, QuantFormat,
    QuantGeometry, QuantTensorGeometry, Qwen4Admission, Qwen4AdmissionError, Qwen4AdmissionRequest,
    Qwen4Artifact, Qwen4ArtifactFormat, Qwen4Capabilities, Qwen4Capability, Qwen4QuantGeometry,
    RequestKind,
};
pub use artifact::{
    admit_hfqm_artifact, admit_inventory_headers, Qwen4ArtifactError, Qwen4HfqmArtifact,
    Qwen4InventoryReceipt, Qwen4PleRows, SidecarRows,
};
#[cfg(feature = "qwen4-dev-profiles")]
pub use artifact::admit_developer_inventory_headers;
pub use config::{
    ple_head_vocab_sizes, LayerType, Qwen4Config, Qwen4EquationSource, Qwen4FfnConfig,
    Qwen4MtpConfig, Qwen4PleConfig, Qwen4Profile, Qwen4SyntheticProfile, RecurrentStateDType,
    SourceDType, ARCHITECTURE_NAME, ARCH_ID, MODEL_TYPE, QWEN4_DEFAULT_CONTEXT, QWEN4_MAX_CONTEXT,
    TEXT_MODEL_TYPE,
};
pub use kv_backend::{qwen4_vmm_refusal, qwen4_vmm_supported, Qwen4ContextCommit, Qwen4KvBackend};
pub use ple::{
    PleHashMetadata, PleHistory, PleMetadataError, PleRowId, PLE_HEAD_COUNT, PLE_HEAD_OFFSETS,
    PLE_MAX_HEAD_COUNT,
    PLE_HEAD_VOCAB_SIZES, PLE_MULTIPLIERS, PLE_MULTIPLIER_COUNT, PLE_PADDED_ROWS,
    PLE_PADDING_MULTIPLE, PLE_ROW_WIDTH, PLE_VALID_ROWS,
};
pub use rdna_compute::tensor_ops::{GdnStateFormat, QsaKvFormat};
pub use state::{resolve_gdn_format, resolve_qsa_format, resolve_state_format, Qwen4StateFormat};
pub use weights::{Qwen4ManifestIdentity, Qwen4TierRecipe};
pub use gpu_forward_lanes::{WaveHead, WaveLane};
