// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed Qwen4 manifest declarations and GPU-owned assembly.
//!
//! The names in this module are the names emitted by the pinned
//! `Qwen/Qwen3.8-Flash-Next` checkpoint.  In particular, the decoder uses the
//! `model.language_model.*` namespace, QSA keeps separate q/k/v projections,
//! and the PLE table is a set of numerically suffixed external rows.  The
//! runtime's [`WeightLoadTransaction`] remains the only allocation owner until
//! this module finalizes handles into [`Qwen4Weights`].

use crate::config::{LayerType, Qwen4Config, Qwen4FfnConfig, Qwen4PleConfig, Qwen4Profile};
use hipfire_runtime::external_rows::RowEncoding;
use hipfire_runtime::model_source::SourceRangeDescriptor;
#[cfg(test)]
use hipfire_runtime::weight_manifest::WeightResidency;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, ExpertSourceLayout, PinTarget, PlacementHint, ShardPolicy, StateEntry,
    StateKind, WeightEntry,
};
use hipfire_runtime::weight_store::{TakenWeight, WeightHandle, WeightLoadTransaction};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// The routed experts' declared targets: gate/up at the aligned-K group width
/// and down at the row-local one.
///
/// The E8 lattice format is deliberately *not* used here.  It rotates and
/// encodes in 256-wide FWHT segments, so `K % 256 == 0` is a precondition of
/// both the encoder and the decode kernels (`groups_per_row = K / 256`), and
/// `experts.down_proj` reduces over `moe_intermediate_size = 640` — three
/// quarters of a segment short.  The row-local MQ4G128V2 group exists for
/// exactly this case, and both tiers cost the same 4.25 bits per weight, so the
/// expert family keeps the pair the qwen4 typed route's kernels are built on.
pub const ROUTED_GATE_UP_DTYPE: DType = DType::MQ4G256V2;
pub const ROUTED_DOWN_DTYPE: DType = DType::MQ4G128V2;
pub const PLE_SHARD_ROWS: usize = 2_500_012;
pub use crate::ple::{PLE_HEAD_COUNT, PLE_ROW_WIDTH};
pub const PLE_SHARD_COUNT: usize = 128;
/// Upper bound on the PLE shard tensors a profile may declare, checked before
/// the manifest allocates anything proportional to the shard count.
const MAX_PLE_SHARD_TENSORS: usize = 65_536;

/// Packed formats this family's planner recognises.
///
/// A declaration outside this set is refused by name rather than demoted to
/// BF16, so the predicate is a recognition list, not the tier policy: the
/// policy lives in [`qwen4_matrix_dtype`].
fn qwen4_quantized_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2 | DType::MQ4G128V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA | DType::Q8_0
    )
}

/// Payloads that stay source-exact BF16 even though they are rank-2 matrices.
///
/// The Qwen4 family quantizes every matrix whose consumers read a quantized
/// weight — the trunk projections, the shared expert, the router, the PLE
/// key/value projections and the embedding/head — mirroring the canonical
/// `qwen3.6-35b-a3b` layout.  The MTP connector matrices are the exception:
/// their owner (`mtp_gpu`) still binds source BF16, so declaring them
/// quantized here would produce an artifact nothing can read.
const QWEN4_SOURCE_EXACT_SUFFIXES: &[&str] = &["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"];

/// Payload classes that stay source-exact BF16 despite being rank-2 matrices.
///
/// The hyper-connection mixer is four output rows over a `hc_count * hidden`
/// reduction — the thinnest matrix in the network against the widest activation
/// — so it keeps source bytes rather than being packed for a rounding error's
/// worth of traffic.
const QWEN4_SOURCE_EXACT_MARKERS: &[&str] = &["hyper_connection"];

/// The classes the repo's MoE recipes hold at eight bits rather than four.
///
/// Every family's embedding arm ships Q8 (never MQ4), and the quantizer's MoE
/// default promotes the whole fixed tier — attention, lm_head, embed, router —
/// to Q8F16 (qt=3, 34 bytes per 32 weights) because it is quality-critical and
/// small relative to the routed experts.  `qwen3.6-35b-a3b.mq6` ships its
/// lm_head, embed and router at qt=3 while its experts carry the six-bit tier.
const QWEN4_Q8_MARKERS: &[&str] = &["embed_tokens.weight"];

/// The untied 248k-vocab head ships at the trunk's six-bit tier (MQ6G256V2):
/// 0.50 GB read per token instead of Q8F16's 0.68, measured on the decode
/// route at +0.0005 KLD against the Q8F16 head (2026-09-28).  Four bits cost
/// +0.014 KLD there, so six is the floor.
///
/// The other matrices decode streams at BF16 (HC read, shared expert) stay
/// BF16 here: decode reads load-time Q8_0 copies of them
/// (`Qwen4GpuForward::decode_q8`) while prefill keeps the BF16 source — Q8 in
/// the file cost the prefill route +0.0021 KLD (HC) and short-prompt prefill
/// 46% (shared expert, no fast exact multi-row Q8 GEMM) for no decode speed.
pub(crate) const QWEN4_LM_HEAD: &str = "lm_head.weight";

fn qwen4_q8_dtype(name: &str) -> bool {
    QWEN4_Q8_MARKERS.iter().any(|marker| name.ends_with(marker))
}

fn qwen4_quantizable_matrix(name: &str, shape: &[usize]) -> bool {
    if shape.len() == 3 {
        return name.ends_with(".experts.down_proj") || name.ends_with(".experts.gate_up_proj");
    }
    if shape.len() != 2
        || QWEN4_SOURCE_EXACT_SUFFIXES
            .iter()
            .any(|s| name.ends_with(s))
        || QWEN4_SOURCE_EXACT_MARKERS
            .iter()
            .any(|marker| name.contains(marker))
    {
        return false;
    }
    // The packed set is the wide attention/GDN projections, which dominate
    // decode weight traffic.  The shared expert, the router, and the PLE
    // projections stay source-exact: the shared expert is as wide as any
    // trunk matrix but rides the sealed-MoE route, and the router decides
    // expert selection from 512 logits.
    if !name.contains(".linear_attn.") && !name.contains(".self_attn.") {
        return false;
    }
    // The eight-bit block format carries any K that is a multiple of its
    // 32-weight block, so alignment beyond the admission boundary above is not
    // a packing precondition.  A K that is not a multiple of 128 stays BF16.
    shape.last().is_some_and(|k| k % 128 == 0)
}

/// The tier each packed class ships at.
///
/// Routed experts keep the four-bit MQ4 family: their error is averaged over the
/// ten experts a token selects, they are where nearly all of the weight traffic
/// is (512 experts x two matrices per layer), and the qwen4 typed route's
/// grouped kernels are built on this exact qt44/qt53 pair.  The rank-2 trunk
/// carries eight bits instead — a dense projection writes its error straight
/// into the residual stream, and this family has already measured what a
/// lossier tier does there: a four-bit trunk opened a reasoning block the model
/// could not close.  Q8F16 costs 8.5 bpw but needs no rotation basis at all, and
/// the trunk is a rounding error next to the experts.
fn qwen4_matrix_dtype(shape: &[usize]) -> DType {
    let k = shape.last().copied().unwrap_or_default();
    if shape.len() == 3 {
        // gate/up reduces over `hidden` (256-aligned) and takes the aligned-K
        // group; down reduces over `moe_intermediate_size` and takes the
        // row-local one.
        if k % 256 == 0 {
            ROUTED_GATE_UP_DTYPE
        } else {
            ROUTED_DOWN_DTYPE
        }
    } else {
        DType::Q8_0
    }
}

fn qwen4_target_dtype(name: &str, shape: &[usize], requested: DType) -> DType {
    if matches!(requested, DType::BF16 | DType::Q8_0) {
        if name == QWEN4_LM_HEAD {
            return DType::MQ6G256V2;
        }
        if qwen4_q8_dtype(name) {
            return DType::Q8_0;
        }
    }
    if qwen4_quantizable_matrix(name, shape)
        && (requested == DType::BF16 || qwen4_quantized_dtype(requested))
    {
        qwen4_matrix_dtype(shape)
    } else {
        requested
    }
}

/// Q8 targets accept a BF16 source (the checkpoint) or an already-converted
/// Q8 payload, and nothing else: any other packed family is a different
/// geometry and would be misread here.
fn qwen4_q8_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0])
}

/// Routed experts accept the checkpoint's BF16 or the published four-bit
/// payload at either declared geometry.
fn qwen4_quant_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, ROUTED_GATE_UP_DTYPE, ROUTED_DOWN_DTYPE])
}

/// The trunk's wide attention/GDN projections are the one packed class whose
/// tier the *artifact* chooses rather than this build.
///
/// Both rungs describe the same logical matrix and load as the same resident
/// handle: the six-bit rung carries MQ6G256V2 (qt=47) and the eight-bit rung
/// carries Q8F16 (qt=3), which is the tier this build declares.  The loader
/// reads the tier the artifact actually ships and derives every row geometry
/// from that tag, so the source contract admits either payload.  Naming only
/// the tier this build happens to write would make a published rung of the
/// other tier unloadable by the very tree that produced it.
///
/// A third rung is admitted for the opt-in dense IU4 trunk
/// (`HIPFIRE_QWEN4_TRUNK_IU4`): symmetric MQ4G256V2 (qt=44), the aligned-K
/// 256 group the trunk's GEMV (`gemv_mq4g256v2`) and the dense IU4 SET
/// kernels read. Other four-bit families stay out: their group width is one
/// the trunk's decode kernels do not read, and their admission would only
/// move the failure to the artifact boundary.
fn qwen4_trunk_matrix_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![
        DType::BF16,
        DType::Q8_0,
        DType::MQ6G256V2,
        DType::MQ4G256V2,
    ])
}

/// True for the rank-2 trunk attention/GDN matrices — the class
/// [`qwen4_trunk_matrix_source`] governs.
fn qwen4_trunk_matrix(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && qwen4_quantizable_matrix(name, shape)
}

/// The packed tier the rank-2 trunk matrices ship at.
///
/// `HIPFIRE_QWEN4_TRUNK_TIER=mq6` puts the wide attention/GDN projections on
/// the aligned-K six-bit group (MQ6G256V2, qt=47, 6.25 bpw) instead of the
/// eight-bit recipe this build declares by default (Q8F16, qt=3, 8.5 bpw).
/// Any other value — an unset variable included — selects that default, so the
/// knob only ever adds the six-bit rung and every recipe and artifact written
/// so far keeps its meaning.
///
/// The producer and the loader reach the declaration through the same
/// [`Qwen4Manifest::build`], and the trunk's source contract
/// ([`qwen4_trunk_matrix_source`]) admits both rungs, so each build loads the
/// other's artifact.  The MTP namespace declares its own targets through
/// [`qwen4_target_dtype`] and never reads this.
fn qwen4_trunk_tier(value: Option<&str>) -> DType {
    match value {
        Some("mq6") => DType::MQ6G256V2,
        _ => DType::Q8_0,
    }
}

/// `HIPFIRE_QWEN4_MTP_TIER=source` declares the rank-2 MTP matrices at their
/// source's own BF16 instead of the eight-bit recipe the shared matrix
/// classifier otherwise picks for them.
///
/// Scratch measurements found no acceptance lift from this tier alone; it
/// remains an opt-in comparison against the published Q8F16 MTP projections.
/// Unset, the existing recipe is unchanged, and the producer and loader use
/// the same manifest declaration.
fn qwen4_mtp_source_tier(value: Option<&str>) -> bool {
    matches!(value, Some("source"))
}

thread_local! {
    /// Test-only override so a manifest assertion can exercise the knob without
    /// a process-global environment write.  Production always reads the env.
    static MTP_SOURCE_TIER_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The predicate the manifest actually consults, so a test can prove the knob
/// *reaches* the five rank-2 MTP attention matrices rather than only parsing.
fn mtp_source_tier_active() -> bool {
    MTP_SOURCE_TIER_OVERRIDE
        .with(|value| value.get())
        .unwrap_or_else(|| {
            qwen4_mtp_source_tier(
                hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_TIER")
                    .ok()
                    .as_deref(),
            )
        })
}

/// The declared target of one entry, with the selected trunk tier applied.
///
/// Only the rank-2 trunk attention/GDN class moves, and only when the request
/// was packed in the first place: a request that [`qwen4_target_dtype`] leaves
/// unpacked (F32, or a class this family keeps source-exact) keeps its dtype.
fn qwen4_trunk_target(name: &str, shape: &[usize], requested: DType, trunk_tier: DType) -> DType {
    let dtype = qwen4_target_dtype(name, shape, requested);
    if dtype == DType::Q8_0 && qwen4_trunk_matrix(name, shape) {
        trunk_tier
    } else {
        dtype
    }
}

/// Metadata key naming the recipe a developer artifact was written under.
/// A Flash-Next artifact's key predates the typed recipe and carries no `id`;
/// only a synthetic profile reads or requires one.
pub const RECIPE_METADATA_KEY: &str = "qwen4_recipe";
/// `id` of [`Qwen4TierRecipe::SyntheticSourceBf16`] in artifact metadata.
pub const SYNTHETIC_SOURCE_BF16_RECIPE_ID: &str = "synthetic-source-bf16";
/// `id` of [`Qwen4TierRecipe::SyntheticPacked`] in artifact metadata.
pub const SYNTHETIC_PACKED_RECIPE_ID: &str = "synthetic-packed";

impl Qwen4TierRecipe {
    /// Stable artifact-metadata id of a synthetic recipe; `None` for the
    /// Flash-Next recipe, which is never selected by metadata.
    pub const fn synthetic_id(&self) -> Option<&'static str> {
        match self {
            Self::FlashNextPacked { .. } => None,
            Self::SyntheticSourceBf16 => Some(SYNTHETIC_SOURCE_BF16_RECIPE_ID),
            Self::SyntheticPacked => Some(SYNTHETIC_PACKED_RECIPE_ID),
        }
    }

    /// The synthetic recipe an artifact-metadata `id` names.
    pub fn from_synthetic_id(id: &str) -> Option<Self> {
        match id {
            SYNTHETIC_SOURCE_BF16_RECIPE_ID => Some(Self::SyntheticSourceBf16),
            SYNTHETIC_PACKED_RECIPE_ID => Some(Self::SyntheticPacked),
            _ => None,
        }
    }
}

/// What [`Qwen4TierRecipe::SyntheticPacked`] declares for one tensor, with the
/// role class and, when the role stays source-exact BF16, the reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyntheticPackDecision {
    pub dtype: DType,
    pub class: &'static str,
    /// Empty when the role is packed.
    pub reason: &'static str,
}

/// The synthetic packed recipe's per-role target: the Flash-Next formats at
/// the role's own reduction width `K` (the last dimension), declared only where
/// `K` tiles the format and a consumer reads it, BF16 otherwise.
///
/// - token embedding: Q8F16 when `K % 32 == 0`;
/// - language head: MQ6G256V2 when `K % 256 == 0`, else Q8F16 when `K % 32 == 0`;
/// - trunk attention/GDN projections (rank 2): Q8F16 when `K % 128 == 0`, the
///   Flash-Next trunk rule, without the `HIPFIRE_QWEN4_TRUNK_TIER` knob;
/// - dense FFN gate/up/down: Q8F16 when `K % 32 == 0`;
/// - routed experts: gate/up MQ4G256V2 and down MQ4G128V2, **both or neither**
///   (the sealed routed route reads the qt44/qt53 pair): packed only when the
///   hidden width is a multiple of 256 and the expert width of 128;
/// - everything else (hyper-connection mixers, router, shared expert, PLE
///   projections, MTP connectors, norms, convolutions) stays BF16 exactly as in
///   the Flash-Next recipe.
pub fn synthetic_packed_decision(name: &str, shape: &[usize]) -> SyntheticPackDecision {
    let k = shape.last().copied().unwrap_or(0);
    let packed = |dtype: DType, class: &'static str| SyntheticPackDecision {
        dtype,
        class,
        reason: "",
    };
    let exact = |class: &'static str, reason: &'static str| SyntheticPackDecision {
        dtype: DType::BF16,
        class,
        reason,
    };
    const Q8_BLOCK: &str = "K is not a multiple of the 32-weight Q8F16 block";
    if shape.len() == 2 && name == QWEN4_LM_HEAD {
        return if k % 256 == 0 {
            packed(DType::MQ6G256V2, "language head")
        } else if k % 32 == 0 {
            packed(DType::Q8_0, "language head")
        } else {
            exact("language head", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && qwen4_q8_dtype(name) {
        return if k % 32 == 0 {
            packed(DType::Q8_0, "token embedding")
        } else {
            exact("token embedding", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && name.contains(".ple_embedding.ngram_embedding.shard_") {
        return if synthetic_packed_ple_row_dtype(k) == DType::Q8_0 {
            packed(DType::Q8_0, "PLE rows")
        } else {
            exact("PLE rows", Q8_BLOCK)
        };
    }
    let gate_up = name.ends_with(".experts.gate_up_proj");
    let down = name.ends_with(".experts.down_proj");
    if shape.len() == 3 && (gate_up || down) {
        let class = if gate_up { "routed gate/up" } else { "routed down" };
        // gate_up is [experts, 2 * intermediate, hidden]; down is
        // [experts, hidden, intermediate].
        let (hidden, intermediate) = if gate_up {
            if shape[1] % 2 != 0 {
                return exact(class, "stacked gate/up rows are not an even pair");
            }
            (shape[2], shape[1] / 2)
        } else {
            (shape[1], shape[2])
        };
        return if hidden % 256 != 0 {
            exact(
                class,
                "hidden width is not a multiple of the 256-weight MQ4G256V2 group \
                 (routed gate/up and down pack together)",
            )
        } else if intermediate % 128 != 0 {
            exact(
                class,
                "expert intermediate width is not a multiple of the 128-weight MQ4G128V2 group \
                 (routed gate/up and down pack together)",
            )
        } else if gate_up {
            packed(ROUTED_GATE_UP_DTYPE, class)
        } else {
            packed(ROUTED_DOWN_DTYPE, class)
        };
    }
    if shape.len() == 2
        && !name.contains(".shared_expert.")
        && name.contains(".mlp.")
        && (name.ends_with(".mlp.gate_proj.weight")
            || name.ends_with(".mlp.up_proj.weight")
            || name.ends_with(".mlp.down_proj.weight"))
    {
        return if k % 32 == 0 {
            packed(DType::Q8_0, "dense FFN")
        } else {
            exact("dense FFN", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && (name.contains(".linear_attn.") || name.contains(".self_attn.")) {
        return if k % 128 == 0 {
            packed(DType::Q8_0, "trunk projection")
        } else {
            exact("trunk projection", "K is not a multiple of 128 (the Flash-Next trunk rule)")
        };
    }
    const KEPT: &str = "the Flash-Next recipe keeps this role source-exact";
    let class = if name.contains("hyper_connection") {
        "hyper-connection mixer"
    } else if name.contains(".shared_expert") {
        "shared expert"
    } else if name.ends_with(".mlp.gate.weight") {
        "router"
    } else if name.contains(".ple.") {
        "PLE projection"
    } else if name.starts_with("mtp.fc_") {
        "MTP connector"
    } else {
        "source-exact parameter"
    };
    exact(class, KEPT)
}

/// Row tier of one synthetic-packed PLE table: Q8F16 rows when the row width
/// is a multiple of the 32-weight block (the reader's own addressing rule),
/// BF16 rows otherwise.
pub fn synthetic_packed_ple_row_dtype(row_width: usize) -> DType {
    if row_width % 32 == 0 {
        DType::Q8_0
    } else {
        DType::BF16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum TensorRole {
    TokenEmbedding,
    LanguageHead,
    HyperConnection,
    HyperConnectionNorm,
    HyperConnectionBlockInject,
    HyperConnectionDown,
    HyperConnectionUp,
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    GdnDtBias,
    GdnALog,
    GdnNorm,
    GdnConv,
    GdnOutput,
    QsaQ,
    QsaK,
    QsaV,
    QsaOutput,
    QsaQNorm,
    QsaKNorm,
    QsaIndexerQk,
    QsaIndexerQNorm,
    QsaIndexerKNorm,
    Router,
    SharedExpertGateScalar,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    RoutedGateUp,
    RoutedDown,
    DenseFfnGate,
    DenseFfnUp,
    DenseFfnDown,
    PleShard,
    PleKey,
    PleValue,
    PleNorm,
    PleConv,
    PleMetadata,
    MtpEmbeddingProjection,
    MtpHiddenProjection,
    MtpNorm,
}

/// Typed logical reference.  The runtime manifest remains authoritative for
/// placement, aliases, source dtype, and residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorRef {
    pub fn new(
        name: impl Into<String>,
        role: TensorRole,
        layer: Option<usize>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
            return Err(WeightError::InvalidShape { name, shape });
        }
        let dtype = qwen4_target_dtype(&name, &shape, dtype);
        Ok(Self {
            name,
            role,
            layer,
            shape,
            dtype,
        })
    }
}

/// I64 metadata is retained as an exact source declaration rather than being
/// coerced into a floating-point `WeightEntry`.  The HFQM writer serializes
/// these arrays under its versioned `qwen4_ple` object; they never become GPU
/// resident weight handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
}

impl MetadataTensor {
    fn i64(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            source_dtype: "I64".to_string(),
        }
    }
}

/// The tier recipe a manifest declares its targets and source constraints
/// under.  It is part of the manifest identity: two manifests with the same
/// names and shapes but different recipes are different artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen4TierRecipe {
    /// The pinned Flash-Next packed recipe: Q8/MQ6 trunk, MQ4 routed experts,
    /// Q8 embedding, MQ6 head, Q8F16 PLE rows.  `trunk_tier` and `mtp_source`
    /// are the two producer/loader-shared knobs, resolved once.
    FlashNextPacked { trunk_tier: DType, mtp_source: bool },
    /// Developer-only synthetic profiles stay conservative: every tensor is
    /// declared source-exact BF16 (and BF16 external PLE rows) until a role
    /// format has an executable consumer, a producer and a proof.  No
    /// environment knob moves this recipe.
    SyntheticSourceBf16,
    /// Developer-only synthetic profiles with the Flash-Next per-role packed
    /// formats, declared only where a role's reduction width tiles the format
    /// ([`synthetic_packed_decision`]); every other role stays source-exact
    /// BF16.  Each role admits exactly the one representation declared for it,
    /// so an artifact's recipe identity is its bytes, not a negotiation.
    SyntheticPacked,
}

/// Profile and recipe identity carried by every manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ManifestIdentity {
    /// `Qwen4Profile::id()` of the config the manifest was built from.
    pub profile_id: String,
    pub developer_only: bool,
    /// `Qwen4EquationSource::as_str()` of the profile.
    pub equation_source: &'static str,
    /// `"dense"` or `"moe"`.
    pub ffn_kind: &'static str,
    pub recipe: Qwen4TierRecipe,
}

/// Pure logical Qwen4 manifest.  No GPU, source, or allocator is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4Manifest {
    pub identity: Qwen4ManifestIdentity,
    pub weights: Vec<WeightEntry>,
    pub state: Vec<StateEntry>,
    pub metadata: Vec<MetadataTensor>,
}

fn positive(name: &str, value: usize) -> Result<usize, WeightError> {
    if value == 0 {
        return Err(WeightError::InvalidShape {
            name: name.to_string(),
            shape: vec![value],
        });
    }
    Ok(value)
}

fn checked_product(name: &str, a: usize, b: usize) -> Result<usize, WeightError> {
    a.checked_mul(b)
        .ok_or_else(|| WeightError::ShapeOverflow(name.to_string()))
}

impl Qwen4Manifest {
    /// The declaration both the producer and the loader read.
    ///
    /// Flash-Next takes its packed tier from `HIPFIRE_QWEN4_TRUNK_TIER`; the
    /// source-tier experiment affects only rank-2 MTP records.  A synthetic
    /// developer profile ignores both knobs and declares
    /// [`Qwen4TierRecipe::SyntheticSourceBf16`].
    pub fn build(config: &Qwen4Config) -> Result<Self, WeightError> {
        let tier = if config.profile.is_flash_next() {
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: qwen4_trunk_tier(
                    hipfire_config::developer_var("HIPFIRE_QWEN4_TRUNK_TIER")
                        .ok()
                        .as_deref(),
                ),
                mtp_source: mtp_source_tier_active(),
            }
        } else {
            Qwen4TierRecipe::SyntheticSourceBf16
        };
        Self::build_with_recipe(config, tier)
    }

    /// [`Self::build`] against an explicit trunk tier, so the selector's effect
    /// on the declaration is testable without a process-global environment.
    #[cfg(test)]
    fn build_with_trunk_tier(config: &Qwen4Config, trunk_tier: DType) -> Result<Self, WeightError> {
        Self::build_with_recipe(
            config,
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier,
                mtp_source: mtp_source_tier_active(),
            },
        )
    }

    /// The one inventory generator.  Names, logical shapes, optional PLE/MTP,
    /// the FFN family and the recipe identity all come from the typed profile
    /// in `config`; nothing here reintroduces a hard-coded Flash-Next count.
    pub fn build_with_recipe(
        config: &Qwen4Config,
        recipe: Qwen4TierRecipe,
    ) -> Result<Self, WeightError> {
        config.validate().map_err(WeightError::Config)?;
        let (packed, trunk_tier, mtp_source) = match &recipe {
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier,
                mtp_source,
            } => (true, *trunk_tier, *mtp_source),
            Qwen4TierRecipe::SyntheticSourceBf16 | Qwen4TierRecipe::SyntheticPacked => {
                (false, DType::BF16, false)
            }
        };
        let synthetic_packed = matches!(recipe, Qwen4TierRecipe::SyntheticPacked);
        if packed != config.profile.is_flash_next() {
            return Err(WeightError::UnsupportedProfile {
                profile: config.profile.id().to_string(),
                reason: format!("profile cannot be declared under the {recipe:?} recipe"),
            });
        }
        let bf16 = DTypeConstraint::source_exact(DType::BF16);
        // Converted HFQM records may already carry either MQ4G256V2 (qt=44)
        // or MQ4G128V2 (qt=53).  The source checkpoint itself remains BF16.
        let quant_matrix = qwen4_quant_source();
        let quant_q8 = qwen4_q8_source();
        let quant_trunk = qwen4_trunk_matrix_source();
        // The head keeps the trunk's six/eight-bit rungs (artifacts written
        // with its earlier Q8F16 tier keep loading) but not the IU4 trunk's
        // four-bit one: four bits cost +0.014 KLD there.
        let quant_head =
            DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0, DType::MQ6G256V2]);
        let model = |name: &str,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            if !packed {
                let dtype = if synthetic_packed {
                    synthetic_packed_decision(name, &shape).dtype
                } else {
                    DType::BF16
                };
                return WeightEntry::model_with_dtype_constraint(
                    name,
                    shape,
                    dtype,
                    DTypeConstraint::source_exact(dtype),
                    policy,
                );
            }
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if name == QWEN4_LM_HEAD {
                &quant_head
            } else if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::model_with_dtype_constraint(name, shape, dtype, source.clone(), policy)
        };
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            if !packed {
                let dtype = if synthetic_packed {
                    synthetic_packed_decision(name, &shape).dtype
                } else {
                    DType::BF16
                };
                return WeightEntry::layer_with_dtype_constraint(
                    name,
                    layer_idx,
                    shape,
                    dtype,
                    DTypeConstraint::source_exact(dtype),
                    policy,
                );
            }
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let mtp_model = |name: &str,
                         shape: Vec<usize>,
                         requested_dtype: DType,
                         policy: ShardPolicy,
                         source: &DTypeConstraint| {
            mtp_entry(
                name,
                shape,
                requested_dtype,
                policy,
                source,
                packed,
                mtp_source,
                synthetic_packed,
            )
        };
        let mtp_layer = |name: &str,
                         _layer: usize,
                         shape: Vec<usize>,
                         requested_dtype: DType,
                         policy: ShardPolicy,
                         source: &DTypeConstraint| {
            // MTP names live in a separate model namespace, not the trunk's
            // layer placement scope.  Keep their manifest identity layerless.
            mtp_entry(
                name,
                shape,
                requested_dtype,
                policy,
                source,
                packed,
                mtp_source,
                synthetic_packed,
            )
        };
        let hidden = positive("hidden_size", config.hidden_size)?;
        let hc_wide = checked_product("hyper-connection width", config.hc_count, hidden)?;
        let hc_rank = config.hc_lowrank;
        let q_dim = checked_product("QSA q width", config.num_attention_heads, config.head_dim)?;
        let kv_dim = checked_product("QSA kv width", config.num_key_value_heads, config.head_dim)?;
        let indexer_heads = config
            .indexer_n_heads
            .checked_add(config.indexer_kv_heads)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA indexer heads".to_string()))?;
        let indexer_qk_dim =
            checked_product("QSA indexer width", indexer_heads, config.indexer_head_dim)?;
        let gdn_qk = checked_product(
            "GDN key width",
            config.linear_num_key_heads,
            config.linear_key_head_dim,
        )?;
        let gdn_v = checked_product(
            "GDN value width",
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )?;
        let gdn_qkv = 2usize
            .checked_mul(gdn_qk)
            .and_then(|v| v.checked_add(gdn_v))
            .ok_or_else(|| WeightError::ShapeOverflow("GDN qkv width".to_string()))?;
        // Absent PLE declares no tensor, metadata record or state at all.
        let ple = config.ple.as_ref();
        if let Some(ple) = ple {
            if ple.shard_count == 0 || ple.shard_count > MAX_PLE_SHARD_TENSORS {
                return Err(WeightError::InvalidShape {
                    name: "PLE shard count".to_string(),
                    shape: vec![ple.shard_count],
                });
            }
        }
        // PLE channels are the hyper-connection wide width (hidden x branches);
        // the table embedding width (`embed_dim`) is only the projections' input
        // columns.  Flash-Next has embed_dim == hidden, so both coincide there.
        let ple_channels = ple.map(|_| hc_wide);
        if let Some(mtp) = &config.mtp {
            if mtp.num_hidden_layers != 1 {
                return Err(WeightError::Config(format!(
                    "native MTP inventory declares exactly one layer, got {}",
                    mtp.num_hidden_layers
                )));
            }
        }

        let mut weights = Vec::with_capacity(
            64usize
                .saturating_add(config.num_hidden_layers.saturating_mul(32))
                .saturating_add(ple.map_or(0, |ple| ple.shard_count)),
        );
        weights.push(
            model(
                "model.language_model.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Embed),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        weights.push(
            model(
                "lm_head.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        // MTP aliases exactly the input embedding.  The target has no MTP
        // lm_head tensor; the distinct trunk lm_head remains the output head.
        // A profile without an MTP head declares no alias.
        if config.mtp.is_some() {
            weights.push(
                model(
                    "mtp.embed_tokens.weight",
                    vec![config.vocab_size, hidden],
                    DType::BF16,
                    ShardPolicy::Tied {
                        source: "model.language_model.embed_tokens.weight".to_string(),
                    },
                    &bf16,
                )
                .with_placement(PlacementHint::Pin(PinTarget::Embed)),
            );
        }
        // Final HC mixer is the only trunk-level post-layer transform.  Its
        // hc_norm is the final normalization; no invented RMSNorm is added.
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                vec![hc_rank, hc_wide],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                vec![hc_wide, hc_rank],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );

        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "attn_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "mlp_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            match kind {
                LayerType::LinearAttention => {
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.A_log"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        layer_idx,
                        vec![gdn_qkv, 1, config.linear_conv_kernel_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        layer_idx,
                        vec![gdn_qkv, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        layer_idx,
                        vec![gdn_v, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        layer_idx,
                        vec![config.linear_value_head_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        layer_idx,
                        vec![hidden, gdn_v],
                        DType::BF16,
                        ShardPolicy::RowShard { axis: 1 },
                        &bf16,
                    ));
                }
                LayerType::FullAttention => {
                    push_qsa_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        q_dim,
                        kv_dim,
                        indexer_qk_dim,
                        config,
                        &bf16,
                    );
                }
            }
            push_ffn_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                hidden,
                &config.ffn,
                &bf16,
                &quant_matrix,
            )?;
            if let (Some(ple), Some(channels)) = (ple, ple_channels) {
                if ple.layer == layer_idx {
                    push_ple_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        channels,
                        ple,
                        packed,
                        synthetic_packed,
                        &bf16,
                    );
                }
            }
        }

        // Native MTP has one full-attention/FFN/HC layer and no PLE.  Keep the
        // source's exact names and shapes; fc_hidden is [H,H] in this
        // checkpoint even though its consumer widens the activation.  A profile
        // without an MTP head declares none of it.
        if config.mtp.is_some() {
            weights.push(mtp_model(
                "mtp.fc_embedding.weight",
                vec![hidden, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.fc_hidden.weight",
                vec![hidden, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.pre_fc_norm_embedding.weight",
                vec![hidden],
                DType::BF16,
                ShardPolicy::Replicate,
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.pre_fc_norm_hidden.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Replicate,
                &bf16,
            ));
            push_mtp_hyper_connection_entries(&mut weights, &mtp_model, hc_wide, hc_rank, &bf16);
            push_hyper_connection_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                "attn_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                "mlp_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_qsa_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
                &bf16,
            );
            push_ffn_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                hidden,
                &config.ffn,
                &bf16,
                &quant_matrix,
            )?;
        }

        // These declarations preserve the exact I64 payload identities for
        // the qwen4_ple container.  They are deliberately not WeightEntry
        // values because no GPU tensor or float dtype may represent them.
        let metadata = match ple {
            Some(ple) => {
                let base = format!(
                    "model.language_model.layers.{}.ple.ple_embedding",
                    ple.layer
                );
                vec![
                    MetadataTensor::i64(
                        format!("{base}.layer_multipliers"),
                        vec![ple.multiplier_count],
                    ),
                    MetadataTensor::i64(
                        format!("{base}.ngram_heads_offsets"),
                        vec![ple.head_count],
                    ),
                    MetadataTensor::i64(
                        format!("{base}.ngram_heads_vocab_sizes"),
                        vec![ple.head_count],
                    ),
                ]
            }
            None => Vec::new(),
        };

        let mut state = Vec::new();
        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            state.push(match kind {
                LayerType::LinearAttention => StateEntry::new(StateKind::Recurrent, layer_idx),
                // Full K/V plus raw/pooled index keys; their storage format is
                // resolved at load (`resolve_qsa_format`, memory.kv_cache).
                LayerType::FullAttention => StateEntry::new(
                    StateKind::Kv {
                        quant: "qsa-full-raw-pooled".into(),
                    },
                    layer_idx,
                ),
            });
            if kind == LayerType::LinearAttention {
                state.push(StateEntry::new(StateKind::Conv, layer_idx));
            }
        }
        // PLE is injected at its declared layer and retains its depthwise
        // convolution history.  StateKind has no family-local name; the
        // architecture state owner keeps this entry distinct by type.  A
        // profile without PLE declares no PLE state.
        if let Some(ple) = ple {
            state.push(StateEntry::new(StateKind::Conv, ple.layer));
        }
        let identity = Qwen4ManifestIdentity {
            profile_id: config.profile.id().to_string(),
            developer_only: config.profile.developer_only(),
            equation_source: config.profile.equation_source().as_str(),
            ffn_kind: match config.ffn {
                Qwen4FfnConfig::Dense { .. } => "dense",
                Qwen4FfnConfig::Moe { .. } => "moe",
            },
            recipe,
        };
        Ok(Self {
            identity,
            weights,
            state,
            metadata,
        })
    }

    pub fn entry(&self, name: &str, layer: Option<usize>) -> Option<&WeightEntry> {
        self.weights
            .iter()
            .find(|entry| entry.name == name && entry.layer == layer)
    }

    pub fn external_entries(&self) -> impl Iterator<Item = &WeightEntry> {
        self.weights
            .iter()
            .filter(|entry| entry.residency.is_external())
    }

    pub fn resident_keys(&self) -> impl Iterator<Item = (&str, Option<usize>)> {
        self.weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
            .map(|entry| (entry.name.as_str(), entry.layer))
    }

    /// Number of source tensor records represented by this text manifest.
    /// The only non-source record is the tied MTP embedding alias (absent
    /// without an MTP head); the exact I64 PLE metadata arrays (absent
    /// without PLE) are counted from `metadata`.
    pub fn source_tensor_count(&self) -> usize {
        let aliases = self
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        self.weights
            .len()
            .saturating_sub(aliases)
            .saturating_add(self.metadata.len())
    }

    /// This manifest without its external PLE n-gram shard tensors.
    ///
    /// A sidecar rows file (`hipfire_runtime::ple_sidecar`) supplies the PLE
    /// rows, so the source file's own shard tensors are neither declared nor
    /// fulfilled: the base may lack them (a split base) or carry them,
    /// shadowed and ignored (the monolithic artifact next to a derived q4
    /// sidecar).  The I64 PLE metadata records stay.  The shards are the
    /// only external-row entries of a Qwen4 manifest; their count comes from
    /// the declaration, not a constant.
    pub fn without_ple_shards(mut self) -> Self {
        self.weights.retain(|entry| !entry.residency.is_external());
        self
    }
}

fn mtp_entry(
    name: &str,
    shape: Vec<usize>,
    requested_dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
    packed: bool,
    mtp_source: bool,
    synthetic_packed: bool,
) -> WeightEntry {
    // A synthetic profile never takes the Flash-Next tier knobs: its MTP
    // tensors are source-exact BF16, or the synthetic packed recipe's per-role
    // decision when that recipe is declared.
    if !packed {
        let dtype = if synthetic_packed {
            synthetic_packed_decision(name, &shape).dtype
        } else {
            DType::BF16
        };
        return WeightEntry::model_with_dtype_constraint(
            name,
            shape,
            dtype,
            DTypeConstraint::source_exact(dtype),
            policy,
        );
    }
    // The existing source-tier experiment changes only rank-2 MTP entries;
    // the routed experts are rank-3, and their writer has no BF16 arm.
    if mtp_source && shape.len() == 2 {
        return WeightEntry::model_with_dtype_constraint(
            name,
            shape,
            DType::BF16,
            DTypeConstraint::source_exact(DType::BF16),
            policy,
        );
    }
    let dtype = qwen4_target_dtype(name, &shape, requested_dtype);
    // Mirror the trunk's `model` closure: the wide attention/GDN projections
    // ship at whichever tier the artifact declares (MQ6G256V2 or Q8), and a Q8
    // target otherwise takes the Q8 source set (BF16 checkpoint or an
    // already-converted Q8 payload).  `qwen4_quantized_dtype` is true for Q8_0
    // as well, so testing it first would hand every eight-bit MTP projection
    // the expert constraint (BF16 | MQ4G256V2 | MQ4G128V2) — families that are
    // a different geometry and would be misread there.
    let source = if qwen4_trunk_matrix(name, &shape) {
        qwen4_trunk_matrix_source()
    } else if dtype == DType::Q8_0 {
        qwen4_q8_source()
    } else if qwen4_quantized_dtype(dtype) {
        qwen4_quant_source()
    } else {
        source.clone()
    };
    WeightEntry::model_with_dtype_constraint(name, shape, dtype, source, policy)
}

fn push_hyper_connection_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    name: &str,
    hc_count: usize,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let prefix = format!("{prefix}.{name}");
    weights.push(layer(
        &format!("{prefix}.block_inject_weight.weight"),
        layer_idx,
        vec![hc_count, hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.hc_norm.weight"),
        layer_idx,
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_down.weight"),
        layer_idx,
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_up.weight"),
        layer_idx,
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    // Final HC mixers (declared separately) have no inject operand.
}

fn push_mtp_hyper_connection_entries<M>(
    weights: &mut Vec<WeightEntry>,
    mtp_model: &M,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    M: Fn(&str, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.hc_norm.weight",
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
}

fn push_qsa_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let push =
        |weights: &mut Vec<WeightEntry>, suffix: &str, shape: Vec<usize>, policy: ShardPolicy| {
            weights.push(layer(
                &format!("{prefix}.self_attn.{suffix}"),
                layer_idx,
                shape,
                DType::BF16,
                policy,
                bf16,
            ));
        };
    push(
        weights,
        "indexer.index_qk_proj.weight",
        vec![indexer_qk_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "indexer.k_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "indexer.q_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "o_proj.weight",
        vec![hidden, q_dim],
        ShardPolicy::RowShard { axis: 1 },
    );
    push(
        weights,
        "q_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "q_proj.weight",
        vec![2 * q_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "v_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
}

/// The FFN inventory of one layer: the profile's single owner of which tensors
/// exist.  A dense FFN is three resident nonexpert matrices (gate/up/down) and
/// nothing else — no router, no expert stacks, no shared selector.  A MoE FFN
/// is the router and the two routed expert stacks, plus the gated shared
/// expert only when the profile declares one.
fn push_ffn_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ffn: &Qwen4FfnConfig,
    bf16: &DTypeConstraint,
    quant_matrix: &DTypeConstraint,
) -> Result<(), WeightError>
where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let mlp = format!("{prefix}.mlp");
    match ffn {
        Qwen4FfnConfig::Dense {
            intermediate_size, ..
        } => {
            let intermediate = positive("dense intermediate_size", *intermediate_size)?;
            checked_product("dense FFN gate/up elements", intermediate, hidden)?;
            weights.push(layer(
                &format!("{mlp}.gate_proj.weight"),
                layer_idx,
                vec![intermediate, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                bf16,
            ));
            weights.push(layer(
                &format!("{mlp}.up_proj.weight"),
                layer_idx,
                vec![intermediate, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                bf16,
            ));
            weights.push(layer(
                &format!("{mlp}.down_proj.weight"),
                layer_idx,
                vec![hidden, intermediate],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                bf16,
            ));
        }
        Qwen4FfnConfig::Moe {
            num_experts,
            intermediate_size,
            shared_intermediate_size,
            ..
        } => {
            let experts = positive("num_experts", *num_experts)?;
            let intermediate = positive("moe intermediate_size", *intermediate_size)?;
            let gate_up_rows = checked_product("routed gate_up rows", 2, intermediate)?;
            checked_product("routed gate_up elements", gate_up_rows, hidden)
                .and_then(|per_expert| checked_product("routed expert elements", per_expert, experts))?;
            weights.push(layer(
                &format!("{mlp}.experts.down_proj"),
                layer_idx,
                vec![experts, hidden, intermediate],
                ROUTED_DOWN_DTYPE,
                ShardPolicy::Replicate,
                quant_matrix,
            ));
            weights.push(layer(
                &format!("{mlp}.experts.gate_up_proj"),
                layer_idx,
                vec![experts, gate_up_rows, hidden],
                ROUTED_GATE_UP_DTYPE,
                ShardPolicy::Replicate,
                quant_matrix,
            ));
            weights.push(layer(
                &format!("{mlp}.gate.weight"),
                layer_idx,
                vec![experts, hidden],
                DType::BF16,
                ShardPolicy::Replicate,
                bf16,
            ));
            if let Some(shared) = shared_intermediate_size {
                let shared = positive("shared expert intermediate_size", *shared)?;
                weights.push(layer(
                    &format!("{mlp}.shared_expert.down_proj.weight"),
                    layer_idx,
                    vec![hidden, shared],
                    DType::BF16,
                    ShardPolicy::RowShard { axis: 1 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert.gate_proj.weight"),
                    layer_idx,
                    vec![shared, hidden],
                    DType::BF16,
                    ShardPolicy::ColumnShard { axis: 0 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert.up_proj.weight"),
                    layer_idx,
                    vec![shared, hidden],
                    DType::BF16,
                    ShardPolicy::ColumnShard { axis: 0 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert_gate.weight"),
                    layer_idx,
                    vec![1, hidden],
                    DType::BF16,
                    ShardPolicy::Replicate,
                    bf16,
                ));
            }
        }
    }
    Ok(())
}

/// One PLE block.  Every dimension (channel width, conv kernel, shard count,
/// shard rows, row width, valid rows) comes from the typed [`Qwen4PleConfig`];
/// the Flash-Next values are that config's canonical instance, not constants
/// here.  `packed` is the Flash-Next recipe: it ships Q8F16 rows and still
/// admits BF16 ones.  `synthetic_packed` ships exactly the one tier the row
/// width allows ([`synthetic_packed_ple_row_dtype`]); any other synthetic recipe
/// ships and admits BF16 rows only.
fn push_ple_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ple_channels: usize,
    config: &Qwen4PleConfig,
    packed: bool,
    synthetic_packed: bool,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let ple = format!("{prefix}.ple");
    weights.push(layer(
        &format!("{ple}.conv1d.weight"),
        layer_idx,
        vec![ple_channels, 1, config.conv_kernel_size],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.key_proj.weight"),
        layer_idx,
        vec![ple_channels, config.embed_dim],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_conv.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_key.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_query.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.value_proj.weight"),
        layer_idx,
        vec![hidden, config.embed_dim],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    // The PLE row tier is declared by the artifact, not by this manifest: a
    // sealed artifact written before the Q8F16 tier carries BF16 rows and must
    // stay loadable, so both tiers are admitted and the reader derives its page
    // and read geometry from whichever one the shard declares.
    let (ple_rows, row_dtype, encoding) = if packed {
        (
            DTypeConstraint::source_from_sources(vec![DType::Q8_0, DType::BF16]),
            DType::Q8_0,
            RowEncoding::Q8F16,
        )
    } else if synthetic_packed && synthetic_packed_ple_row_dtype(config.row_width) == DType::Q8_0 {
        (
            DTypeConstraint::source_exact(DType::Q8_0),
            DType::Q8_0,
            RowEncoding::Q8F16,
        )
    } else {
        (bf16.clone(), DType::BF16, RowEncoding::Bf16)
    };
    for shard in 0..config.shard_count {
        let name = format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight");
        weights.push(
            layer(
                &name,
                layer_idx,
                vec![config.shard_rows, config.row_width],
                row_dtype,
                ShardPolicy::Replicate,
                &ple_rows,
            )
            // Q8F16 is the current writer's tier: five `[f16 scale][32 x i8]`
            // blocks per 160-value row, 170 bytes instead of the 320 a BF16 row
            // costs. The declared stride here is the default; a sealed BF16
            // shard is accepted by the same constraint and validated against its
            // own dtype-derived stride.
            .external_rows(
                encoding.encoded_row_bytes(config.row_width),
                config.shard_valid_rows(shard),
            ),
        );
    }
}

/// GPU placement requested by the typed Qwen4 assembler.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qwen4Placement {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionReadWeights {
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionWeights {
    pub block_inject: TensorRef,
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct GdnWeights {
    pub a_log: TensorRef,
    pub dt_bias: TensorRef,
    pub in_proj_a: TensorRef,
    pub in_proj_b: TensorRef,
    pub qkv: TensorRef,
    pub z: TensorRef,
    pub norm: TensorRef,
    pub conv: TensorRef,
    pub output: TensorRef,
}

#[derive(Clone, Debug)]
pub struct QsaWeights {
    pub indexer_qk: TensorRef,
    pub indexer_k_norm: TensorRef,
    pub indexer_q_norm: TensorRef,
    pub k_norm: TensorRef,
    pub k: TensorRef,
    pub output: TensorRef,
    pub q_norm: TensorRef,
    pub q: TensorRef,
    pub v: TensorRef,
}

#[derive(Clone, Debug)]
pub struct PleWeights {
    pub conv: TensorRef,
    pub key: TensorRef,
    pub norm_conv: TensorRef,
    pub norm_key: TensorRef,
    pub norm_query: TensorRef,
    pub value: TensorRef,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub experts_down: TensorRef,
    pub experts_gate_up: TensorRef,
    /// The pinned checkpoint carries one stacked gate/up source.  Gate and up
    /// remain distinct logical projection slices in the sealed runtime plan;
    /// do not rewrite the artifact merely to spell them as separate sources.
    pub expert_source_layout: ExpertSourceLayout,
    pub gate: TensorRef,
    pub shared_down: TensorRef,
    pub shared_gate: TensorRef,
    pub shared_up: TensorRef,
    pub shared_gate_scalar: TensorRef,
}

/// Dense gated-MLP projections: `down(silu(gate(x)) * up(x))`.
#[derive(Clone, Debug)]
pub struct DenseFfnWeights {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

/// The FFN block of one decoder layer (or of the MTP layer).
#[derive(Clone, Debug)]
pub enum Qwen4FfnWeights {
    Dense(DenseFfnWeights),
    Moe(MoeWeights),
}

impl Qwen4FfnWeights {
    pub fn moe(&self) -> Option<&MoeWeights> {
        match self {
            Self::Moe(moe) => Some(moe),
            Self::Dense(_) => None,
        }
    }

    pub fn dense(&self) -> Option<&DenseFfnWeights> {
        match self {
            Self::Dense(dense) => Some(dense),
            Self::Moe(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen4LayerWeights {
    pub layer: usize,
    pub kind: LayerType,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: Option<QsaWeights>,
    pub gdn: Option<GdnWeights>,
    pub ple: Option<PleWeights>,
    pub ffn: Qwen4FfnWeights,
}

#[derive(Clone, Debug)]
pub struct Qwen4MtpWeights {
    pub fc_embedding: TensorRef,
    pub fc_hidden: TensorRef,
    pub pre_fc_norm_embedding: TensorRef,
    pub pre_fc_norm_hidden: TensorRef,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: QsaWeights,
    pub ffn: Qwen4FfnWeights,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Typed trunk-level roots required by the native forward path.
///
/// The references are immutable logical identities.  Their resident buffers
/// are resolved through [`Qwen4Weights::resident`] so aliases remain owned by
/// the canonical load transaction and the forward path never performs source
/// I/O or an ad-hoc upload.
#[derive(Clone, Debug)]
pub struct Qwen4RootWeights {
    pub embedding: TensorRef,
    pub lm_head: TensorRef,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Published resident ownership.  External PLE descriptors remain in the
/// attached transaction census; no fake handle is stored here.
pub struct Qwen4Weights {
    pub manifest: Qwen4Manifest,
    pub taken: Vec<TakenWeight>,
    /// `taken` slots by tensor name: `resident` runs ~1000 times per forward,
    /// and a linear scan over every taken weight cost ~3 ms per decode token.
    taken_by_name: HashMap<String, Vec<usize>>,
    pub root: Qwen4RootWeights,
    pub layer_refs: Vec<Qwen4LayerWeights>,
    /// `Some` iff `config.mtp` is declared.
    pub mtp: Option<Qwen4MtpWeights>,
}

impl Qwen4Weights {
    /// Publish the resident weights of `tx` for `config` under the manifest
    /// the artifact was admitted with: `recipe` is `Qwen4HfqmArtifact::recipe()`
    /// (the manifest identity's recipe), so the declaration is rebuilt
    /// exactly as admission built it instead of from `config` alone.  For
    /// Flash-Next the recipe is the single `FlashNextPacked` carrying the
    /// trunk-tier and MTP-source knobs resolved at admission, which is what
    /// [`Qwen4Manifest::build`] resolved from the same environment before.  A
    /// recipe the profile cannot be declared under is refused by
    /// [`Qwen4Manifest::build_with_recipe`].
    pub fn assemble(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        recipe: &Qwen4TierRecipe,
        placements: &[Qwen4Placement],
    ) -> Result<Self, WeightError> {
        let manifest = assembly_manifest(config, recipe)?;
        let expected = manifest
            .resident_keys()
            .map(|(name, layer)| (name.to_string(), layer))
            .collect::<BTreeSet<_>>();
        let provided = placements
            .iter()
            .map(|placement| (placement.name.clone(), placement.layer))
            .collect::<BTreeSet<_>>();
        if expected != provided {
            return Err(WeightError::PlacementSetMismatch {
                expected: expected.len(),
                actual: provided.len(),
            });
        }
        let mut assembly = tx.begin_assembly();
        for placement in placements {
            if assembly
                .take(&placement.name, placement.layer, placement.device)
                .is_none()
            {
                return Err(WeightError::MissingPlacement {
                    name: placement.name.clone(),
                    layer: placement.layer,
                    device: placement.device,
                });
            }
        }
        let taken = assembly.commit().finalize();
        let mut taken_by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (slot, weight) in taken.iter().enumerate() {
            taken_by_name
                .entry(weight.key.name.clone())
                .or_default()
                .push(slot);
        }
        let root = build_root_refs(config)?;
        let layer_refs = build_layer_refs(config)?;
        let mtp = config
            .mtp
            .is_some()
            .then(|| build_mtp_refs(config))
            .transpose()?;
        Ok(Self {
            manifest,
            taken,
            taken_by_name,
            root,
            layer_refs,
            mtp,
        })
    }

    /// Resolve one immutable typed reference to its resident GPU buffer.
    ///
    /// Qwen4 is currently admitted only on the Single route, so device zero is
    /// the sole logical placement.  Tied weights are followed by their
    /// canonical source name and never trigger I/O or allocation.
    pub fn resident(&self, reference: &TensorRef) -> Result<&GpuTensor, WeightError> {
        self.resident_named(&reference.name, reference.layer, 0, 0)
    }

    fn resident_named(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
        depth: usize,
    ) -> Result<&GpuTensor, WeightError> {
        if depth > self.taken.len() {
            return Err(WeightError::DescriptorMismatch(format!(
                "{name}: resident alias cycle"
            )));
        }
        let Some(taken) = self
            .taken_by_name
            .get(name)
            .into_iter()
            .flatten()
            .map(|&slot| &self.taken[slot])
            .find(|taken| taken.key.layer == layer && taken.key.device == device)
        else {
            return Err(WeightError::MissingResident {
                name: name.to_string(),
                layer,
                device,
            });
        };
        match &taken.handle {
            WeightHandle::Resident(tensor) => Ok(tensor),
            WeightHandle::Alias(source) => {
                let source = source.clone();
                self.resident_named(&source, layer, device, depth + 1)
            }
        }
    }

    pub fn external_descriptor<'a>(
        tx: &'a WeightLoadTransaction,
        placement: &Qwen4Placement,
    ) -> Option<&'a SourceRangeDescriptor> {
        tx.external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Explicit GPU teardown for handles finalized from the canonical
    /// transaction.  The attached transaction census is drained separately,
    /// after readers and these resident handles are gone.
    pub fn free_gpu(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        let mut first = None;
        for taken in self.taken {
            if let WeightHandle::Resident(tensor) = taken.handle {
                if let Err(error) = gpu.free_tensor(tensor) {
                    if first.is_none() {
                        first = Some(error);
                    }
                }
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Load-time precision experiments: `HIPFIRE_QWEN4_REQUANT="pat=fmt;..."`
    /// replaces every resident rank-2 weight whose name contains `pat` (first
    /// matching rule wins) with a requantized copy, `fmt` = `mq2` .. `mq6`
    /// (MQ G256 V2) | `q8` (from BF16 only) | `qt44:DIR` (pre-encoded QT44
    /// bytes from `DIR/<tensor name>.bin` where that file exists, e.g. an
    /// offline GPTQ solve; it must hold exactly the `m x k` matrix). Decode
    /// and prefill both
    /// bind it.
    pub fn requant_from_env(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let Ok(spec) = hipfire_config::developer_var("HIPFIRE_QWEN4_REQUANT") else {
            return Ok(());
        };
        let mut rules = Vec::new();
        for rule in spec.split(';').filter(|rule| !rule.trim().is_empty()) {
            let (pattern, format) = rule
                .split_once('=')
                .ok_or_else(|| format!("HIPFIRE_QWEN4_REQUANT rule {rule:?} is not pat=fmt"))?;
            let (target, dir) = match format.trim() {
                "mq2" => (DType::MQ2G256V2, None),
                "mq3" => (DType::MQ3G256V2, None),
                "mq4" => (DType::MQ4G256V2, None),
                "mq5" => (DType::MQ5G256V2, None),
                "mq6" => (DType::MQ6G256V2, None),
                "q8" => (DType::Q8_0, None),
                other => match other.strip_prefix("qt44:") {
                    Some(dir) => (DType::MQ4G256V2, Some(std::path::PathBuf::from(dir))),
                    None => return Err(format!("HIPFIRE_QWEN4_REQUANT format {other:?}")),
                },
            };
            rules.push((pattern.trim().to_string(), target, dir));
        }
        let (mut count, mut before, mut after) = (0usize, 0usize, 0usize);
        for taken in &mut self.taken {
            let WeightHandle::Resident(tensor) = &taken.handle else {
                continue;
            };
            let Some((_, target, dir)) = rules
                .iter()
                .find(|(pattern, _, _)| taken.key.name.contains(pattern.as_str()))
            else {
                continue;
            };
            let target = *target;
            let [m, k] = taken.projection.logical_shape[..] else {
                continue;
            };
            let copy = if let Some(dir) = dir {
                // A matching tensor without a file keeps its payload (the
                // MTP attention shares the trunk's names).
                let path = dir.join(format!("{}.bin", taken.key.name));
                if !path.exists() {
                    continue;
                }
                let bytes =
                    std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                if k % 256 != 0 || bytes.len() != m * (k / 256) * rdna_compute::MQ4V2_GROUP_BYTES {
                    return Err(format!(
                        "{}: {} bytes is not a {m}x{k} QT44 matrix",
                        path.display(),
                        bytes.len()
                    ));
                }
                let mut copy = gpu
                    .upload_raw(&bytes, &[bytes.len()])
                    .map_err(|error| error.to_string())?;
                copy.dtype = DType::MQ4G256V2;
                copy
            } else {
                if tensor.dtype == target {
                    continue;
                }
                if target == DType::Q8_0 {
                    gpu.quantize_bf16_q8_0(tensor, m, k)
                } else {
                    gpu.requant_g256(tensor, m, k, target)
                }
                .map_err(|error| format!("requant {} -> {target:?}: {error}", taken.key.name))?
            };
            count += 1;
            before += tensor.buf.size();
            after += copy.buf.size();
            let WeightHandle::Resident(old) =
                std::mem::replace(&mut taken.handle, WeightHandle::Resident(copy))
            else {
                unreachable!()
            };
            gpu.free_tensor(old).map_err(|error| error.to_string())?;
        }
        eprintln!(
            "[qwen4] HIPFIRE_QWEN4_REQUANT {spec:?}: {count} tensors, {before} -> {after} bytes"
        );
        Ok(())
    }
}

fn build_root_refs(config: &Qwen4Config) -> Result<Qwen4RootWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let embedding = TensorRef::new(
        "model.language_model.embed_tokens.weight",
        TensorRole::TokenEmbedding,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    let lm_head = TensorRef::new(
        "lm_head.weight",
        TensorRole::LanguageHead,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    Ok(Qwen4RootWeights {
        embedding,
        lm_head,
        final_hyper: build_hyper_read_refs(
            "model.language_model.hyper_connection_mixer",
            None,
            hc_wide,
            config.hc_lowrank,
        )?,
    })
}

fn build_hyper_read_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionReadWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(HyperConnectionReadWeights {
        hc_norm: tr(
            "hc_norm.weight",
            TensorRole::HyperConnectionNorm,
            vec![hc_wide],
        )?,
        input_mix_down: tr(
            "input_mix_weight_down.weight",
            TensorRole::HyperConnectionDown,
            vec![hc_rank, hc_wide],
        )?,
        input_mix_up: tr(
            "input_mix_weight_up.weight",
            TensorRole::HyperConnectionUp,
            vec![hc_wide, hc_rank],
        )?,
    })
}

fn build_hyper_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_count: usize,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionWeights, WeightError> {
    let read = build_hyper_read_refs(prefix, layer, hc_wide, hc_rank)?;
    let block_inject = TensorRef::new(
        format!("{prefix}.block_inject_weight.weight"),
        TensorRole::HyperConnectionBlockInject,
        layer,
        vec![hc_count, hc_wide],
        DType::BF16,
    )?;
    Ok(HyperConnectionWeights {
        block_inject,
        hc_norm: read.hc_norm,
        input_mix_down: read.input_mix_down,
        input_mix_up: read.input_mix_up,
    })
}

fn build_layer_refs(config: &Qwen4Config) -> Result<Vec<Qwen4LayerWeights>, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let gdn_qk = config.linear_num_key_heads * config.linear_key_head_dim;
    let gdn_v = config.linear_num_value_heads * config.linear_value_head_dim;
    let gdn_qkv = 2 * gdn_qk + gdn_v;
    let mut refs = Vec::with_capacity(config.num_hidden_layers);
    for (layer, kind) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        let attn_hyper = build_hyper_refs(
            &format!("{prefix}.attn_hyper_connection"),
            Some(layer),
            config.hc_count,
            hc_wide,
            hc_rank,
        )?;
        let mlp_hyper = build_hyper_refs(
            &format!("{prefix}.mlp_hyper_connection"),
            Some(layer),
            config.hc_count,
            hc_wide,
            hc_rank,
        )?;
        let attention = if kind == LayerType::FullAttention {
            Some(build_qsa_refs(
                &format!("{prefix}.self_attn"),
                Some(layer),
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
            )?)
        } else {
            None
        };
        let gdn = if kind == LayerType::LinearAttention {
            Some(build_gdn_refs(
                &format!("{prefix}.linear_attn"),
                Some(layer),
                hidden,
                gdn_qkv,
                gdn_v,
                config,
            )?)
        } else {
            None
        };
        let ple = match &config.ple {
            Some(ple_config) if ple_config.layer == layer => {
                let ple_channels = config
                    .hidden_size
                    .checked_mul(config.hc_count)
                    .ok_or_else(|| WeightError::ShapeOverflow("PLE channel width".to_string()))?;
                Some(build_ple_refs(
                    &format!("{prefix}.ple"),
                    Some(layer),
                    hidden,
                    ple_channels,
                    ple_config,
                )?)
            }
            _ => None,
        };
        let ffn = build_ffn_refs(&format!("{prefix}.mlp"), Some(layer), hidden, config)?;
        refs.push(Qwen4LayerWeights {
            layer,
            kind,
            attn_hyper,
            mlp_hyper,
            attention,
            gdn,
            ple,
            ffn,
        });
    }
    Ok(refs)
}

/// Bind the native MTP head; the caller binds it only when `config.mtp` is
/// declared.
fn build_mtp_refs(config: &Qwen4Config) -> Result<Qwen4MtpWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    Ok(Qwen4MtpWeights {
        fc_embedding: TensorRef::new(
            "mtp.fc_embedding.weight",
            TensorRole::MtpEmbeddingProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        fc_hidden: TensorRef::new(
            "mtp.fc_hidden.weight",
            TensorRole::MtpHiddenProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        pre_fc_norm_embedding: TensorRef::new(
            "mtp.pre_fc_norm_embedding.weight",
            TensorRole::MtpNorm,
            None,
            vec![hidden],
            DType::BF16,
        )?,
        pre_fc_norm_hidden: TensorRef::new(
            "mtp.pre_fc_norm_hidden.weight",
            TensorRole::MtpNorm,
            None,
            vec![hc_wide],
            DType::BF16,
        )?,
        attn_hyper: build_hyper_refs(
            "mtp.layers.0.attn_hyper_connection",
            None,
            config.hc_count,
            hc_wide,
            hc_rank,
        )?,
        mlp_hyper: build_hyper_refs(
            "mtp.layers.0.mlp_hyper_connection",
            None,
            config.hc_count,
            hc_wide,
            hc_rank,
        )?,
        attention: build_qsa_refs(
            "mtp.layers.0.self_attn",
            None,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
        )?,
        ffn: build_ffn_refs("mtp.layers.0.mlp", None, hidden, config)?,
        final_hyper: build_hyper_read_refs("mtp.hyper_connection_mixer", None, hc_wide, hc_rank)?,
    })
}

fn build_gdn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    qkv: usize,
    value: usize,
    config: &Qwen4Config,
) -> Result<GdnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(GdnWeights {
        a_log: tr(
            "A_log",
            TensorRole::GdnALog,
            vec![config.linear_num_value_heads],
        )?,
        dt_bias: tr(
            "dt_bias",
            TensorRole::GdnDtBias,
            vec![config.linear_num_value_heads],
        )?,
        in_proj_a: tr(
            "in_proj_a.weight",
            TensorRole::GdnA,
            vec![config.linear_num_value_heads, hidden],
        )?,
        in_proj_b: tr(
            "in_proj_b.weight",
            TensorRole::GdnB,
            vec![config.linear_num_value_heads, hidden],
        )?,
        qkv: tr("in_proj_qkv.weight", TensorRole::GdnQkv, vec![qkv, hidden])?,
        z: tr("in_proj_z.weight", TensorRole::GdnZ, vec![value, hidden])?,
        norm: tr(
            "norm.weight",
            TensorRole::GdnNorm,
            vec![config.linear_value_head_dim],
        )?,
        conv: tr(
            "conv1d.weight",
            TensorRole::GdnConv,
            vec![qkv, 1, config.linear_conv_kernel_dim],
        )?,
        output: tr(
            "out_proj.weight",
            TensorRole::GdnOutput,
            vec![hidden, value],
        )?,
    })
}

fn build_qsa_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
) -> Result<QsaWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(QsaWeights {
        indexer_qk: tr(
            "indexer.index_qk_proj.weight",
            TensorRole::QsaIndexerQk,
            vec![indexer_qk_dim, hidden],
        )?,
        indexer_k_norm: tr(
            "indexer.k_layernorm.weight",
            TensorRole::QsaIndexerKNorm,
            vec![config.indexer_head_dim],
        )?,
        indexer_q_norm: tr(
            "indexer.q_layernorm.weight",
            TensorRole::QsaIndexerQNorm,
            vec![config.indexer_head_dim],
        )?,
        k_norm: tr("k_norm.weight", TensorRole::QsaKNorm, vec![config.head_dim])?,
        k: tr("k_proj.weight", TensorRole::QsaK, vec![kv_dim, hidden])?,
        output: tr("o_proj.weight", TensorRole::QsaOutput, vec![hidden, q_dim])?,
        q_norm: tr("q_norm.weight", TensorRole::QsaQNorm, vec![config.head_dim])?,
        q: tr("q_proj.weight", TensorRole::QsaQ, vec![2 * q_dim, hidden])?,
        v: tr("v_proj.weight", TensorRole::QsaV, vec![kv_dim, hidden])?,
    })
}

fn build_ple_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    channels: usize,
    config: &Qwen4PleConfig,
) -> Result<PleWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(PleWeights {
        conv: tr(
            "conv1d.weight",
            TensorRole::PleConv,
            vec![channels, 1, config.conv_kernel_size],
        )?,
        key: tr(
            "key_proj.weight",
            TensorRole::PleKey,
            vec![channels, config.embed_dim],
        )?,
        norm_conv: tr("norm_conv.weight", TensorRole::PleNorm, vec![channels])?,
        norm_key: tr("norm_key.weight", TensorRole::PleNorm, vec![channels])?,
        norm_query: tr("norm_query.weight", TensorRole::PleNorm, vec![channels])?,
        value: tr(
            "value_proj.weight",
            TensorRole::PleValue,
            vec![hidden, config.embed_dim],
        )?,
    })
}

/// Refuse, before any allocation or placement, every profile the typed
/// executor cannot run.  Header-only developer inventories are admitted by
/// `artifact` independently of this executor admission.
///
/// * Flash-Next: the pinned shared-expert MoE with PLE and the native MTP head.
/// * Synthetic developer profiles: only under `cfg(test)` or the
///   `qwen4-dev-profiles` feature, and only a dense FFN (PLE and MTP optional
///   at this layer).  A synthetic MoE has no sealed route (see
///   [`admit_synthetic_profile`]).
pub(crate) fn admit_executable_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    let unsupported = |reason: &str| WeightError::UnsupportedProfile {
        profile: config.profile.id().to_string(),
        reason: reason.to_string(),
    };
    match &config.profile {
        Qwen4Profile::FlashNext => {
            if !matches!(
                &config.ffn,
                Qwen4FfnConfig::Moe {
                    shared_intermediate_size: Some(_),
                    ..
                }
            ) {
                return Err(unsupported(
                    "the Qwen4 executor binds only the shared-expert MoE FFN",
                ));
            }
            if config.ple.is_none() {
                return Err(unsupported("the Qwen4 executor requires PLE"));
            }
            if config.mtp.is_none() {
                return Err(unsupported(
                    "the Qwen4 executor requires the native MTP head",
                ));
            }
            Ok(())
        }
        Qwen4Profile::Synthetic(_) => admit_synthetic_profile(config),
    }
}

/// The declaration [`Qwen4Weights::assemble`] publishes against: the profile
/// must be executable, and the manifest is the one `recipe` declares for it
/// (the admitted artifact's own recipe), never a rebuild from `config` alone.
fn assembly_manifest(
    config: &Qwen4Config,
    recipe: &Qwen4TierRecipe,
) -> Result<Qwen4Manifest, WeightError> {
    admit_executable_profile(config)?;
    Qwen4Manifest::build_with_recipe(config, recipe.clone())
}

/// Production builds never execute a developer profile.
#[cfg(not(any(test, feature = "qwen4-dev-profiles")))]
fn admit_synthetic_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    Err(WeightError::UnsupportedProfile {
        profile: config.profile.id().to_string(),
        reason: "developer profiles execute only in builds with the qwen4-dev-profiles feature"
            .to_string(),
    })
}

#[cfg(any(test, feature = "qwen4-dev-profiles"))]
fn admit_synthetic_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    match &config.ffn {
        Qwen4FfnConfig::Dense { .. } => Ok(()),
        Qwen4FfnConfig::Moe {
            num_experts,
            top_k,
            shared_intermediate_size: None,
            ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: format!(
                "MoE without a shared expert has no executor (experts={num_experts}, top_k={top_k})"
            ),
        }),
        Qwen4FfnConfig::Moe {
            num_experts, top_k, ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: format!(
                "no sealed MoE route exists for this geometry (experts={num_experts}, top_k={top_k}): \
                 the canonical qt44/qt53 route pins 512 experts, top-10, hidden 2560 and intermediate 640, \
                 and there is no BF16 routed-expert route or top-k other than 8/10"
            ),
        }),
    }
}

/// Bind one FFN block's typed references: dense gate/up/down, or the pinned
/// shared-expert MoE.
fn build_ffn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    config: &Qwen4Config,
) -> Result<Qwen4FfnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>, dtype: DType| {
        TensorRef::new(format!("{prefix}.{suffix}"), role, layer, shape, dtype)
    };
    match &config.ffn {
        Qwen4FfnConfig::Dense { intermediate_size } => {
            let intermediate = *intermediate_size;
            Ok(Qwen4FfnWeights::Dense(DenseFfnWeights {
                gate: tr(
                    "gate_proj.weight",
                    TensorRole::DenseFfnGate,
                    vec![intermediate, hidden],
                    DType::BF16,
                )?,
                up: tr(
                    "up_proj.weight",
                    TensorRole::DenseFfnUp,
                    vec![intermediate, hidden],
                    DType::BF16,
                )?,
                down: tr(
                    "down_proj.weight",
                    TensorRole::DenseFfnDown,
                    vec![hidden, intermediate],
                    DType::BF16,
                )?,
            }))
        }
        Qwen4FfnConfig::Moe {
            num_experts,
            intermediate_size,
            shared_intermediate_size: Some(shared),
            ..
        } => {
            let (experts, intermediate, shared) = (*num_experts, *intermediate_size, *shared);
            Ok(Qwen4FfnWeights::Moe(MoeWeights {
                experts_down: tr(
                    "experts.down_proj",
                    TensorRole::RoutedDown,
                    vec![experts, hidden, intermediate],
                    ROUTED_DOWN_DTYPE,
                )?,
                experts_gate_up: tr(
                    "experts.gate_up_proj",
                    TensorRole::RoutedGateUp,
                    vec![experts, 2 * intermediate, hidden],
                    ROUTED_GATE_UP_DTYPE,
                )?,
                expert_source_layout: ExpertSourceLayout::PackedFused {
                    gate_up: format!("{prefix}.experts.gate_up_proj"),
                    down: format!("{prefix}.experts.down_proj"),
                    sidecars: Vec::new(),
                },
                gate: tr(
                    "gate.weight",
                    TensorRole::Router,
                    vec![experts, hidden],
                    DType::BF16,
                )?,
                shared_down: tr(
                    "shared_expert.down_proj.weight",
                    TensorRole::SharedExpertDown,
                    vec![hidden, shared],
                    DType::BF16,
                )?,
                shared_gate: tr(
                    "shared_expert.gate_proj.weight",
                    TensorRole::SharedExpertGate,
                    vec![shared, hidden],
                    DType::BF16,
                )?,
                shared_up: tr(
                    "shared_expert.up_proj.weight",
                    TensorRole::SharedExpertUp,
                    vec![shared, hidden],
                    DType::BF16,
                )?,
                shared_gate_scalar: tr(
                    "shared_expert_gate.weight",
                    TensorRole::SharedExpertGateScalar,
                    vec![1, hidden],
                    DType::BF16,
                )?,
            }))
        }
        Qwen4FfnConfig::Moe {
            shared_intermediate_size: None,
            ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: "the Qwen4 executor binds only the shared-expert MoE FFN".to_string(),
        }),
    }
}

/// Whether `name` is one of the [`PLE_SHARD_COUNT`] canonical Flash-Next
/// n-gram PLE shard tensor names (`..ngram_embedding.shard_{0..127}.weight`,
/// no leading zeros).
pub fn is_ple_shard_name(name: &str) -> bool {
    is_ple_shard_name_within(name, PLE_SHARD_COUNT)
}

/// Whether `name` is a canonical PLE shard tensor name whose index is below
/// `shard_count` (the profile's declared [`Qwen4PleConfig::shard_count`]).
pub fn is_ple_shard_name_within(name: &str, shard_count: usize) -> bool {
    name.strip_prefix(hipfire_runtime::ple_sidecar::PLE_SHARD_PREFIX)
        .and_then(|name| name.strip_suffix(hipfire_runtime::ple_sidecar::PLE_SHARD_SUFFIX))
        .and_then(|digits| {
            let index = digits.parse::<usize>().ok()?;
            (index < shard_count && index.to_string() == digits).then_some(index)
        })
        .is_some()
}

/// Valid rows of Flash-Next PLE shard `shard`.  Profile-generic callers use
/// [`Qwen4PleConfig::shard_valid_rows`].
pub fn ple_valid_rows_for_shard(shard: usize) -> usize {
    crate::ple::PLE_VALID_ROWS
        .saturating_sub((shard as u64).saturating_mul(PLE_SHARD_ROWS as u64))
        .min(PLE_SHARD_ROWS as u64) as usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    Config(String),
    InvalidShape {
        name: String,
        shape: Vec<usize>,
    },
    ShapeOverflow(String),
    DescriptorMismatch(String),
    PlacementSetMismatch {
        expected: usize,
        actual: usize,
    },
    MissingPlacement {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingExternalDescriptor {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingResident {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    PleShardCount {
        expected: usize,
        actual: usize,
    },
    /// The profile is valid developer inventory (or a different identity) but
    /// the typed executor has no binder for it.
    UnsupportedProfile {
        profile: String,
        reason: String,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 manifest config: {message}"),
            Self::InvalidShape { name, shape } => write!(f, "{name}: invalid shape {shape:?}"),
            Self::ShapeOverflow(name) => write!(f, "{name}: shape overflow"),
            Self::DescriptorMismatch(name) => {
                write!(f, "{name}: external descriptor mismatch")
            }
            Self::PlacementSetMismatch { expected, actual } => {
                write!(f, "Qwen4 placement set expected {expected}, got {actual}")
            }
            Self::MissingPlacement {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident placement {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingExternalDescriptor {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing external descriptor {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingResident {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident tensor {name}[layer {layer:?}] on device {device}"
            ),
            Self::PleShardCount { expected, actual } => {
                write!(f, "Qwen4 PLE shard count expected {expected}, got {actual}")
            }
            Self::UnsupportedProfile { profile, reason } => {
                write!(f, "Qwen4 profile {profile:?} is not executable: {reason}")
            }
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ple_shards_exclude_padding() {
        let total: usize = (0..PLE_SHARD_COUNT).map(ple_valid_rows_for_shard).sum();
        assert_eq!(total, 320_001_446);
        assert_eq!(ple_valid_rows_for_shard(127), 2_499_922);
    }

    #[test]
    fn sidecar_manifest_drops_exactly_the_ple_shards() {
        let full = Qwen4Manifest::build(&pinned_config()).unwrap();
        assert_eq!(full.external_entries().count(), PLE_SHARD_COUNT);
        let trimmed = full.clone().without_ple_shards();
        assert_eq!(full.weights.len() - trimmed.weights.len(), PLE_SHARD_COUNT);
        assert_eq!(trimmed.external_entries().count(), 0);
        assert_eq!(trimmed.metadata, full.metadata);
        assert_eq!(trimmed.state, full.state);
        assert_eq!(
            full.source_tensor_count() - trimmed.source_tensor_count(),
            PLE_SHARD_COUNT
        );
        // Every kept entry is untouched and none is a shard.
        assert!(trimmed
            .weights
            .iter()
            .all(|entry| !is_ple_shard_name(&entry.name)));
        assert!(trimmed
            .weights
            .iter()
            .all(|entry| full.weights.contains(entry)));
    }

    #[test]
    fn ple_shard_name_predicate_is_canonical() {
        let name = |index: &str| {
            format!(
                "{}{index}{}",
                hipfire_runtime::ple_sidecar::PLE_SHARD_PREFIX,
                hipfire_runtime::ple_sidecar::PLE_SHARD_SUFFIX
            )
        };
        assert!(is_ple_shard_name(&name("0")));
        assert!(is_ple_shard_name(&name("127")));
        assert!(!is_ple_shard_name(&name("128")));
        assert!(!is_ple_shard_name(&name("007")));
        assert!(!is_ple_shard_name(&name("")));
        assert!(!is_ple_shard_name("model.embed_tokens.weight"));
    }
    fn pinned_config() -> Qwen4Config {
        crate::config::compact_test_config()
    }

    fn put(
        out: &mut Vec<WeightEntry>,
        layer: Option<usize>,
        name: String,
        shape: &[usize],
        dtype: DType,
        constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) {
        out.push(match layer {
            Some(layer) => WeightEntry::layer_with_dtype_constraint(
                &name,
                layer,
                shape.to_vec(),
                dtype,
                constraint,
                policy,
            ),
            None => WeightEntry::model_with_dtype_constraint(
                &name,
                shape.to_vec(),
                dtype,
                constraint,
                policy,
            ),
        });
    }

    fn oracle_bf16() -> DTypeConstraint {
        DTypeConstraint::source_exact(DType::BF16)
    }

    fn oracle_q8() -> DTypeConstraint {
        DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0])
    }

    fn oracle_trunk() -> DTypeConstraint {
        DTypeConstraint::source_from_sources(vec![
            DType::BF16,
            DType::Q8_0,
            DType::MQ6G256V2,
            DType::MQ4G256V2,
        ])
    }

    fn oracle_hc(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        for (suffix, shape, policy) in [
            (
                "block_inject_weight.weight",
                &[4usize, 10240][..],
                ShardPolicy::Replicate,
            ),
            ("hc_norm.weight", &[10240][..], ShardPolicy::Replicate),
            (
                "input_mix_weight_down.weight",
                &[320, 10240][..],
                ShardPolicy::ColumnShard { axis: 0 },
            ),
            (
                "input_mix_weight_up.weight",
                &[10240, 320][..],
                ShardPolicy::RowShard { axis: 1 },
            ),
        ] {
            put(
                out,
                layer,
                format!("{prefix}.{suffix}"),
                shape,
                DType::BF16,
                oracle_bf16(),
                policy,
            );
        }
    }

    fn oracle_qsa(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        let col = ShardPolicy::ColumnShard { axis: 0 };
        let row = ShardPolicy::RowShard { axis: 1 };
        let rep = ShardPolicy::Replicate;
        for (suffix, shape, packed, policy) in [
            ("indexer.index_qk_proj.weight", &[640usize, 2560][..], true, col.clone()),
            ("indexer.k_layernorm.weight", &[128][..], false, rep.clone()),
            ("indexer.q_layernorm.weight", &[128][..], false, rep.clone()),
            ("k_norm.weight", &[256][..], false, rep.clone()),
            ("k_proj.weight", &[512, 2560][..], true, col.clone()),
            ("o_proj.weight", &[2560, 6144][..], true, row),
            ("q_norm.weight", &[256][..], false, rep),
            ("q_proj.weight", &[12288, 2560][..], true, col.clone()),
            ("v_proj.weight", &[512, 2560][..], true, col),
        ] {
            let (dtype, constraint) = if packed {
                (DType::Q8_0, oracle_trunk())
            } else {
                (DType::BF16, oracle_bf16())
            };
            put(
                out,
                layer,
                format!("{prefix}.self_attn.{suffix}"),
                shape,
                dtype,
                constraint,
                policy,
            );
        }
    }

    fn oracle_moe(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        let experts = DTypeConstraint::source_from_sources(vec![
            DType::BF16,
            DType::MQ4G256V2,
            DType::MQ4G128V2,
        ]);
        let col = ShardPolicy::ColumnShard { axis: 0 };
        let rep = ShardPolicy::Replicate;
        let name = |suffix: &str| format!("{prefix}.mlp.{suffix}");
        put(
            out,
            layer,
            name("experts.down_proj"),
            &[512, 2560, 640],
            DType::MQ4G128V2,
            experts.clone(),
            rep.clone(),
        );
        put(
            out,
            layer,
            name("experts.gate_up_proj"),
            &[512, 1280, 2560],
            DType::MQ4G256V2,
            experts,
            rep.clone(),
        );
        put(
            out,
            layer,
            name("gate.weight"),
            &[512, 2560],
            DType::BF16,
            oracle_bf16(),
            rep.clone(),
        );
        put(
            out,
            layer,
            name("shared_expert.down_proj.weight"),
            &[2560, 640],
            DType::BF16,
            oracle_bf16(),
            ShardPolicy::RowShard { axis: 1 },
        );
        put(
            out,
            layer,
            name("shared_expert.gate_proj.weight"),
            &[640, 2560],
            DType::BF16,
            oracle_bf16(),
            col.clone(),
        );
        put(
            out,
            layer,
            name("shared_expert.up_proj.weight"),
            &[640, 2560],
            DType::BF16,
            oracle_bf16(),
            col,
        );
        put(
            out,
            layer,
            name("shared_expert_gate.weight"),
            &[1, 2560],
            DType::BF16,
            oracle_bf16(),
            rep,
        );
    }

    /// The complete pinned Flash-Next inventory, enumerated from literal names,
    /// shapes, dtypes, source constraints, shard policies and placements.  No
    /// tier/recipe helper or config field of this module derives any of it, so
    /// it is an independent oracle for the profile-owned generator.
    fn flash_next_oracle() -> Vec<WeightEntry> {
        let mut out = Vec::new();
        let head = DTypeConstraint::source_from_sources(vec![
            DType::BF16,
            DType::Q8_0,
            DType::MQ6G256V2,
        ]);
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "model.language_model.embed_tokens.weight",
                vec![248320, 2560],
                DType::Q8_0,
                oracle_q8(),
                ShardPolicy::Pin(PinTarget::Embed),
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "lm_head.weight",
                vec![248320, 2560],
                DType::MQ6G256V2,
                head,
                ShardPolicy::Pin(PinTarget::Output),
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "mtp.embed_tokens.weight",
                vec![248320, 2560],
                DType::Q8_0,
                oracle_q8(),
                ShardPolicy::Tied {
                    source: "model.language_model.embed_tokens.weight".to_string(),
                },
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        for (suffix, shape, policy) in [
            (
                "hc_norm.weight",
                &[10240usize][..],
                ShardPolicy::Pin(PinTarget::Output),
            ),
            (
                "input_mix_weight_down.weight",
                &[320, 10240][..],
                ShardPolicy::ColumnShard { axis: 0 },
            ),
            (
                "input_mix_weight_up.weight",
                &[10240, 320][..],
                ShardPolicy::RowShard { axis: 1 },
            ),
        ] {
            out.push(
                WeightEntry::model_with_dtype_constraint(
                    &format!("model.language_model.hyper_connection_mixer.{suffix}"),
                    shape.to_vec(),
                    DType::BF16,
                    oracle_bf16(),
                    policy,
                )
                .with_placement(PlacementHint::Pin(PinTarget::Output)),
            );
        }
        for layer in 0..48usize {
            let prefix = format!("model.language_model.layers.{layer}");
            oracle_hc(&mut out, Some(layer), &format!("{prefix}.attn_hyper_connection"));
            oracle_hc(&mut out, Some(layer), &format!("{prefix}.mlp_hyper_connection"));
            if layer % 4 == 3 {
                oracle_qsa(&mut out, Some(layer), &prefix);
            } else {
                let col = ShardPolicy::ColumnShard { axis: 0 };
                for (suffix, shape, packed, policy) in [
                    ("A_log", &[48usize][..], false, ShardPolicy::Replicate),
                    ("conv1d.weight", &[10240, 1, 4][..], false, ShardPolicy::Replicate),
                    ("dt_bias", &[48][..], false, ShardPolicy::Replicate),
                    ("in_proj_a.weight", &[48, 2560][..], true, col.clone()),
                    ("in_proj_b.weight", &[48, 2560][..], true, col.clone()),
                    ("in_proj_qkv.weight", &[10240, 2560][..], true, col.clone()),
                    ("in_proj_z.weight", &[6144, 2560][..], true, col),
                    ("norm.weight", &[128][..], false, ShardPolicy::Replicate),
                    (
                        "out_proj.weight",
                        &[2560, 6144][..],
                        true,
                        ShardPolicy::RowShard { axis: 1 },
                    ),
                ] {
                    let (dtype, constraint) = if packed {
                        (DType::Q8_0, oracle_trunk())
                    } else {
                        (DType::BF16, oracle_bf16())
                    };
                    put(
                        &mut out,
                        Some(layer),
                        format!("{prefix}.linear_attn.{suffix}"),
                        shape,
                        dtype,
                        constraint,
                        policy,
                    );
                }
            }
            oracle_moe(&mut out, Some(layer), &prefix);
            if layer == 1 {
                let ple = format!("{prefix}.ple");
                let col = ShardPolicy::ColumnShard { axis: 0 };
                for (suffix, shape, policy) in [
                    ("conv1d.weight", &[10240usize, 1, 4][..], ShardPolicy::Replicate),
                    ("key_proj.weight", &[10240, 2560][..], col),
                    ("norm_conv.weight", &[10240][..], ShardPolicy::Replicate),
                    ("norm_key.weight", &[10240][..], ShardPolicy::Replicate),
                    ("norm_query.weight", &[10240][..], ShardPolicy::Replicate),
                    (
                        "value_proj.weight",
                        &[2560, 2560][..],
                        ShardPolicy::RowShard { axis: 1 },
                    ),
                ] {
                    put(
                        &mut out,
                        Some(layer),
                        format!("{ple}.{suffix}"),
                        shape,
                        DType::BF16,
                        oracle_bf16(),
                        policy,
                    );
                }
                for shard in 0..128usize {
                    let valid = 320_001_446usize
                        .saturating_sub(shard * 2_500_012)
                        .min(2_500_012);
                    out.push(
                        WeightEntry::layer_with_dtype_constraint(
                            &format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight"),
                            layer,
                            vec![2_500_012, 160],
                            DType::Q8_0,
                            oracle_q8(),
                            ShardPolicy::Replicate,
                        )
                        .external_rows(170, valid),
                    );
                }
            }
        }
        let mtp = |out: &mut Vec<WeightEntry>, name: &str, shape: &[usize], policy: ShardPolicy| {
            put(
                out,
                None,
                name.to_string(),
                shape,
                DType::BF16,
                oracle_bf16(),
                policy,
            );
        };
        mtp(
            &mut out,
            "mtp.fc_embedding.weight",
            &[2560, 2560],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.fc_hidden.weight",
            &[2560, 2560],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.pre_fc_norm_embedding.weight",
            &[2560],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.pre_fc_norm_hidden.weight",
            &[10240],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.hc_norm.weight",
            &[10240],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            &[320, 10240],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            &[10240, 320],
            ShardPolicy::RowShard { axis: 1 },
        );
        oracle_hc(&mut out, None, "mtp.layers.0.attn_hyper_connection");
        oracle_hc(&mut out, None, "mtp.layers.0.mlp_hyper_connection");
        oracle_qsa(&mut out, None, "mtp.layers.0");
        oracle_moe(&mut out, None, "mtp.layers.0");
        out
    }

    /// Flash-Next is an exact profile: the profile-owned generator reproduces
    /// the pinned inventory entry for entry, in order — names, layer keys,
    /// logical shapes, target dtypes, source constraints, shard policies,
    /// placements and external-row residencies — plus the exact state and I64
    /// metadata inventories and the manifest identity.
    #[test]
    fn flash_next_manifest_is_byte_identical_to_the_independent_oracle() {
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(false)));
        let built = Qwen4Manifest::build_with_trunk_tier(&pinned_config(), DType::Q8_0);
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        let manifest = built.expect("pinned manifest");
        let oracle = flash_next_oracle();
        assert_eq!(manifest.weights.len(), 1323, "weights (1325 source records)");
        assert_eq!(oracle.len(), manifest.weights.len());
        for (index, (built, expected)) in manifest.weights.iter().zip(&oracle).enumerate() {
            assert_eq!(built, expected, "entry {index} {}", expected.name);
        }
        let mut state = Vec::new();
        for layer in 0..48usize {
            if layer % 4 == 3 {
                state.push(StateEntry::new(
                    StateKind::Kv {
                        quant: "qsa-full-raw-pooled".into(),
                    },
                    layer,
                ));
            } else {
                state.push(StateEntry::new(StateKind::Recurrent, layer));
                state.push(StateEntry::new(StateKind::Conv, layer));
            }
        }
        state.push(StateEntry::new(StateKind::Conv, 1));
        assert_eq!(manifest.state, state);
        assert_eq!(manifest.state.len(), 85);
        let metadata_base = "model.language_model.layers.1.ple.ple_embedding";
        assert_eq!(
            manifest.metadata,
            vec![
                MetadataTensor::i64(format!("{metadata_base}.layer_multipliers"), vec![3]),
                MetadataTensor::i64(format!("{metadata_base}.ngram_heads_offsets"), vec![16]),
                MetadataTensor::i64(format!("{metadata_base}.ngram_heads_vocab_sizes"), vec![16]),
            ]
        );
        assert_eq!(
            manifest.identity,
            Qwen4ManifestIdentity {
                profile_id: "qwen3.8-flash-next".to_string(),
                developer_only: false,
                equation_source: "pinned-upstream",
                ffn_kind: "moe",
                recipe: Qwen4TierRecipe::FlashNextPacked {
                    trunk_tier: DType::Q8_0,
                    mtp_source: false,
                },
            }
        );
    }

    fn synthetic(ffn: Qwen4FfnConfig, hidden: usize, layers: usize, ple: bool, mtp: bool) -> Qwen4Config {
        crate::config::synthetic_test_config(ffn, hidden, layers, ple, mtp)
    }

    fn dense_ffn() -> Qwen4FfnConfig {
        Qwen4FfnConfig::Dense {
            intermediate_size: 17_408,
        }
    }

    fn small_moe(shared: Option<usize>) -> Qwen4FfnConfig {
        Qwen4FfnConfig::Moe {
            num_experts: 8,
            top_k: 2,
            intermediate_size: 64,
            shared_intermediate_size: shared,
            norm_topk_prob: true,
        }
    }

    /// Expected weight count of a synthetic inventory from closed-form per-role
    /// counts (HC 4+4, GDN 9 or QSA 9, FFN roles, MTP block, PLE block).
    fn expected_weight_count(config: &Qwen4Config) -> usize {
        let ffn = match &config.ffn {
            Qwen4FfnConfig::Dense { .. } => 3,
            Qwen4FfnConfig::Moe {
                shared_intermediate_size,
                ..
            } => 3 + if shared_intermediate_size.is_some() { 4 } else { 0 },
        };
        // embedding, head, final HC mixer (3)
        let mut count = 5 + config.num_hidden_layers * (8 + 9 + ffn);
        if config.mtp.is_some() {
            // alias, fc x2, pre-fc norms x2, mixer x3, HC 8, QSA 9, FFN
            count += 1 + 4 + 3 + 8 + 9 + ffn;
        }
        if let Some(ple) = &config.ple {
            count += 6 + ple.shard_count;
        }
        count
    }

    fn assert_synthetic_bf16_inventory(config: &Qwen4Config, manifest: &Qwen4Manifest) {
        assert_eq!(manifest.weights.len(), expected_weight_count(config));
        assert_eq!(manifest.identity.profile_id, config.profile.id());
        assert!(manifest.identity.developer_only);
        assert_eq!(
            manifest.identity.equation_source,
            config.profile.equation_source().as_str()
        );
        assert_eq!(manifest.identity.recipe, Qwen4TierRecipe::SyntheticSourceBf16);
        for entry in &manifest.weights {
            assert_eq!(entry.dtype, DType::BF16, "{} must stay BF16", entry.name);
            assert!(entry.dtype_constraint.accepts(DType::BF16), "{}", entry.name);
            for packed in [
                DType::Q8_0,
                DType::MQ6G256V2,
                DType::MQ4G256V2,
                DType::MQ4G128V2,
            ] {
                assert!(
                    !entry.dtype_constraint.accepts(packed),
                    "{} must not admit {packed:?} before an executable role proof",
                    entry.name
                );
            }
        }
        assert_eq!(
            manifest.metadata.len(),
            if config.ple.is_some() { 3 } else { 0 }
        );
        assert_eq!(
            manifest.external_entries().count(),
            config.ple.as_ref().map_or(0, |ple| ple.shard_count)
        );
        let has = |needle: &str| manifest.weights.iter().any(|e| e.name.contains(needle));
        assert_eq!(has(".ple."), config.ple.is_some());
        assert_eq!(has("mtp."), config.mtp.is_some());
        let aliases = manifest
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        assert_eq!(aliases, usize::from(config.mtp.is_some()));
        let recurrent_layers = config
            .layer_types
            .iter()
            .filter(|kind| **kind == LayerType::LinearAttention)
            .count();
        let full_layers = config.layer_types.len() - recurrent_layers;
        assert_eq!(
            manifest.state.len(),
            recurrent_layers * 2 + full_layers + usize::from(config.ple.is_some())
        );
    }

    #[test]
    fn dense_64_layer_hidden_5120_inventory_has_only_gate_up_down_roles() {
        for (ple, mtp) in [(false, false), (true, false), (false, true), (true, true)] {
            let config = synthetic(dense_ffn(), 5120, 64, ple, mtp);
            let manifest = Qwen4Manifest::build(&config).expect("dense manifest");
            assert_synthetic_bf16_inventory(&config, &manifest);
            assert_eq!(manifest.identity.ffn_kind, "dense");
            for layer in [0usize, 63] {
                let name = |role: &str| {
                    format!("model.language_model.layers.{layer}.mlp.{role}.weight")
                };
                for (role, shape) in [
                    ("gate_proj", [17_408usize, 5120]),
                    ("up_proj", [17_408, 5120]),
                    ("down_proj", [5120, 17_408]),
                ] {
                    let entry = manifest
                        .entry(&name(role), Some(layer))
                        .unwrap_or_else(|| panic!("missing dense {role} at layer {layer}"));
                    assert_eq!(entry.logical_shape, shape);
                    assert!(!entry.residency.is_external(), "dense roles are resident");
                }
            }
            for entry in &manifest.weights {
                for forbidden in [".experts.", "shared_expert", ".mlp.gate.weight"] {
                    assert!(
                        !entry.name.contains(forbidden),
                        "dense profile declared MoE tensor {}",
                        entry.name
                    );
                }
            }
            // The wide HC width at hidden 5120 is hc_count * hidden.
            let mixer = manifest
                .entry(
                    "model.language_model.hyper_connection_mixer.hc_norm.weight",
                    None,
                )
                .expect("final HC norm");
            assert_eq!(mixer.logical_shape, [config.hc_count * 5120]);
        }
    }

    #[test]
    fn absent_ple_and_mtp_leave_no_tensors_metadata_aliases_or_state() {
        let config = synthetic(dense_ffn(), 5120, 64, false, false);
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        assert!(manifest.metadata.is_empty());
        assert_eq!(manifest.external_entries().count(), 0);
        assert!(manifest
            .weights
            .iter()
            .all(|entry| !entry.name.starts_with("mtp.") && !entry.name.contains(".ple.")));
        assert!(manifest
            .weights
            .iter()
            .all(|entry| !matches!(entry.policy, ShardPolicy::Tied { .. })));
        assert_eq!(manifest.source_tensor_count(), manifest.weights.len());
        let full = Qwen4Manifest::build(&synthetic(dense_ffn(), 5120, 64, true, true))
            .expect("full manifest");
        assert_eq!(full.state.len(), manifest.state.len() + 1, "PLE conv state");
        assert!(full.source_tensor_count() > manifest.source_tensor_count());
    }

    #[test]
    fn small_moe_inventory_with_and_without_a_shared_expert() {
        for shared in [Some(64usize), None] {
            let config = synthetic(small_moe(shared), 256, 4, true, true);
            let manifest = Qwen4Manifest::build(&config).expect("small moe manifest");
            assert_synthetic_bf16_inventory(&config, &manifest);
            assert_eq!(manifest.identity.ffn_kind, "moe");
            let layer = |role: &str| format!("model.language_model.layers.0.mlp.{role}");
            let expert = manifest
                .entry(&layer("experts.gate_up_proj"), Some(0))
                .expect("routed gate_up");
            assert_eq!(expert.logical_shape, [8, 128, 256]);
            let down = manifest
                .entry(&layer("experts.down_proj"), Some(0))
                .expect("routed down");
            assert_eq!(down.logical_shape, [8, 256, 64]);
            assert_eq!(
                manifest.entry(&layer("gate.weight"), Some(0)).unwrap().logical_shape,
                [8, 256]
            );
            for role in [
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
                "shared_expert_gate.weight",
            ] {
                assert_eq!(
                    manifest.entry(&layer(role), Some(0)).is_some(),
                    shared.is_some(),
                    "{role}"
                );
            }
            assert!(manifest
                .entry(&layer("gate_proj.weight"), Some(0))
                .is_none());
        }
    }

    #[test]
    fn ple_shard_inventory_follows_the_typed_geometry() {
        let config = synthetic(small_moe(Some(64)), 256, 4, true, false);
        let ple = config.ple.clone().expect("synthetic PLE");
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let shards = manifest.external_entries().collect::<Vec<_>>();
        assert_eq!(shards.len(), ple.shard_count);
        let mut total_valid = 0usize;
        for (index, entry) in shards.iter().enumerate() {
            assert_eq!(entry.layer, Some(ple.layer));
            assert_eq!(entry.logical_shape, [ple.shard_rows, ple.row_width]);
            let WeightResidency::ExternalRows {
                row_bytes,
                valid_rows,
            } = entry.residency
            else {
                panic!("PLE shard must be external rows");
            };
            assert_eq!(row_bytes, RowEncoding::Bf16.encoded_row_bytes(ple.row_width));
            assert_eq!(valid_rows, ple.shard_valid_rows(index));
            total_valid += valid_rows;
        }
        assert_eq!(total_valid as u64, ple.valid_rows);
        let base = format!("model.language_model.layers.{}.ple.ple_embedding", ple.layer);
        assert_eq!(
            manifest.metadata,
            vec![
                MetadataTensor::i64(format!("{base}.layer_multipliers"), vec![ple.multiplier_count]),
                MetadataTensor::i64(format!("{base}.ngram_heads_offsets"), vec![ple.head_count]),
                MetadataTensor::i64(
                    format!("{base}.ngram_heads_vocab_sizes"),
                    vec![ple.head_count]
                ),
            ]
        );
        let trimmed = manifest.clone().without_ple_shards();
        assert_eq!(trimmed.external_entries().count(), 0);
        assert_eq!(manifest.weights.len() - trimmed.weights.len(), ple.shard_count);
        assert!(is_ple_shard_name_within(
            &shards[0].name.replace(&format!("layers.{}.", ple.layer), "layers.1."),
            ple.shard_count
        ) || ple.layer != 1);
    }

    #[test]
    fn recipe_and_profile_must_agree_and_synthetic_ignores_tier_environment() {
        let synthetic_config = synthetic(small_moe(Some(64)), 256, 4, false, false);
        let error = Qwen4Manifest::build_with_recipe(
            &synthetic_config,
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: DType::MQ6G256V2,
                mtp_source: false,
            },
        )
        .unwrap_err();
        assert!(matches!(error, WeightError::UnsupportedProfile { .. }), "{error}");
        let error = Qwen4Manifest::build_with_recipe(
            &pinned_config(),
            Qwen4TierRecipe::SyntheticSourceBf16,
        )
        .unwrap_err();
        assert!(matches!(error, WeightError::UnsupportedProfile { .. }), "{error}");
        // The MTP source-tier switch and the trunk tier do not move a
        // synthetic recipe.
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(true)));
        let built = Qwen4Manifest::build(&synthetic(small_moe(Some(64)), 256, 4, false, true));
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        let manifest = built.expect("synthetic manifest");
        assert_eq!(manifest.identity.recipe, Qwen4TierRecipe::SyntheticSourceBf16);
        assert!(manifest.weights.iter().all(|entry| entry.dtype == DType::BF16));
    }

    /// PLE channels are `hc_count * hidden` and the key/value projections read
    /// `embed_dim` columns: with hidden 5120 and a 16-wide embedding the conv
    /// and norms are 10240 wide, key is [10240, 16] and value is [5120, 16]
    /// (the S0 dense-5120 source shapes).  Flash-Next (embed == hidden) keeps
    /// [10240, 2560] / [2560, 2560], asserted by the FN oracle.
    #[test]
    fn ple_channels_follow_hidden_and_projections_follow_embed_dim() {
        let config = synthetic(dense_ffn(), 5120, 64, true, false);
        let ple = config.ple.clone().expect("PLE");
        assert_ne!(ple.embed_dim, config.hidden_size, "mixed hidden/embed fixture");
        let channels = config.hc_count * config.hidden_size;
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let shape = |suffix: &str| {
            manifest
                .entry(
                    &format!("model.language_model.layers.{}.ple.{suffix}", ple.layer),
                    Some(ple.layer),
                )
                .unwrap_or_else(|| panic!("missing {suffix}"))
                .logical_shape
                .clone()
        };
        assert_eq!(shape("conv1d.weight"), [channels, 1, ple.conv_kernel_size]);
        assert_eq!(shape("key_proj.weight"), [channels, ple.embed_dim]);
        assert_eq!(shape("value_proj.weight"), [config.hidden_size, ple.embed_dim]);
        for norm in ["norm_conv.weight", "norm_key.weight", "norm_query.weight"] {
            assert_eq!(shape(norm), [channels], "{norm}");
        }
    }

    /// The HC inject operand has `hc_count` rows (not a literal 4): a
    /// synthetic profile with `hc_count = 2` declares `[2, wide]` for the
    /// attention and FFN connections of every layer and of the MTP layer,
    /// and the final mixers (trunk and MTP) declare no inject operand at all.
    #[test]
    fn hyper_connection_inject_rows_follow_hc_count() {
        let config = synthetic(small_moe(Some(64)), 256, 4, false, true);
        assert_eq!(config.hc_count, 2, "fixture must exercise a non-FN count");
        let wide = config.hc_count * config.hidden_size;
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let mut injects = 0usize;
        for entry in &manifest.weights {
            if entry.name.ends_with(".block_inject_weight.weight") {
                assert_eq!(entry.logical_shape, [config.hc_count, wide], "{}", entry.name);
                injects += 1;
            }
        }
        // attention + FFN per trunk layer, plus attention + FFN for MTP.
        assert_eq!(injects, 2 * config.num_hidden_layers + 2);
        for name in [
            "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
            "mtp.hyper_connection_mixer.block_inject_weight.weight",
        ] {
            assert!(manifest.entry(name, None).is_none(), "{name}");
        }
        for (name, layer) in [
            ("model.language_model.layers.1.mlp_hyper_connection", Some(1)),
            ("mtp.layers.0.attn_hyper_connection", None),
        ] {
            let shape = |suffix: &str| {
                manifest
                    .entry(&format!("{name}.{suffix}"), layer)
                    .unwrap_or_else(|| panic!("{name}.{suffix}"))
                    .logical_shape
                    .clone()
            };
            assert_eq!(shape("hc_norm.weight"), [wide]);
            assert_eq!(shape("input_mix_weight_down.weight"), [config.hc_lowrank, wide]);
            assert_eq!(shape("input_mix_weight_up.weight"), [wide, config.hc_lowrank]);
        }
    }

    #[test]
    fn ffn_shape_overflow_is_an_error_not_a_panic() {
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let bf16 = oracle_bf16();
        let quant = qwen4_quant_source();
        let mut weights = Vec::new();
        for ffn in [
            Qwen4FfnConfig::Dense {
                intermediate_size: usize::MAX,
            },
            Qwen4FfnConfig::Moe {
                num_experts: 8,
                top_k: 2,
                intermediate_size: usize::MAX / 2 + 1,
                shared_intermediate_size: None,
                norm_topk_prob: true,
            },
            Qwen4FfnConfig::Moe {
                num_experts: usize::MAX,
                top_k: 2,
                intermediate_size: 2,
                shared_intermediate_size: None,
                norm_topk_prob: true,
            },
        ] {
            let error = push_ffn_entries(&mut weights, &layer, "p", 0, 4096, &ffn, &bf16, &quant)
                .expect_err("overflowing FFN geometry must be refused");
            assert!(matches!(error, WeightError::ShapeOverflow(_)), "{error}");
        }
        assert!(weights.is_empty(), "nothing is declared past a refusal");
        let error = push_ffn_entries(
            &mut weights,
            &layer,
            "p",
            0,
            4096,
            &Qwen4FfnConfig::Dense { intermediate_size: 0 },
            &bf16,
            &quant,
        )
        .expect_err("a zero-width FFN is refused");
        assert!(matches!(error, WeightError::InvalidShape { .. }), "{error}");
    }

    #[test]
    fn executor_admits_flash_next_and_dense_and_refuses_synthetic_moe() {
        admit_executable_profile(&pinned_config()).expect("Flash-Next is executable");
        for config in [
            synthetic(dense_ffn(), 5120, 64, true, true),
            synthetic(dense_ffn(), 256, 4, false, false),
        ] {
            admit_executable_profile(&config).expect("dense developer profile is executable");
        }
        for shared in [Some(64), None] {
            let config = synthetic(small_moe(shared), 256, 4, true, true);
            let error = admit_executable_profile(&config).unwrap_err();
            let WeightError::UnsupportedProfile { profile, reason } = &error else {
                panic!("{error}");
            };
            assert_eq!(profile, config.profile.id());
            assert!(
                reason.contains("experts=") && reason.contains("top_k="),
                "{reason}"
            );
            if shared.is_some() {
                assert!(reason.contains("no sealed MoE route"), "{reason}");
            }
        }
    }

    #[test]
    fn assembly_declares_the_admitted_recipes_manifest() {
        // A packed synthetic artifact is declared packed, not as the
        // profile's default source-exact manifest `build` returns.
        let config = synthetic(dense_ffn(), 5120, 64, true, true);
        let default = Qwen4Manifest::build(&config).expect("default manifest");
        assert_eq!(
            default.identity.recipe,
            Qwen4TierRecipe::SyntheticSourceBf16
        );
        let packed = Qwen4Manifest::build_with_recipe(&config, Qwen4TierRecipe::SyntheticPacked)
            .expect("packed manifest");
        let assembled = assembly_manifest(&config, &packed.identity.recipe).expect("assembly");
        assert_eq!(assembled, packed);
        assert_eq!(assembled.identity.recipe, Qwen4TierRecipe::SyntheticPacked);
        assert_ne!(
            assembled.weights, default.weights,
            "the packed recipe declares packed formats"
        );
        // Same tensors, so the placement set is unchanged by the recipe.
        assert_eq!(
            assembled.resident_keys().collect::<Vec<_>>(),
            default.resident_keys().collect::<Vec<_>>()
        );
        // The source-exact recipe reproduces `build` exactly.
        let source = assembly_manifest(&config, &default.identity.recipe).expect("source assembly");
        assert_eq!(source, default);
        // A recipe the profile cannot be declared under is refused.
        let flash_next = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        };
        let error = assembly_manifest(&config, &flash_next).unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
    }

    #[test]
    fn flash_next_assembly_manifest_is_exactly_the_default_build() {
        // The recipe admission recorded (carrying the trunk-tier and
        // MTP-source knobs) rebuilds the manifest `Qwen4Manifest::build`
        // declares, tier overrides included.
        let config = pinned_config();
        let built = Qwen4Manifest::build(&config).expect("pinned manifest");
        let assembled = assembly_manifest(&config, &built.identity.recipe).expect("assembly");
        assert_eq!(assembled, built);
        for tier in [DType::Q8_0, DType::MQ6G256V2] {
            let tiered = Qwen4Manifest::build_with_trunk_tier(&config, tier).expect("tiered");
            let assembled =
                assembly_manifest(&config, &tiered.identity.recipe).expect("tiered assembly");
            assert_eq!(assembled, tiered);
            assert_eq!(
                assembled.resident_keys().collect::<Vec<_>>(),
                built.resident_keys().collect::<Vec<_>>()
            );
        }
        // A synthetic recipe cannot declare Flash-Next.
        let error = assembly_manifest(&config, &Qwen4TierRecipe::SyntheticPacked).unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
    }

    #[test]
    fn assembly_refuses_a_synthetic_moe_with_the_executor_text() {
        let config = synthetic(small_moe(Some(64)), 256, 4, true, true);
        let error = assembly_manifest(&config, &Qwen4TierRecipe::SyntheticSourceBf16).unwrap_err();
        let WeightError::UnsupportedProfile { reason, .. } = &error else {
            panic!("{error}");
        };
        assert!(reason.contains("no sealed MoE route"), "{reason}");
    }

    #[test]
    fn dense_binders_follow_the_profile_geometry() {
        let config = synthetic(dense_ffn(), 256, 4, true, true);
        let Qwen4FfnConfig::Dense { intermediate_size } = config.ffn else {
            panic!("dense profile");
        };
        let manifest = Qwen4Manifest::build(&config).expect("dense manifest");
        let layers = build_layer_refs(&config).expect("dense layer refs");
        let mtp = build_mtp_refs(&config).expect("dense mtp refs");
        let check = |reference: &TensorRef| {
            let entry = manifest
                .entry(&reference.name, reference.layer)
                .unwrap_or_else(|| panic!("manifest lacks {}", reference.name));
            assert_eq!(entry.logical_shape, reference.shape, "{}", reference.name);
        };
        for layer in &layers {
            let dense = layer.ffn.dense().expect("dense FFN binding");
            assert!(layer.ffn.moe().is_none());
            assert_eq!(
                dense.gate.shape,
                vec![intermediate_size, config.hidden_size]
            );
            assert_eq!(
                dense.down.shape,
                vec![config.hidden_size, intermediate_size]
            );
            for reference in [&dense.gate, &dense.up, &dense.down] {
                check(reference);
            }
            check(&layer.mlp_hyper.block_inject);
            assert_eq!(
                layer.mlp_hyper.block_inject.shape,
                vec![config.hc_count, config.hc_count * config.hidden_size]
            );
        }
        let dense = mtp.ffn.dense().expect("dense MTP FFN binding");
        for reference in [&dense.gate, &dense.up, &dense.down] {
            check(reference);
        }
        check(&mtp.mlp_hyper.block_inject);
    }

    #[test]
    fn manifest_matches_pinned_text_inventory_and_rejects_legacy_names() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        // The source has 1,658 records, of which 333 are vision records.  The
        // text manifest accounts for all 1,325 remaining records; its one MTP
        // embedding alias is excluded by source_tensor_count and its three
        // exact I64 PLE arrays are included through `metadata`.
        assert_eq!(manifest.source_tensor_count(), 1325);
        assert_eq!(manifest.source_tensor_count() + 333, 1658);
        let expect = |name: &str, layer: Option<usize>, shape: &[usize]| {
            let entry = manifest
                .entry(name, layer)
                .unwrap_or_else(|| panic!("missing exact inventory tensor {name} layer {layer:?}"));
            assert_eq!(entry.logical_shape, shape, "{name} shape");
        };
        expect(
            "model.language_model.layers.0.linear_attn.A_log",
            Some(0),
            &[48],
        );
        expect(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            Some(0),
            &[10240, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            Some(3),
            &[640, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Some(3),
            &[12288, 2560],
        );
        expect(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            Some(1),
            &[PLE_SHARD_ROWS, PLE_ROW_WIDTH],
        );
        expect("mtp.fc_hidden.weight", None, &[2560, 2560]);
        expect("mtp.layers.0.self_attn.q_proj.weight", None, &[12288, 2560]);
        expect(
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            None,
            &[10240],
        );
        assert!(manifest
            .entry("model.layers.0.linear_attn.in_proj_qkv.weight", Some(0))
            .is_none());
        assert!(manifest
            .entry(
                "model.language_model.layers.0.self_attn.qkv_proj.weight",
                Some(0)
            )
            .is_none());
        assert!(manifest
            .entry("model.language_model.layers.0.mlp.router.weight", Some(0))
            .is_none());
        assert!(manifest.metadata.iter().any(|tensor| {
            tensor.name == "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
                && tensor.source_dtype == "I64"
                && tensor.shape == [3]
        }));
        assert!(matches!(
            manifest
                .entry(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                    Some(1)
                )
                .expect("PLE shard"),
            WeightEntry {
                residency: WeightResidency::ExternalRows { .. },
                ..
            }
        ));
    }
    #[test]
    fn final_hyper_manifest_has_read_operands_only() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        assert!(
            manifest
                .entry(
                    "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
                    None,
                )
                .is_none(),
            "final HC must not invent a write operand"
        );
        let root = build_root_refs(&config).expect("root references");
        assert_eq!(
            root.final_hyper.hc_norm.name,
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        );
        let mtp = build_mtp_refs(&config).expect("MTP references");
        assert_eq!(
            mtp.final_hyper.input_mix_up.name,
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
        );
    }

    #[test]
    fn qwen4_moe_keeps_fused_source_replicated_for_single_execution() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        for name in [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
        ] {
            let entry = manifest
                .entry(name, Some(0))
                .unwrap_or_else(|| panic!("missing Qwen4 expert source {name}"));
            assert_eq!(
                entry.policy,
                ShardPolicy::Replicate,
                "{name} must be replicated for the sealed Single route"
            );
        }

        let layers = build_layer_refs(&config).expect("Qwen4 layer references");
        let moe = layers[0].ffn.moe().expect("Flash-Next binds the MoE FFN");
        match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused {
                gate_up,
                down,
                sidecars,
            } => {
                assert_eq!(
                    gate_up,
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"
                );
                assert_eq!(down, "model.language_model.layers.0.mlp.experts.down_proj");
                assert!(sidecars.is_empty());
            }
            other => panic!("pinned Qwen4 source inventory must stay PackedFused: {other:?}"),
        }
    }
    #[test]
    fn matrices_quantize_by_group_width_and_unrepresentable_k_stays_bf16() {
        let tensor = |name: &str, shape: &[usize]| {
            TensorRef::new(
                name,
                TensorRole::RoutedGateUp,
                Some(0),
                shape.to_vec(),
                DType::BF16,
            )
            .expect("tensor reference")
        };
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                &[2, 4, 256],
            )
            .dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.down_proj",
                &[2, 4, 128],
            )
            .dtype,
            ROUTED_DOWN_DTYPE,
            "a K the aligned group cannot carry falls to the row-local one"
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.gate_up_proj", &[2, 4, 256]).dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.down_proj", &[2, 4, 128]).dtype,
            ROUTED_DOWN_DTYPE
        );

        // Rank-2 matrices the packed set admits all take the eight-bit class,
        // whatever their alignment above the 128-weight admission boundary:
        // the block format carries any K, so there is no fallback tier.
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ] {
            for k in [256usize, 384] {
                assert_eq!(
                    tensor(name, &[2, k]).dtype,
                    DType::Q8_0,
                    "rank-2 matrix {name} K={k} must ship eight-bit"
                );
            }
        }

        // The token embedding is eight-bit; the language head is six-bit.
        assert_eq!(
            tensor("model.language_model.embed_tokens.weight", &[2, 256]).dtype,
            DType::Q8_0
        );
        assert_eq!(tensor("lm_head.weight", &[2, 256]).dtype, DType::MQ6G256V2);

        // The thinnest-over-widest reduction keeps source bytes whatever its K:
        // the hyper-connection mixer is four rows over a
        // `hc_count * hidden` reduction.
        for name in [
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        ] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} keeps source bytes"
            );
        }

        // K = 320 is neither 256- nor 128-aligned (the hyper-connection up
        // projection), and K = 160 is the PLE row width; both stay source BF16.
        for (name, k) in [
            (
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                320usize,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                160,
            ),
        ] {
            assert_eq!(
                tensor(name, &[2, k]).dtype,
                DType::BF16,
                "{name} K={k} must stay source-exact BF16"
            );
        }

        // The MTP connector matrices still have a source-BF16 owner.
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} stays source-exact BF16 until its owner reads quantized weights"
            );
        }
    }

    /// The trunk recipe is a decision, not a default: the wide attention/GDN
    /// projections ship eight-bit because the paired candidate tiers are worse
    /// here — four bits on a dense projection (which writes straight into the
    /// residual stream) produced a model that opened a reasoning block it could
    /// not close, and BF16 is the unpacked path this packing exists to replace.
    /// Either fallback changes the dtype assertion or the count below.  The
    /// eight-bit tier is what an unset `HIPFIRE_QWEN4_TRUNK_TIER` declares; the
    /// six-bit rung is the selector's, and is asserted below.
    #[test]
    fn mtp_tier_knob_reaches_the_five_attention_matrices() {
        let config = pinned_config();
        // Measured on the published rung: these five ship qt=3 (Q8F16) while the
        // trunk carries the mq6 rung, and the knob must move exactly these.
        let mtp_attention = [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        ];
        let trunk_attention = "model.language_model.layers.2.linear_attn.in_proj_qkv.weight";
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(false)));
        let shipped = Qwen4Manifest::build(&config).expect("default manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(true)));
        let source = Qwen4Manifest::build(&config).expect("source-tier manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        for name in mtp_attention {
            let default_entry = shipped
                .entry(name, None)
                .expect("default MTP attention entry");
            let source_entry = source
                .entry(name, None)
                .expect("source-tier MTP attention entry");
            assert_ne!(default_entry.dtype, DType::BF16, "{name} ships quantized");
            assert_eq!(
                source_entry.dtype,
                DType::BF16,
                "{name} must follow HIPFIRE_QWEN4_MTP_TIER=source"
            );
        }
        // The rank-3 routed experts keep their packed tier: the expert writer
        // carries no BF16 arm, and the knob is scoped to rank-2 matrices.
        for name in [
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        ] {
            let default_entry = shipped.entry(name, None).expect("default expert entry");
            let source_entry = source.entry(name, None).expect("source-tier expert entry");
            assert_eq!(
                default_entry.dtype, source_entry.dtype,
                "{name} must not move with the matrix knob"
            );
        }
        let default_trunk = shipped
            .entry(trunk_attention, Some(2))
            .expect("trunk entry");
        let source_trunk = source.entry(trunk_attention, Some(2)).expect("trunk entry");
        assert_eq!(
            default_trunk.dtype, source_trunk.dtype,
            "trunk tier must not move"
        );
        let norm = source
            .entry("mtp.layers.0.self_attn.q_norm.weight", None)
            .expect("MTP norm entry");
        assert_eq!(
            norm.dtype,
            DType::BF16,
            "MTP norms are already source-exact"
        );
    }

    #[test]
    fn mtp_source_tier_knob_only_answers_to_its_own_value() {
        assert!(!qwen4_mtp_source_tier(None), "unset keeps the shipped tier");
        assert!(!qwen4_mtp_source_tier(Some("mq6")));
        assert!(qwen4_mtp_source_tier(Some("source")));
    }

    #[test]
    fn trunk_attention_projections_are_pinned_to_eight_bits() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        let mut packed = 0usize;
        for entry in &manifest.weights {
            let trunk_attention =
                entry.name.contains(".linear_attn.") || entry.name.contains(".self_attn.");
            if !trunk_attention || entry.logical_shape.len() != 2 {
                continue;
            }
            let k = *entry.logical_shape.last().expect("rank checked above");
            if k % 128 != 0 {
                continue;
            }
            assert_eq!(
                entry.dtype,
                DType::Q8_0,
                "{} is a trunk attention projection and must ship at eight bits",
                entry.name
            );
            packed += 1;
        }
        // 36 linear-attention layers x (qkv, z, in_proj_a, in_proj_b, out_proj)
        // + 12 full-attention layers x (q, k, v, o, indexer qk)
        // + the MTP layer's five full-attention projections.
        assert_eq!(packed, 245, "trunk packing count");
    }

    /// The trunk's declared tier follows `HIPFIRE_QWEN4_TRUNK_TIER`, and only
    /// the trunk's does.
    ///
    /// The default the line above pins is the eight-bit recipe; `mq6` is the
    /// only value that selects the six-bit rung, and it moves the two hundred
    /// and forty rank-2 trunk attention/GDN matrices — not the MTP namespace,
    /// which reaches the manifest through its own constructor and keeps its own
    /// targets.
    #[test]
    fn trunk_tier_selector_moves_the_declared_trunk_target() {
        assert_eq!(qwen4_trunk_tier(None), DType::Q8_0, "unset knob default");
        assert_eq!(qwen4_trunk_tier(Some("mq6")), DType::MQ6G256V2);
        assert_eq!(
            qwen4_trunk_tier(Some("mfp4e8")),
            DType::Q8_0,
            "a retired value must not resurrect a third tier"
        );

        let config = pinned_config();
        let six = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(Some("mq6")))
            .expect("six-bit trunk manifest");
        let eight = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(None))
            .expect("eight-bit trunk manifest");
        assert_eq!(
            six.weights.len(),
            eight.weights.len(),
            "tier, not the entry set"
        );

        let mut moved = 0usize;
        let mut held = 0usize;
        for (six_entry, eight_entry) in six.weights.iter().zip(&eight.weights) {
            assert_eq!(six_entry.name, eight_entry.name, "entries must not move");
            let trunk_matrix = qwen4_trunk_matrix(&six_entry.name, &six_entry.logical_shape)
                && six_entry.layer.is_some();
            // The layer index is what separates the trunk declaration from the
            // MTP namespace's: MTP names are layerless by construction.
            if trunk_matrix {
                assert_eq!(
                    six_entry.dtype,
                    DType::MQ6G256V2,
                    "{} trunk target under HIPFIRE_QWEN4_TRUNK_TIER=mq6",
                    six_entry.name
                );
                assert_eq!(
                    eight_entry.dtype,
                    DType::Q8_0,
                    "{} trunk target with the knob unset",
                    eight_entry.name
                );
                moved += 1;
            } else {
                assert_eq!(
                    six_entry.dtype, eight_entry.dtype,
                    "{} must not follow the trunk selector",
                    six_entry.name
                );
            }
            // Under either declaration the trunk keeps admitting both rungs,
            // which is what keeps a published artifact of the other tier
            // loadable by this build.
            if trunk_matrix {
                for tier in [DType::BF16, DType::Q8_0, DType::MQ6G256V2] {
                    assert!(
                        six_entry.dtype_constraint.accepts(tier),
                        "{} must keep admitting {tier:?}",
                        six_entry.name
                    );
                }
            }
        }
        assert_eq!(moved, 240, "trunk matrices the selector moves");

        // The MTP namespace's five attention projections are the same class
        // under a separate declaration: Q8F16 under the six-bit build too.
        for entry in &six.weights {
            if entry.name.starts_with("mtp.")
                && qwen4_trunk_matrix(&entry.name, &entry.logical_shape)
            {
                assert_eq!(entry.dtype, DType::Q8_0, "{} MTP tier", entry.name);
                held += 1;
            }
        }
        assert_eq!(held, 5, "MTP attention projections left alone");
    }

    #[test]
    fn manifest_admits_quantized_matrices_and_keeps_source_exact_payloads_bf16() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        for (name, layer, dtype) in [
            (
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                Some(3),
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "model.language_model.layers.3.mlp.experts.down_proj",
                Some(3),
                ROUTED_DOWN_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                None,
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.down_proj",
                None,
                ROUTED_DOWN_DTYPE,
            ),
        ] {
            let entry = manifest.entry(name, layer).expect("routed expert entry");
            assert_eq!(entry.dtype, dtype, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(dtype), "{name} source");
            assert!(
                !entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must not accept an aligned-K family its geometry is not"
            );
        }

        // Dense matrices are quantized targets that still accept BF16 source
        // bytes, which is what lets an older artifact load against a newer
        // manifest.  The trunk's wide attention/GDN projections are also the
        // class whose packed tier the artifact chooses: the six-bit rung
        // carries MQ6G256V2 (qt47) there and the eight-bit rung Q8F16 (qt3).
        // Both must satisfy the entry's source contract, or a published rung
        // of one tier cannot be loaded by the tree that produced it.
        let assert_quantized = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("quantized matrix entry");
            assert_eq!(entry.dtype, DType::Q8_0, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(DType::Q8_0));
            assert!(
                entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must admit the six-bit trunk rung"
            );
        };
        for (name, layer) in [
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                Some(3),
            ),
            (
                "model.language_model.layers.2.linear_attn.in_proj_qkv.weight",
                Some(2),
            ),
            // The MTP layer's attention projections are the same class under
            // the model namespace, and reach the manifest through their own
            // constructor.
            ("mtp.layers.0.self_attn.q_proj.weight", None),
        ] {
            assert_quantized(name, layer);
        }

        let assert_bf16 = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("BF16 entry");
            assert_eq!(entry.dtype, DType::BF16, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G128V2));
        };
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_bf16(name, None);
        }
        // The embedding: BF16 in the checkpoint, Q8 in the artifact.
        let entry = manifest
            .entry("model.language_model.embed_tokens.weight", None)
            .expect("embedding entry");
        assert_eq!(entry.dtype, DType::Q8_0);
        assert!(entry.dtype_constraint.accepts(DType::BF16));
        assert!(entry.dtype_constraint.accepts(DType::Q8_0));
        assert!(!entry.dtype_constraint.accepts(DType::MQ6G256V2));
        // The head ships six-bit and still admits the earlier Q8F16 tier.
        let entry = manifest.entry("lm_head.weight", None).expect("head entry");
        assert_eq!(entry.dtype, DType::MQ6G256V2);
        for dtype in [DType::BF16, DType::Q8_0, DType::MQ6G256V2] {
            assert!(
                entry.dtype_constraint.accepts(dtype),
                "head admits {dtype:?}"
            );
        }
        assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
        // The shared expert, the router, and the PLE projections stay
        // source-exact: they ride the sealed-MoE route or select among 512
        // experts, and neither is a wide attention projection.
        assert_bf16("model.language_model.layers.3.mlp.gate.weight", Some(3));
        assert_bf16(
            "model.language_model.layers.3.mlp.shared_expert.up_proj.weight",
            Some(3),
        );
        let ple = manifest
            .weights
            .iter()
            .find(|entry| entry.name.contains(".ple.") && entry.name.contains("key"))
            .map(|entry| (entry.name.clone(), entry.layer))
            .expect("PLE key projection entry");
        assert_bf16(&ple.0, ple.1);
        // The hyper-connection mixer and block-inject matrices are the thin
        // side of the widest reduction in the network; they keep source bytes.
        assert_bf16(
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        );
        assert_bf16(
            "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight",
            Some(3),
        );

        let ple = manifest
            .entry(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                Some(1),
            )
            .expect("PLE shard");
        assert_eq!(ple.dtype, DType::Q8_0);
        assert!(ple.dtype_constraint.accepts(DType::Q8_0));
        // A sealed artifact written before the Q8F16 tier carries BF16 rows and
        // must stay loadable: both tiers are admitted by the same declaration.
        assert!(ple.dtype_constraint.accepts(DType::BF16));
        assert!(!ple.dtype_constraint.accepts(DType::MQ4G128V2));
    }
}
