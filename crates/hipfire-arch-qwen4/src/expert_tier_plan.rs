// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure CPU per-expert placement plan (B5-C1): which experts live in the
//! device-resident owner and which in the host-mapped owner.
//!
//! One mechanism serves pure-MQ4 HotCache, Consumer24 B5 (MQ4 + MQ2-I4) and
//! sub-4 K4 (MQ4 + MQ3-I4). Placement never changes format, numerics, slot
//! order or ids. The **model id** of an expert is `layer * 512 + expert`.
//!
//! * At most two model-level payload owners exist: one [`Placement::Device`]
//!   and one [`Placement::HostMapped`]. An empty tier allocates nothing
//!   (`owner_bytes == 0`).
//! * Every projection starts 256-byte aligned inside its owner
//!   ([`ExpertTierPlan::PAYLOAD_ALIGN`]); every non-empty owner carries
//!   [`ARENA_TAIL_PAD_BYTES`] (at least 256 B) of zeroed tail pad.
//! * Offsets and lengths are checked `u64`; model-level owners exceed 4 GiB.
//! * Destination layout inside a tier is `(layer, expert)` ascending, `gate_up`
//!   then `down`, independent of the rank order that chose the set.
//!
//! No GPU, environment or file access. Route/arch admission of the tier route
//! as a whole belongs to C2 admission, not to [`ExpertTierPlan::check_executable`],
//! which only refuses expert *formats* that have no kernel on an arch.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan, ExpertSlabTable};
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};

/// Codec id of an MQ2-I4 (qt133/qt134) expert. Reserved; refused by
/// [`ExpertTierPlan::check_executable`] until W2 kernels and route admission land.
pub const CODEC_W2: u8 = 2;

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const TOTAL_EXPERTS: usize = TRUNK_LAYERS as usize * EXPERTS;
const DIGEST_DOMAIN: &[u8] = b"hipfire.expert_tier_plan";

/// Which model-level owner holds an expert's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Device-resident owner.
    Device,
    /// Pinned, device-mapped host owner.
    HostMapped,
}

/// One expert's format and its destination inside its tier's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertTierEntry {
    /// From the artifact; never altered by placement.
    pub format: ExpertFormat,
    pub tier: Placement,
    /// Relative to the base of THIS entry's tier owner.
    pub gate_up_offset: u64,
    pub gate_up_len: u64,
    /// Relative to the base of THIS entry's tier owner.
    pub down_offset: u64,
    pub down_len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertTierError {
    /// A rank id is `>= 24,576`.
    RankIdOutOfRange { id: u32 },
    /// A rank id appears twice.
    RankDuplicate { id: u32 },
    /// A source span is empty, overflows, or the expert's two spans overlap.
    SourceSpan { layer: usize, expert: usize },
    /// Checked `u64` arithmetic overflowed.
    Overflow,
    /// The expert format has no kernel on `arch`.
    UnsupportedFormat { format: ExpertFormat, arch: String },
    /// The slab list is not exactly `48 * 512 = 24,576` entries.
    SlabCount { got: usize },
    /// An aggregate (legacy) artifact projection is malformed
    /// ([`crate::expert_slab_aggregate`]).
    Aggregate {
        layer: usize,
        reason: &'static str,
    },
}

impl fmt::Display for ExpertTierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankIdOutOfRange { id } => {
                write!(f, "rank id {id} is out of range (model ids are < {TOTAL_EXPERTS})")
            }
            Self::RankDuplicate { id } => write!(f, "rank id {id} appears more than once"),
            Self::SourceSpan { layer, expert } => write!(
                f,
                "layer {layer} expert {expert} has an empty, overflowing or overlapping source span"
            ),
            Self::Overflow => write!(f, "expert tier plan arithmetic overflowed u64"),
            Self::UnsupportedFormat { format, arch } => {
                write!(f, "expert format {format:?} has no kernels on arch {arch:?}")
            }
            Self::SlabCount { got } => write!(
                f,
                "expected exactly {TOTAL_EXPERTS} expert slabs, got {got}"
            ),
            Self::Aggregate { layer, reason } => {
                write!(f, "aggregate expert tensors of layer {layer}: {reason}")
            }
        }
    }
}

impl std::error::Error for ExpertTierError {}

/// Which projection of an expert a [`TierCopy`] moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TierProj {
    GateUp,
    Down,
}

/// One bounded source-to-owner copy of the materialization pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierCopy {
    pub layer: usize,
    pub expert: usize,
    pub proj: TierProj,
    /// Source span inside the artifact.
    pub src: ExpertSlabSpan,
    /// Destination offset inside the owner of the copy's tier.
    pub dst_offset: u64,
}

/// Extra payload spans (loaded MTP expert payloads) placed in the HOST owner
/// after the trunk host experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostWindow {
    pub label: String,
    /// `(artifact span, destination offset inside the host owner)`.
    pub spans: Vec<(ExpertSlabSpan, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertTierPlan {
    entries: Vec<ExpertTierEntry>,
    source: Vec<(ExpertSlabSpan, ExpertSlabSpan)>,
    device_bytes: u64,
    host_bytes: u64,
    device_experts: usize,
    host_windows: Vec<HostWindow>,
    artifact_sha256: Option<[u8; 32]>,
    digest: [u8; 32],
}

fn align_up(x: u64) -> Option<u64> {
    let a = ExpertTierPlan::PAYLOAD_ALIGN;
    Some(x.checked_add(a - 1)? & !(a - 1))
}

fn tail_pad() -> u64 {
    // `max(ARENA_TAIL_PAD_BYTES, 256)` rounded to 256; the constant is far from u64::MAX.
    align_up((ARENA_TAIL_PAD_BYTES as u64).max(ExpertTierPlan::PAYLOAD_ALIGN))
        .expect("tail pad fits u64")
}

fn tier_index(tier: Placement) -> usize {
    match tier {
        Placement::Device => 0,
        Placement::HostMapped => 1,
    }
}

fn digest_of(
    entries: &[ExpertTierEntry],
    source: &[(ExpertSlabSpan, ExpertSlabSpan)],
    device_bytes: u64,
    host_bytes: u64,
    windows: &[HostWindow],
    artifact_sha256: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(DIGEST_DOMAIN);
    h.update(ExpertTierPlan::VERSION.to_le_bytes());
    match artifact_sha256 {
        Some(sha) => {
            h.update([1u8]);
            h.update(sha);
        }
        None => {
            h.update([0u8]);
            h.update([0u8; 32]);
        }
    }
    for (entry, (gu, dn)) in entries.iter().zip(source) {
        let (gu_qt, dn_qt) = entry.format.pair();
        h.update([gu_qt, dn_qt]);
        h.update((gu.offset as u64).to_le_bytes());
        h.update((gu.len as u64).to_le_bytes());
        h.update((dn.offset as u64).to_le_bytes());
        h.update((dn.len as u64).to_le_bytes());
        h.update([tier_index(entry.tier) as u8]);
        h.update(entry.gate_up_offset.to_le_bytes());
        h.update(entry.gate_up_len.to_le_bytes());
        h.update(entry.down_offset.to_le_bytes());
        h.update(entry.down_len.to_le_bytes());
    }
    h.update(device_bytes.to_le_bytes());
    h.update(host_bytes.to_le_bytes());
    for window in windows {
        h.update((window.label.len() as u64).to_le_bytes());
        h.update(window.label.as_bytes());
        h.update((window.spans.len() as u64).to_le_bytes());
        for (span, dst) in &window.spans {
            h.update((span.offset as u64).to_le_bytes());
            h.update((span.len as u64).to_le_bytes());
            h.update(dst.to_le_bytes());
        }
    }
    h.finalize().into()
}

impl ExpertTierPlan {
    /// Version of the plan layout and digest stream.
    pub const VERSION: u32 = 1;
    /// Alignment of every projection start inside an owner.
    pub const PAYLOAD_ALIGN: u64 = 256;

    fn index(layer: usize, expert: usize) -> Option<usize> {
        (layer < TRUNK_LAYERS as usize && expert < EXPERTS).then_some(layer * EXPERTS + expert)
    }

    pub fn entry(&self, layer: usize, expert: usize) -> Option<&ExpertTierEntry> {
        self.entries.get(Self::index(layer, expert)?)
    }

    /// `(gate_up, down)` source spans in the artifact.
    pub fn source(&self, layer: usize, expert: usize) -> Option<(ExpertSlabSpan, ExpertSlabSpan)> {
        self.source.get(Self::index(layer, expert)?).copied()
    }

    /// Padded owner size including the tail pad; 0 means no allocation.
    pub fn owner_bytes(&self, tier: Placement) -> u64 {
        match tier {
            Placement::Device => self.device_bytes,
            Placement::HostMapped => self.host_bytes,
        }
    }

    pub fn device_experts(&self) -> usize {
        self.device_experts
    }

    /// SHA-256 of the canonical little-endian plan stream (receipts and
    /// diagnostics only, never authorization).
    pub fn logical_digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn host_windows(&self) -> &[HostWindow] {
        &self.host_windows
    }

    /// Refuse expert formats that have no kernel on `arch`: W2 everywhere,
    /// W3 anywhere but `gfx1201`; MQ4 is always accepted. Route/arch admission
    /// of the tier route as a whole belongs to C2 admission, not here.
    pub fn check_executable(&self, arch: &str) -> Result<(), ExpertTierError> {
        for entry in &self.entries {
            let ok = match entry.format {
                ExpertFormat::Mq4Xts => true,
                ExpertFormat::Mq3I4 => arch == "gfx1201",
                ExpertFormat::Mq2I4 => false,
            };
            if !ok {
                return Err(ExpertTierError::UnsupportedFormat {
                    format: entry.format,
                    arch: arch.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Copies of `tier` in deterministic `(layer, expert, gate_up, down)` order.
    pub fn copies(&self, tier: Placement) -> impl Iterator<Item = TierCopy> + '_ {
        self.entries
            .iter()
            .zip(&self.source)
            .enumerate()
            .filter(move |(_, (entry, _))| entry.tier == tier)
            .flat_map(|(i, (entry, (gu, dn)))| {
                let layer = i / EXPERTS;
                let expert = i % EXPERTS;
                [
                    TierCopy {
                        layer,
                        expert,
                        proj: TierProj::GateUp,
                        src: *gu,
                        dst_offset: entry.gate_up_offset,
                    },
                    TierCopy {
                        layer,
                        expert,
                        proj: TierProj::Down,
                        src: *dn,
                        dst_offset: entry.down_offset,
                    },
                ]
            })
    }

    /// Append `spans` (e.g. loaded MTP expert payloads) to the host owner after
    /// the trunk host experts. The host owner grows; its tail pad is counted once.
    pub fn with_host_window(
        mut self,
        label: &str,
        spans: &[ExpertSlabSpan],
    ) -> Result<Self, ExpertTierError> {
        let tail = tail_pad();
        // End of the host payloads, before the tail pad.
        let mut cursor = self.host_bytes.saturating_sub(tail);
        let mut placed = Vec::with_capacity(spans.len());
        for span in spans {
            if span.len == 0 || span.end().is_none() {
                return Err(ExpertTierError::Overflow);
            }
            let dst = align_up(cursor).ok_or(ExpertTierError::Overflow)?;
            cursor = dst
                .checked_add(span.len as u64)
                .ok_or(ExpertTierError::Overflow)?;
            placed.push((*span, dst));
        }
        self.host_bytes = if cursor == 0 {
            0
        } else {
            align_up(cursor)
                .and_then(|c| c.checked_add(tail))
                .ok_or(ExpertTierError::Overflow)?
        };
        self.host_windows.push(HostWindow {
            label: label.to_owned(),
            spans: placed,
        });
        self.digest = self.recompute_digest();
        Ok(self)
    }

    /// Pin the artifact SHA-256 into the plan digest.
    pub fn with_artifact_sha256(mut self, sha: [u8; 32]) -> Self {
        self.artifact_sha256 = Some(sha);
        self.digest = self.recompute_digest();
        self
    }

    fn recompute_digest(&self) -> [u8; 32] {
        digest_of(
            &self.entries,
            &self.source,
            self.device_bytes,
            self.host_bytes,
            &self.host_windows,
            self.artifact_sha256,
        )
    }
}

/// Plan placement from the K4-validated span table.
pub fn plan_expert_tiers(
    slabs: &ExpertSlabTable,
    rank: &[u32],
    vram_budget: u64,
) -> Result<ExpertTierPlan, ExpertTierError> {
    plan_expert_tiers_from_slabs(slabs.slabs(), rank, vram_budget)
}

/// `rank`: unique model ids in priority order (it may be a subset); omitted ids
/// are host-mapped. The device set is the LONGEST rank PREFIX whose padded
/// payloads plus tail pad fit `vram_budget`; the walk stops at the first id
/// that does not fit (no skipping). `slabs` must hold exactly 24,576 entries in
/// `layer * 512 + expert` order.
pub fn plan_expert_tiers_from_slabs(
    slabs: &[ExpertSlab],
    rank: &[u32],
    vram_budget: u64,
) -> Result<ExpertTierPlan, ExpertTierError> {
    if slabs.len() != TOTAL_EXPERTS {
        return Err(ExpertTierError::SlabCount { got: slabs.len() });
    }

    // 1. Validate sources and compute each expert's padded size.
    let mut padded = Vec::with_capacity(TOTAL_EXPERTS);
    for (i, slab) in slabs.iter().enumerate() {
        let bad = || ExpertTierError::SourceSpan {
            layer: i / EXPERTS,
            expert: i % EXPERTS,
        };
        let (gu, dn) = (slab.gate_up, slab.down);
        if gu.len == 0 || dn.len == 0 {
            return Err(bad());
        }
        let gu_end = gu.end().ok_or_else(bad)?;
        let dn_end = dn.end().ok_or_else(bad)?;
        if !(gu_end <= dn.offset || dn_end <= gu.offset) {
            return Err(bad());
        }
        let gu_pad = align_up(gu.len as u64).ok_or(ExpertTierError::Overflow)?;
        let dn_pad = align_up(dn.len as u64).ok_or(ExpertTierError::Overflow)?;
        padded.push(gu_pad.checked_add(dn_pad).ok_or(ExpertTierError::Overflow)?);
    }

    // 2. Validate rank.
    let mut seen = vec![false; TOTAL_EXPERTS];
    for &id in rank {
        let idx = id as usize;
        if idx >= TOTAL_EXPERTS {
            return Err(ExpertTierError::RankIdOutOfRange { id });
        }
        if std::mem::replace(&mut seen[idx], true) {
            return Err(ExpertTierError::RankDuplicate { id });
        }
    }

    // 3./4. Longest rank prefix that fits, tail pad included.
    let tail = tail_pad();
    let mut on_device = vec![false; TOTAL_EXPERTS];
    let mut used = 0u64;
    let mut device_experts = 0usize;
    for &id in rank {
        let idx = id as usize;
        let next = used
            .checked_add(padded[idx])
            .ok_or(ExpertTierError::Overflow)?;
        let with_tail = next.checked_add(tail).ok_or(ExpertTierError::Overflow)?;
        if with_tail > vram_budget {
            break;
        }
        used = next;
        on_device[idx] = true;
        device_experts += 1;
    }

    // 5. Layout, (layer, expert) ascending within each tier.
    let mut cursor = [0u64; 2];
    let mut count = [0usize; 2];
    let mut entries = Vec::with_capacity(TOTAL_EXPERTS);
    let mut source = Vec::with_capacity(TOTAL_EXPERTS);
    for (i, slab) in slabs.iter().enumerate() {
        let tier = if on_device[i] {
            Placement::Device
        } else {
            Placement::HostMapped
        };
        let t = tier_index(tier);
        let gate_up_len = slab.gate_up.len as u64;
        let down_len = slab.down.len as u64;
        let gate_up_offset = align_up(cursor[t]).ok_or(ExpertTierError::Overflow)?;
        let after_gu = gate_up_offset
            .checked_add(gate_up_len)
            .ok_or(ExpertTierError::Overflow)?;
        let down_offset = align_up(after_gu).ok_or(ExpertTierError::Overflow)?;
        cursor[t] = down_offset
            .checked_add(down_len)
            .ok_or(ExpertTierError::Overflow)?;
        count[t] += 1;
        entries.push(ExpertTierEntry {
            format: slab.format,
            tier,
            gate_up_offset,
            gate_up_len,
            down_offset,
            down_len,
        });
        source.push((slab.gate_up, slab.down));
    }
    let owner = |t: usize| -> Result<u64, ExpertTierError> {
        if count[t] == 0 {
            return Ok(0);
        }
        align_up(cursor[t])
            .and_then(|c| c.checked_add(tail))
            .ok_or(ExpertTierError::Overflow)
    };
    let device_bytes = owner(0)?;
    let host_bytes = owner(1)?;
    assert_eq!(
        device_bytes,
        if device_experts == 0 { 0 } else { used + tail },
        "device owner layout must equal the admitted prefix size"
    );

    let digest = digest_of(&entries, &source, device_bytes, host_bytes, &[], None);
    Ok(ExpertTierPlan {
        entries,
        source,
        device_bytes,
        host_bytes,
        device_experts,
        host_windows: Vec::new(),
        artifact_sha256: None,
        digest,
    })
}

/// Model ids ordered by descending count, ties by model id ascending, keeping
/// only ids for which `eligible` is true. Deterministic.
pub fn rank_from_counts(counts: &[u64], eligible: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..counts.len())
        .filter_map(|i| u32::try_from(i).ok())
        .filter(|&id| eligible(id))
        .collect();
    ids.sort_unstable_by(|&a, &b| {
        counts[b as usize]
            .cmp(&counts[a as usize])
            .then_with(|| a.cmp(&b))
    });
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;

    fn tail() -> u64 {
        tail_pad()
    }

    fn slabs_with(f: impl Fn(usize) -> (ExpertFormat, usize, usize)) -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..TOTAL_EXPERTS)
            .map(|i| {
                let (format, gu, dn) = f(i);
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: gu,
                };
                cursor += gu;
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: dn,
                };
                cursor += dn;
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    fn uniform() -> Vec<ExpertSlab> {
        slabs_with(|_| (ExpertFormat::Mq4Xts, GU, DN))
    }

    #[test]
    fn budget_zero_is_all_host() {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[0, 1, 2], 0).unwrap();
        assert_eq!(plan.device_experts(), 0);
        assert_eq!(plan.owner_bytes(Placement::Device), 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        for layer in 0..48 {
            for expert in 0..512 {
                assert_eq!(
                    plan.entry(layer, expert).unwrap().tier,
                    Placement::HostMapped
                );
            }
        }
        assert_eq!(plan.copies(Placement::Device).count(), 0);
    }

    #[test]
    fn tier_stage_bytes_is_the_largest_host_payload_per_parity_plus_pad() {
        use crate::expert_residency::expert_tier_stage_bytes;
        let pad = crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64;
        // Nothing in VRAM: both parity buffers hold a full host layer.
        let all_host = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        assert_eq!(expert_tier_stage_bytes(&all_host), 2 * (512 * PADDED + pad));
        // Experts 0..256 of every layer resident: each buffer holds 256 host experts.
        let rank: Vec<u32> = (0..48u32).flat_map(|layer| (0..256).map(move |e| layer * 512 + e)).collect();
        let half = plan_expert_tiers_from_slabs(&uniform(), &rank, u64::MAX).unwrap();
        assert_eq!(expert_tier_stage_bytes(&half), 2 * (256 * PADDED + pad));
        // Layer 1 additionally all resident: layer 3 still sizes the odd buffer.
        let mut more = rank.clone();
        more.extend((256..512).map(|e| 512 + e));
        let more = plan_expert_tiers_from_slabs(&uniform(), &more, u64::MAX).unwrap();
        assert_eq!(expert_tier_stage_bytes(&more), 2 * (256 * PADDED + pad));
    }

    #[test]
    fn exact_fit_and_one_byte_short() {
        let rank = [10u32, 5, 700, 3];
        let n = rank.len() as u64;
        let exact = n * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, exact).unwrap();
        assert_eq!(plan.device_experts(), 4);
        assert_eq!(plan.owner_bytes(Placement::Device), exact);

        let short = plan_expert_tiers_from_slabs(&uniform(), &rank, exact - 1).unwrap();
        assert_eq!(short.device_experts(), 3);
        assert_eq!(
            short.owner_bytes(Placement::Device),
            (n - 1) * PADDED + tail()
        );
        for &id in &rank[..3] {
            let (l, e) = (id as usize / 512, id as usize % 512);
            assert_eq!(short.entry(l, e).unwrap().tier, Placement::Device);
        }
        let last = rank[3] as usize;
        assert_eq!(
            short.entry(last / 512, last % 512).unwrap().tier,
            Placement::HostMapped
        );
    }

    #[test]
    fn rank_prefix_stops_at_first_non_fit() {
        // id 5 is huge; id 6 is small and would fit, but must not be admitted.
        let slabs = slabs_with(|i| {
            if i == 5 {
                (ExpertFormat::Mq4Xts, 100_000, 100_000)
            } else {
                (ExpertFormat::Mq4Xts, GU, DN)
            }
        });
        let budget = 2 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&slabs, &[0, 5, 6], budget).unwrap();
        assert_eq!(plan.device_experts(), 1);
        assert_eq!(plan.entry(0, 0).unwrap().tier, Placement::Device);
        assert_eq!(plan.entry(0, 5).unwrap().tier, Placement::HostMapped);
        assert_eq!(plan.entry(0, 6).unwrap().tier, Placement::HostMapped);
    }

    #[test]
    fn alignment_padding_with_odd_lengths() {
        let slabs = slabs_with(|_| (ExpertFormat::Mq4Xts, 1001, 3));
        let plan = plan_expert_tiers_from_slabs(&slabs, &[0, 1], 1 << 20).unwrap();
        // 1001 -> 1024, 3 -> 256.
        assert_eq!(plan.owner_bytes(Placement::Device), 2 * 1280 + tail());
        let e0 = plan.entry(0, 0).unwrap();
        let e1 = plan.entry(0, 1).unwrap();
        assert_eq!((e0.gate_up_offset, e0.down_offset), (0, 1024));
        assert_eq!((e1.gate_up_offset, e1.down_offset), (1280, 1280 + 1024));
        assert_eq!((e1.gate_up_len, e1.down_len), (1001, 3));
        // Host tier starts at its own base.
        let h = plan.entry(0, 2).unwrap();
        assert_eq!(h.tier, Placement::HostMapped);
        assert_eq!((h.gate_up_offset, h.down_offset), (0, 1024));
        for entry in [e0, e1, h] {
            assert_eq!(entry.gate_up_offset % 256, 0);
            assert_eq!(entry.down_offset % 256, 0);
        }
    }

    #[test]
    fn host_owner_beyond_4_gib_keeps_u64_offsets() {
        // Synthetic spans: nothing is allocated, only numbers.
        const L: usize = 1 << 28; // 256 MiB per projection
        let slabs = slabs_with(|_| (ExpertFormat::Mq4Xts, L, L));
        let plan = plan_expert_tiers_from_slabs(&slabs, &[], 0).unwrap();
        let per = 2 * L as u64;
        let last = plan.entry(47, 511).unwrap();
        let idx = (TOTAL_EXPERTS - 1) as u64;
        assert_eq!(last.gate_up_offset, idx * per);
        assert_eq!(last.down_offset, idx * per + L as u64);
        assert!(last.gate_up_offset > u64::from(u32::MAX));
        assert!(last.down_offset > u64::from(u32::MAX));
        assert_eq!(
            plan.owner_bytes(Placement::HostMapped),
            TOTAL_EXPERTS as u64 * per + tail()
        );
        let first_over = plan
            .copies(Placement::HostMapped)
            .find(|c| c.dst_offset > u64::from(u32::MAX))
            .unwrap();
        assert_eq!(first_over.dst_offset % 256, 0);
    }

    #[test]
    fn invalid_rank_is_rejected() {
        let slabs = uniform();
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[1, 24_576], 1 << 20),
            Err(ExpertTierError::RankIdOutOfRange { id: 24_576 })
        );
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[1, 2, 1], 1 << 20),
            Err(ExpertTierError::RankDuplicate { id: 1 })
        );
        // Validation covers ids past the budget stop point too.
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[4, 4], 0),
            Err(ExpertTierError::RankDuplicate { id: 4 })
        );
    }

    #[test]
    fn slab_count_and_source_spans_are_validated() {
        let mut slabs = uniform();
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs[..10], &[], 0),
            Err(ExpertTierError::SlabCount { got: 10 })
        );
        slabs.push(slabs[0]);
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[], 0),
            Err(ExpertTierError::SlabCount { got: TOTAL_EXPERTS + 1 })
        );
        slabs.pop();

        let mut empty = slabs.clone();
        empty[513].down.len = 0;
        assert_eq!(
            plan_expert_tiers_from_slabs(&empty, &[], 0),
            Err(ExpertTierError::SourceSpan { layer: 1, expert: 1 })
        );

        let mut overlap = slabs.clone();
        overlap[7].down.offset = overlap[7].gate_up.offset + 1;
        assert_eq!(
            plan_expert_tiers_from_slabs(&overlap, &[], 0),
            Err(ExpertTierError::SourceSpan { layer: 0, expert: 7 })
        );

        let mut wrap = slabs.clone();
        wrap[3].gate_up.offset = usize::MAX - 5;
        assert_eq!(
            plan_expert_tiers_from_slabs(&wrap, &[], 0),
            Err(ExpertTierError::SourceSpan { layer: 0, expert: 3 })
        );
    }

    #[test]
    fn u64_overflow_is_reported() {
        let mut slabs = uniform();
        slabs[0].gate_up = ExpertSlabSpan {
            offset: 0,
            len: usize::MAX - 10,
        };
        slabs[0].down = ExpertSlabSpan {
            offset: usize::MAX - 5,
            len: 1,
        };
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[], 0),
            Err(ExpertTierError::Overflow)
        );
    }

    #[test]
    fn layout_is_layer_expert_order_regardless_of_rank_order() {
        let budget = 3 * PADDED + tail();
        let a = plan_expert_tiers_from_slabs(&uniform(), &[600, 2, 100], budget).unwrap();
        let b = plan_expert_tiers_from_slabs(&uniform(), &[2, 100, 600], budget).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.logical_digest(), b.logical_digest());
        // Ascending model-id order inside the device owner.
        assert_eq!(a.entry(0, 2).unwrap().gate_up_offset, 0);
        assert_eq!(a.entry(0, 100).unwrap().gate_up_offset, PADDED);
        assert_eq!(a.entry(1, 88).unwrap().gate_up_offset, 2 * PADDED);
        let order: Vec<_> = a
            .copies(Placement::Device)
            .map(|c| (c.layer, c.expert, c.proj))
            .collect();
        assert_eq!(
            order,
            vec![
                (0, 2, TierProj::GateUp),
                (0, 2, TierProj::Down),
                (0, 100, TierProj::GateUp),
                (0, 100, TierProj::Down),
                (1, 88, TierProj::GateUp),
                (1, 88, TierProj::Down),
            ]
        );
    }

    #[test]
    fn digest_covers_placement_format_source_artifact_and_window() {
        let budget = 2 * PADDED + tail();
        let base = plan_expert_tiers_from_slabs(&uniform(), &[0, 1], budget).unwrap();
        let d0 = base.logical_digest();
        assert_eq!(
            d0,
            plan_expert_tiers_from_slabs(&uniform(), &[0, 1], budget)
                .unwrap()
                .logical_digest()
        );

        // One placement bit.
        let placement = plan_expert_tiers_from_slabs(&uniform(), &[0, 2], budget).unwrap();
        assert_ne!(d0, placement.logical_digest());

        // One format (same lengths).
        let mut fmt_slabs = uniform();
        fmt_slabs[9].format = ExpertFormat::Mq3I4;
        let fmt = plan_expert_tiers_from_slabs(&fmt_slabs, &[0, 1], budget).unwrap();
        assert_ne!(d0, fmt.logical_digest());

        // One source span.
        let mut src_slabs = uniform();
        src_slabs[20].gate_up.offset -= 1;
        let src = plan_expert_tiers_from_slabs(&src_slabs, &[0, 1], budget).unwrap();
        assert_ne!(d0, src.logical_digest());

        let sha = base.clone().with_artifact_sha256([7u8; 32]);
        assert_ne!(d0, sha.logical_digest());
        assert_ne!(
            sha.logical_digest(),
            base.clone().with_artifact_sha256([8u8; 32]).logical_digest()
        );

        let win = base
            .clone()
            .with_host_window("mtp", &[ExpertSlabSpan { offset: 1 << 40, len: 100 }])
            .unwrap();
        assert_ne!(d0, win.logical_digest());
    }

    #[test]
    fn mixed_formats_are_preserved_and_check_executable_gates_by_arch() {
        let slabs = slabs_with(|i| match i % 3 {
            0 => (ExpertFormat::Mq4Xts, 1000, 500),
            1 => (ExpertFormat::Mq3I4, 800, 400),
            _ => (ExpertFormat::Mq2I4, 600, 300),
        });
        let plan = plan_expert_tiers_from_slabs(&slabs, &[1, 2, 3], 1 << 20).unwrap();
        for (i, slab) in slabs.iter().enumerate() {
            let e = plan.entry(i / 512, i % 512).unwrap();
            assert_eq!(e.format, slab.format);
            assert_eq!((e.gate_up_len, e.down_len), (slab.gate_up.len as u64, slab.down.len as u64));
            assert_eq!(plan.source(i / 512, i % 512), Some((slab.gate_up, slab.down)));
        }
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            assert!(matches!(
                plan.check_executable(arch),
                Err(ExpertTierError::UnsupportedFormat {
                    format: ExpertFormat::Mq3I4 | ExpertFormat::Mq2I4,
                    ..
                })
            ));
        }
        assert_eq!(
            plan.check_executable("gfx1201"),
            Err(ExpertTierError::UnsupportedFormat {
                format: ExpertFormat::Mq2I4,
                arch: "gfx1201".into()
            })
        );

        let w3 = slabs_with(|i| {
            if i % 2 == 0 {
                (ExpertFormat::Mq4Xts, GU, DN)
            } else {
                (ExpertFormat::Mq3I4, GU, DN)
            }
        });
        let plan = plan_expert_tiers_from_slabs(&w3, &[], 0).unwrap();
        assert_eq!(plan.check_executable("gfx1201"), Ok(()));
        for arch in ["gfx1100", "gfx1151"] {
            assert_eq!(
                plan.check_executable(arch),
                Err(ExpertTierError::UnsupportedFormat {
                    format: ExpertFormat::Mq3I4,
                    arch: arch.into()
                })
            );
        }

        let mq4 = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            assert_eq!(mq4.check_executable(arch), Ok(()));
        }
    }

    #[test]
    fn tail_pad_follows_the_last_copy_in_every_owner() {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[0, 1, 2], 1 << 20).unwrap();
        assert!(tail() >= ARENA_TAIL_PAD_BYTES as u64);
        assert_eq!(tail() % 256, 0);
        for tier in [Placement::Device, Placement::HostMapped] {
            let last_end = plan
                .copies(tier)
                .map(|c| c.dst_offset + c.src.len as u64)
                .max()
                .unwrap();
            assert!(
                plan.owner_bytes(tier) >= last_end + ARENA_TAIL_PAD_BYTES as u64,
                "{tier:?}"
            );
            assert_eq!(plan.owner_bytes(tier) % 256, 0);
        }
    }

    #[test]
    fn copy_census_is_exact_and_non_overlapping() {
        let rank: Vec<u32> = (0..24_576).step_by(7).collect();
        let budget = 1000 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), 1000);

        let mut seen = vec![0u8; TOTAL_EXPERTS * 2];
        for tier in [Placement::Device, Placement::HostMapped] {
            let bound = plan.owner_bytes(tier) - tail();
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for c in plan.copies(tier) {
                let slot = (c.layer * 512 + c.expert) * 2 + usize::from(c.proj == TierProj::Down);
                seen[slot] += 1;
                assert_eq!(plan.entry(c.layer, c.expert).unwrap().tier, tier);
                ranges.push((c.dst_offset, c.dst_offset + c.src.len as u64));
            }
            ranges.sort_unstable();
            for w in ranges.windows(2) {
                assert!(w[0].1 <= w[1].0, "overlap {w:?}");
            }
            assert!(ranges.last().unwrap().1 <= bound);
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn host_window_appends_after_trunk_host_end_with_single_tail() {
        let base = plan_expert_tiers_from_slabs(&uniform(), &[0], PADDED + tail()).unwrap();
        let host_before = base.owner_bytes(Placement::HostMapped);
        let host_end = host_before - tail();
        let spans = [
            ExpertSlabSpan { offset: 5_000_000_000, len: 1000 },
            ExpertSlabSpan { offset: 6_000_000_000, len: 513 },
        ];
        let plan = base.clone().with_host_window("mtp", &spans).unwrap();
        let w = &plan.host_windows()[0];
        assert_eq!(w.label, "mtp");
        assert_eq!(w.spans[0], (spans[0], host_end));
        assert_eq!(w.spans[1], (spans[1], host_end + 1024));
        assert_eq!(
            plan.owner_bytes(Placement::HostMapped),
            host_end + 1024 + 768 + tail()
        );
        assert_eq!(
            plan.owner_bytes(Placement::Device),
            base.owner_bytes(Placement::Device)
        );

        // Window into an otherwise empty host tier allocates it, tail once.
        let all_dev = plan_expert_tiers_from_slabs(&uniform(), &(0..24_576).collect::<Vec<_>>(), u64::MAX)
            .unwrap();
        assert_eq!(all_dev.owner_bytes(Placement::HostMapped), 0);
        let w = all_dev.with_host_window("mtp", &spans[..1]).unwrap();
        assert_eq!(w.owner_bytes(Placement::HostMapped), 1024 + tail());
        assert_eq!(w.host_windows()[0].spans[0].1, 0);

        assert_eq!(
            base.clone()
                .with_host_window("x", &[ExpertSlabSpan { offset: 0, len: 0 }]),
            Err(ExpertTierError::Overflow)
        );
        assert_eq!(
            base.with_host_window("x", &[ExpertSlabSpan { offset: usize::MAX, len: 2 }]),
            Err(ExpertTierError::Overflow)
        );
    }

    #[test]
    fn rank_from_counts_breaks_ties_by_id_and_filters() {
        let counts = [5u64, 9, 5, 0, 9, 5];
        assert_eq!(rank_from_counts(&counts, |_| true), vec![1, 4, 0, 2, 5, 3]);
        assert_eq!(
            rank_from_counts(&counts, |id| id % 2 == 0),
            vec![4, 0, 2]
        );
        assert!(rank_from_counts(&counts, |_| false).is_empty());
        assert!(rank_from_counts(&[], |_| true).is_empty());
    }

    #[test]
    fn error_display_is_informative() {
        let e = ExpertTierError::UnsupportedFormat {
            format: ExpertFormat::Mq2I4,
            arch: "gfx1100".into(),
        };
        assert!(e.to_string().contains("gfx1100"));
        let _: &dyn std::error::Error = &e;
        assert_eq!(CODEC_W2, 2);
    }
}
