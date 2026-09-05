// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 hybrid-state checkpoint pool for serving prefix-cache resume
//! (spec §4.5 C5).
//!
//! [`QwenCheckpointPool`] is a byte-bounded LRU of immutable captured
//! state bundles. Each entry is keyed by `(`[`CacheDomain`]`, boundary_p)`
//! where `p` is the number of tokens `[0, p)` the target has processed.
//! The value holds a [`DeltaNetSnapshot`] covering all of: DN matrices,
//! scales, convolution rings/indices, and error-feedback residuals.
//!
//! # Capture alignment
//!
//! Checkpoints are captured only at page-aligned completed boundaries
//! (`p % [`PAGE_TOKENS`] == 0` or `p == 0`). A state at the end of a chunk
//! cannot be relabelled as an earlier state (spec §4.5).
//!
//! # Private restore
//!
//! Running requests restore into private mutable buffers; the pool never
//! shares a mutable [`DeltaNetState`](crate::qwen35::DeltaNetState) between
//! requests. [`restore_private`] copies an immutable cached snapshot into a
//! caller-owned private snapshot via device-to-device memcpy.
//!
//! # Checkpoint ids
//!
//! [`CheckpointId`] is a monotonic id minted by this pool, starting at 1.
//! [`CheckpointId::NONE`] (value 0) means "no checkpoint." The radix index
//! (P2-index) stores only the id + boundary; this pool owns the bytes. The
//! id namespace is private to this crate — ids are not stable across pool
//! restarts.

use crate::speculative::DeltaNetSnapshot;
use hipfire_runtime::serve_contract::{
    CacheDomain, DrafterDecision, LastTokenHandling, MissReason, PrefixLookup, ResumeBundle,
    ResumePlan, ResumePlanError,
};
use rdna_compute::page_pool::PAGE_TOKENS;
use std::collections::{HashMap, HashSet};

// ───────────────────────────────────────────────────────────────────────────
// Checkpoint id — re-exported from hipfire_runtime::serve_contract
// ───────────────────────────────────────────────────────────────────────────

pub use hipfire_runtime::serve_contract::CheckpointId;

// ───────────────────────────────────────────────────────────────────────────
// CheckpointBlob trait
// ───────────────────────────────────────────────────────────────────────────

/// Abstraction over the stored checkpoint bytes, used so the pool can be
/// tested on the host without GPU device buffers (spec §4.5 C5).
///
/// For GPU use, `DeltaNetSnapshot` implements this via its `bytes_len()`
/// method. For host tests, a simple byte-counting test double suffices.
pub trait CheckpointBlob {
    /// Total device/host bytes this blob occupies, for LRU accounting.
    fn bytes_len(&self) -> u64;
}

impl CheckpointBlob for DeltaNetSnapshot {
    fn bytes_len(&self) -> u64 {
        DeltaNetSnapshot::bytes_len(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Pool entry
// ───────────────────────────────────────────────────────────────────────────

/// Key for a checkpoint: `(domain, boundary_p)`.
type CheckpointKey = (CacheDomain, u64);

#[derive(Debug)]
struct CheckpointEntry<B> {
    id: CheckpointId,
    blob: B,
    pinned: bool,
    /// Monotonic LRU access stamp; smaller = older.
    lru_stamp: u64,
}

// ───────────────────────────────────────────────────────────────────────────
// QwenCheckpointPool
// ───────────────────────────────────────────────────────────────────────────

/// Byte-bounded LRU pool of immutable Qwen3.5 hybrid-state checkpoint
/// bundles (spec §4.5 C5).
///
/// Keyed by `(CacheDomain, boundary_p)`. Entries are captured only at
/// page-aligned boundaries. When the pool exceeds `max_bytes`, the oldest
/// **unpinned** checkpoint is evicted until the pool fits. Pinned
/// checkpoints survive eviction.
///
/// Generic over `B: CheckpointBlob` so host tests can use a byte-counting
/// test double without GPU device buffers. The GPU-backed capture and
/// restore paths use `B = DeltaNetSnapshot`.
pub struct QwenCheckpointPool<B: CheckpointBlob> {
    entries: HashMap<CheckpointKey, CheckpointEntry<B>>,
    /// Keys that were explicitly evicted (for distinguishing
    /// [`MissReason::Evicted`] from [`MissReason::NoCheckpoint`]).
    evicted: HashSet<CheckpointKey>,
    total_bytes: u64,
    max_bytes: u64,
    next_id: u64,
    lru_clock: u64,
}

impl<B: CheckpointBlob> QwenCheckpointPool<B> {
    /// Create a pool with a byte capacity of `max_bytes`.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            evicted: HashSet::new(),
            total_bytes: 0,
            max_bytes,
            next_id: 1,
            lru_clock: 0,
        }
    }

    /// Maximum byte capacity.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Current total bytes across all entries.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Number of entries currently in the pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify `p` is a valid capture boundary: `p == 0` or `p` is a
    /// multiple of [`PAGE_TOKENS`].
    fn is_aligned(p: u64) -> bool {
        p == 0 || p % PAGE_TOKENS as u64 == 0
    }

    /// Insert (or replace) a captured checkpoint blob at `(domain, p)`.
    ///
    /// `p` must be page-aligned (`p % 128 == 0` or `p == 0`); otherwise the
    /// entry is **not** inserted and [`CheckpointId::NONE`] is returned
    /// (spec §4.5: "a state at the end of a chunk cannot be relabelled as
    /// an earlier state").
    ///
    /// If the pool cannot afford the new capture, the oldest **unpinned**
    /// checkpoint is evicted repeatedly until the pool fits or no unpinned
    /// entries remain (spec §4.5: "drop the oldest unpinned checkpoint").
    ///
    /// Returns the minted [`CheckpointId`] for the radix index to store.
    pub fn insert(&mut self, domain: CacheDomain, p: u64, blob: B) -> CheckpointId {
        if !Self::is_aligned(p) {
            return CheckpointId::NONE;
        }

        let bytes = blob.bytes_len();
        let key = (domain, p);

        // If an entry already exists at this key, replace it.
        if let Some(old) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.blob.bytes_len());
            // Keep the same id for stability across re-capture.
        }

        // Clear any prior eviction record for this key.
        self.evicted.remove(&key);

        // Evict oldest unpinned until we can afford the new blob.
        while self.total_bytes + bytes > self.max_bytes {
            match self.find_oldest_unpinned_key() {
                Some(evict_key) => {
                    self.evict_internal(evict_key);
                }
                None => break, // all remaining are pinned; insert anyway
            }
        }

        let id = CheckpointId(self.next_id);
        self.next_id += 1;
        self.lru_clock += 1;
        self.total_bytes += bytes;
        self.entries.insert(
            key,
            CheckpointEntry {
                id,
                blob,
                pinned: false,
                lru_stamp: self.lru_clock,
            },
        );

        id
    }

    /// Find the key of the oldest (smallest `lru_stamp`) unpinned entry.
    fn find_oldest_unpinned_key(&self) -> Option<CheckpointKey> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.pinned)
            .min_by_key(|(_, e)| e.lru_stamp)
            .map(|(k, _)| k.clone())
    }

    /// Remove an entry by key, accounting bytes and recording eviction.
    fn evict_internal(&mut self, key: CheckpointKey) {
        if let Some(entry) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.blob.bytes_len());
            self.evicted.insert(key);
        }
    }

    /// Explicitly evict the checkpoint at `(domain, p)`.
    ///
    /// Returns `true` if an entry was removed.
    pub fn evict(&mut self, domain: &CacheDomain, p: u64) -> bool {
        let key = (domain.clone(), p);
        if self.entries.contains_key(&key) {
            self.evict_internal(key);
            true
        } else {
            false
        }
    }

    /// Check whether a checkpoint exists at `(domain, p)`.
    pub fn contains(&self, domain: &CacheDomain, p: u64) -> bool {
        self.entries.contains_key(&(domain.clone(), p))
    }

    /// Get the [`CheckpointId`] for `(domain, p)`, if present.
    pub fn id_of(&self, domain: &CacheDomain, p: u64) -> Option<CheckpointId> {
        self.entries.get(&(domain.clone(), p)).map(|e| e.id)
    }

    /// Borrow the blob at `(domain, p)`, refreshing its LRU stamp.
    pub fn get(&mut self, domain: &CacheDomain, p: u64) -> Option<&B> {
        let key = (domain.clone(), p);
        if let Some(entry) = self.entries.get_mut(&key) {
            self.lru_clock += 1;
            entry.lru_stamp = self.lru_clock;
            Some(&entry.blob)
        } else {
            None
        }
    }

    /// Borrow the blob at `(domain, p)` without refreshing LRU (read-only).
    pub fn peek(&self, domain: &CacheDomain, p: u64) -> Option<&B> {
        self.entries.get(&(domain.clone(), p)).map(|e| &e.blob)
    }

    /// Pin the checkpoint at `(domain, p)` so it survives LRU eviction.
    ///
    /// Returns `true` if the entry was found and pinned.
    pub fn pin(&mut self, domain: &CacheDomain, p: u64) -> bool {
        let key = (domain.clone(), p);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpin the checkpoint at `(domain, p)`, making it eligible for LRU
    /// eviction again.
    ///
    /// Returns `true` if the entry was found and unpinned.
    pub fn unpin(&mut self, domain: &CacheDomain, p: u64) -> bool {
        let key = (domain.clone(), p);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pinned = false;
            true
        } else {
            false
        }
    }

    /// Whether the checkpoint at `(domain, p)` is pinned.
    pub fn is_pinned(&self, domain: &CacheDomain, p: u64) -> bool {
        self.entries
            .get(&(domain.clone(), p))
            .map(|e| e.pinned)
            .unwrap_or(false)
    }

    /// Drain and return all stored blobs, clearing the pool. Used by the
    /// serve engine's `free_gpu` to explicitly free each `DeltaNetSnapshot`'s
    /// device buffers on shutdown (the pool itself has no `Drop` impl that
    /// touches the GPU).
    pub fn drain_blobs(&mut self) -> Vec<B> {
        self.total_bytes = 0;
        self.entries.drain().map(|(_, e)| e.blob).collect()
    }

    /// Check whether a checkpoint at any page-aligned boundary `≤ max_p`
    /// for `domain` was previously evicted (for [`MissReason::Evicted`]
    /// reporting).
    fn was_evicted(&self, domain: &CacheDomain, max_p: u64) -> bool {
        self.evicted
            .iter()
            .any(|(d, p)| d == domain && *p <= max_p && *p > 0)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// plan_resume
// ───────────────────────────────────────────────────────────────────────────

/// Build a [`ResumeBundle`] with all required component flags true.
///
/// When a checkpoint exists in the pool at boundary `p`, the snapshot
/// covers DN matrices, scales, conv rings/indices, and EF residuals.
/// `attention_pages` is true because `lookup.resumable_tokens >= p`
/// (the index guarantees all state components exist at that boundary).
fn complete_bundle(drafter: DrafterDecision) -> ResumeBundle {
    ResumeBundle {
        attention_pages: true,
        dn_matrices_scales: true,
        conv_rings: true,
        ef_residual: true,
        drafter,
    }
}

/// Find the largest page-aligned checkpoint boundary `p > 0` with
/// `p <= max_p` in the pool for `domain`.
///
/// Does NOT return `p = 0` — the initial state does not require a
/// checkpoint and is handled separately by the caller.
fn find_largest_pool_checkpoint<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    max_p: u64,
) -> Option<u64> {
    let page = PAGE_TOKENS as u64;
    let mut p = (max_p / page) * page;
    while p > 0 {
        if pool.contains(domain, p) {
            return Some(p);
        }
        p -= page;
    }
    None
}

/// Find the largest page-aligned checkpoint boundary `p > 0` with
/// `p < below_p` in the pool for `domain`.
fn find_largest_pool_checkpoint_below<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    below_p: u64,
) -> Option<u64> {
    let page = PAGE_TOKENS as u64;
    if below_p <= page {
        return None;
    }
    let mut p = below_p - page;
    while p > 0 {
        if pool.contains(domain, p) {
            return Some(p);
        }
        p -= page;
    }
    None
}

/// Plan a resume from the checkpoint pool (spec §4.5 C5).
///
/// Chooses the largest `p ≤ lookup.resumable_tokens` that is page-aligned
/// and has a complete Qwen bundle in the pool. By default uses
/// [`LastTokenHandling::SuffixRecompute`] with `p < prompt_len`.
///
/// # Last-token semantics
///
/// If `prompt_len == p` (the prompt exactly matches the cached boundary),
/// an **earlier** valid checkpoint or the initial state (`p = 0`) is
/// selected with [`LastTokenHandling::EarlierBoundary`] — never restore
/// `S_prompt_len` and re-run the last token (spec §4.5). An empty prompt
/// (`prompt_len == 0`) cannot underflow: `p = 0` is returned directly.
///
/// # Drafter decision
///
/// `drafter` is an input from admission. Reusing the target prefix is not
/// evidence that the drafter is ready; the caller decides
/// [`DrafterDecision::Checkpoint`] only when a separately identity-qualified
/// drafter checkpoint exists (spec §4.5).
///
/// # Errors
///
/// Returns [`MissReason::NoCheckpoint`] if no checkpoint exists at any
/// page-aligned boundary `≤ resumable_tokens` (and `resumable_tokens > 0`).
/// Returns [`MissReason::Evicted`] if a checkpoint was previously resident
/// but has been evicted.
///
/// # Materialized boundary
///
/// The returned `ResumePlan.boundary` is the **materialized committed
/// prefix** — tokens `[0, p)` processed by the target. It does NOT include
/// the last sampled token, which may not yet have a KV row (spec §4.5,
/// §6.1/X1). When constructing a [`hipfire_runtime::serve_contract::CommitBoundary`],
/// `committed_tokens` and `materialized_rows` must reflect only this
/// processed prefix, not the accepted token history.
pub fn plan_resume<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    prompt_len: u64,
    lookup: &PrefixLookup,
    drafter: DrafterDecision,
) -> Result<ResumePlan, MissReason> {
    let resumable = lookup.resumable_tokens;

    // Find the largest page-aligned checkpoint p > 0, p <= resumable.
    let best_p = find_largest_pool_checkpoint(pool, domain, resumable);

    match best_p {
        Some(p) if prompt_len == p && prompt_len > 0 => {
            // Exact match: select an earlier boundary, never restore S_prompt_len.
            // Prefer an earlier checkpoint; fall back to the initial state p=0.
            let earlier = find_largest_pool_checkpoint_below(pool, domain, p);
            let ep = earlier.unwrap_or(0);
            let byte_cost = if ep == 0 {
                0
            } else {
                pool.peek(domain, ep).map(|b| b.bytes_len()).unwrap_or(0)
            };
            let bundle = complete_bundle(drafter);
            ResumePlan::new(ep, bundle, byte_cost, LastTokenHandling::EarlierBoundary)
                .map_err(|e| match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                })
        }
        Some(p) => {
            // Normal: p < prompt_len (or prompt_len == 0 with p > 0 — shouldn't
            // normally happen, but SuffixRecompute is still safe).
            let byte_cost = pool.peek(domain, p).map(|b| b.bytes_len()).unwrap_or(0);
            let bundle = complete_bundle(drafter);
            ResumePlan::new(p, bundle, byte_cost, LastTokenHandling::SuffixRecompute)
                .map_err(|e| match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                })
        }
        None => {
            // No checkpoint at any page-aligned boundary <= resumable.
            if resumable == 0 && prompt_len == 0 {
                // Empty prompt: p=0, no underflow (spec §4.5).
                let bundle = complete_bundle(drafter);
                ResumePlan::new(0, bundle, 0, LastTokenHandling::SuffixRecompute)
                    .map_err(|e| match e {
                        ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                    })
            } else if pool.was_evicted(domain, resumable) {
                Err(MissReason::Evicted)
            } else {
                Err(MissReason::NoCheckpoint)
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// GPU-backed capture and restore (require deltanet + a Gpu)
// ───────────────────────────────────────────────────────────────────────────

use crate::qwen35::DeltaNetState;
use hip_bridge::{HipError, HipResult};
use rdna_compute::Gpu;

/// Capture a checkpoint from the live `DeltaNetState` at boundary `p`
/// and insert it into `pool` (spec §4.5 C5).
///
/// `p` must be page-aligned (`p % 128 == 0` or `p == 0`). Allocates a
/// fresh `DeltaNetSnapshot` via [`DeltaNetSnapshot::new_for`], copies the
/// live state into it via [`DeltaNetSnapshot::save_from`], and inserts it
/// into the pool. The pool evicts oldest unpinned entries as needed.
///
/// Returns the minted [`CheckpointId`], or an error if snapshot allocation
/// or the device-to-device copy fails.
///
/// **P2-wire will call this** from the serve engine's commit/prefill
/// path at page-aligned completed boundaries.
pub fn capture_checkpoint(
    gpu: &mut Gpu,
    pool: &mut QwenCheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    state: &DeltaNetState,
) -> HipResult<CheckpointId> {
    if !QwenCheckpointPool::<DeltaNetSnapshot>::is_aligned(p) {
        return Err(HipError::new(0, "capture_checkpoint: boundary not page-aligned"));
    }

    let mut snap = DeltaNetSnapshot::new_for(gpu, state)?;
    snap.save_from(state, gpu)?;

    Ok(pool.insert(domain.clone(), p, snap))
}

/// Restore an immutable cached checkpoint into a caller-owned **private**
/// `DeltaNetSnapshot` via device-to-device copy (spec §4.5 C5).
///
/// The pool's snapshot is never mutated; `dst` receives a private copy.
/// `dst` must have been pre-allocated with matching shapes (e.g. via
/// [`DeltaNetSnapshot::new_for`] against the same model state).
///
/// Returns an error if the checkpoint is not found or the copy fails.
///
/// **P2-wire will call this** when executing a [`ResumePlan`] to obtain
/// private recurrent state for a running request.
pub fn restore_private(
    gpu: &mut Gpu,
    pool: &mut QwenCheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    dst: &mut DeltaNetSnapshot,
) -> HipResult<()> {
    let src = pool
        .peek(domain, p)
        .ok_or_else(|| HipError::new(0, "restore_private: checkpoint not found"))?;
    src.copy_to(dst, gpu)
}

// ───────────────────────────────────────────────────────────────────────────
// Tests (host-only — no GPU/HIP required)
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity,
        TokenizerIdentity,
    };

    /// Host test double: just byte-length accounting, no device buffers.
    #[derive(Clone, Debug)]
    struct HostBlob {
        bytes: u64,
    }

    impl CheckpointBlob for HostBlob {
        fn bytes_len(&self) -> u64 {
            self.bytes
        }
    }

    /// Build a minimal `CacheDomain` for testing.
    fn test_domain(tag: &str) -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![0u8; 32],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![1u8; 16],
                config_digest: vec![2u8; 16],
            },
            template: TemplateIdentity {
                template_digest: vec![3u8; 16],
                normalization_tag: "default".to_string(),
            },
            arch_policy: ArchPolicy {
                arch_tag: tag.to_string(),
                state_abi_tag: "q8".to_string(),
                position_attention_tag: "causal".to_string(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![128],
                v_stride_bytes: vec![128],
                layout_tag: "q8".to_string(),
            },
            device: DeviceTopology {
                device_id: "gpu0".to_string(),
                topology_id: "single".to_string(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "test".to_string(),
            },
        }
    }

    /// Build a `PrefixLookup` with the given `resumable_tokens`.
    fn lookup(resumable: u64) -> PrefixLookup {
        PrefixLookup {
            matched_tokens: resumable,
            resident_kv_tokens: resumable,
            resumable_tokens: resumable,
        }
    }

    const PAGE: u64 = PAGE_TOKENS as u64; // 128

    // ── A7: structural — capture at p=128, plan 200-token prompt ────────

    #[test]
    fn a7_capture_at_128_plan_200() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7");

        // Capture at p=128 (page-aligned).
        let id = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert_ne!(id, CheckpointId::NONE, "insert should mint a nonzero id");

        // Plan for a 200-token prompt: p=128 < 200 → SuffixRecompute.
        let plan = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.last_token, LastTokenHandling::SuffixRecompute);
        assert!(plan.bundle.attention_pages);
        assert!(plan.bundle.dn_matrices_scales);
        assert!(plan.bundle.conv_rings);
        assert!(plan.bundle.ef_residual);
        assert_eq!(plan.bundle.drafter, DrafterDecision::Ar);
    }

    // ── A7: exact-match prompt selects earlier boundary ─────────────────

    #[test]
    fn a7_exact_match_selects_earlier_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-exact");

        // Capture at p=128 only.
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Prompt of exactly 128 tokens: must NOT resume at p=128.
        // Should select p=0 (initial state) with EarlierBoundary.
        let plan = plan_resume(
            &pool,
            &dom,
            128,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_ne!(
            plan.boundary, 128,
            "must not restore S_prompt_len for an exact-match prompt"
        );
        assert_eq!(plan.boundary, 0);
        assert_eq!(plan.last_token, LastTokenHandling::EarlierBoundary);
    }

    // ── A7: exact match with two checkpoints selects previous page ──────

    #[test]
    fn a7_exact_match_with_prior_checkpoint() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-prior");

        // Capture at p=128 and p=256.
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        // Prompt of exactly 256 tokens: should select p=128, not p=256.
        let plan = plan_resume(
            &pool,
            &dom,
            256,
            &lookup(256),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.last_token, LastTokenHandling::EarlierBoundary);
    }

    // ── A7: empty prompt does not underflow ──────────────────────────────

    #[test]
    fn a7_empty_prompt_no_underflow() {
        let pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-empty");

        // No checkpoints, empty prompt, resumable=0.
        let plan = plan_resume(
            &pool,
            &dom,
            0,
            &lookup(0),
            DrafterDecision::Ar,
        )
        .expect("empty prompt should not underflow");

        assert_eq!(plan.boundary, 0);
        assert_eq!(plan.last_token, LastTokenHandling::SuffixRecompute);
    }

    // ── A8: missing checkpoint → NoCheckpoint error ─────────────────────

    #[test]
    fn a8_missing_checkpoint_is_no_checkpoint() {
        let pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-missing");

        // Lookup claims 128 resumable tokens, but no checkpoint in pool.
        let err = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("should be a miss");

        assert_eq!(err, MissReason::NoCheckpoint);
    }

    // ── A8: evicted checkpoint → Evicted error ──────────────────────────

    #[test]
    fn a8_evicted_checkpoint_is_evicted() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-evicted");

        // Insert at p=128, then evict it.
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert!(pool.evict(&dom, 128));

        // Lookup still claims 128 resumable, but checkpoint was evicted.
        let err = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("should be a miss");

        assert_eq!(err, MissReason::Evicted);
    }

    // ── A8: not a silent Hit ─────────────────────────────────────────────

    #[test]
    fn a8_missing_is_not_silent_hit() {
        let pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-silent");

        // resumable > 0 but no checkpoint → must error, not return a plan.
        let result = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(64),
            DrafterDecision::Ar,
        );

        assert!(result.is_err(), "must not be a silent Hit");
    }

    // ── A9: materialized boundary excludes uncomputed last token ────────

    #[test]
    fn a9_boundary_is_materialized_prefix() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a9");

        // Capture at p=128.
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Plan for 200-token prompt.
        let plan = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .unwrap();

        // The boundary is the materialized committed prefix (tokens [0,128)
        // processed by the target). It does NOT include the 129th token or
        // any sampled-but-not-yet-materialized token. The suffix [128,200)
        // will be processed to obtain first-token logits.
        assert_eq!(plan.boundary, 128);
        assert!(plan.boundary < 200, "boundary must be < prompt_len");
        // If a CommitBoundary were constructed from this plan,
        // committed_tokens = materialized_rows = plan.boundary = 128,
        // NOT 200 (the full prompt) or 129 (a sampled last token).
    }

    // ── A9: exact-match boundary is strictly less than prompt_len ───────

    #[test]
    fn a9_exact_match_boundary_strictly_less() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a9-exact");

        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        let plan = plan_resume(
            &pool,
            &dom,
            256, // exact match
            &lookup(256),
            DrafterDecision::Ar,
        )
        .unwrap();

        // For an exact-match prompt, the resume boundary must be strictly
        // less than prompt_len — the last sampled token is NOT included
        // in the materialized prefix.
        assert!(plan.boundary < 256);
        assert_eq!(plan.boundary, 128);
    }

    // ── LRU byte bound: inserting over capacity evicts oldest unpinned ──

    #[test]
    fn lru_evicts_oldest_unpinned() {
        // Capacity: 2 entries of 4096 bytes each.
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru");

        let id0 = pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        let id128 = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.total_bytes(), 8192);

        // Insert a third — should evict the oldest (p=0).
        let id256 = pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2, "should still have 2 entries after eviction");
        assert!(!pool.contains(&dom, 0), "oldest (p=0) should be evicted");
        assert!(pool.contains(&dom, 128));
        assert!(pool.contains(&dom, 256));
        assert_ne!(id0, CheckpointId::NONE);
        assert_ne!(id128, CheckpointId::NONE);
        assert_ne!(id256, CheckpointId::NONE);
    }

    // ── LRU byte bound: pinned checkpoints survive eviction ─────────────

    #[test]
    fn lru_pinned_survives_eviction() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru-pinned");

        pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Pin the oldest (p=0).
        assert!(pool.pin(&dom, 0));
        assert!(pool.is_pinned(&dom, 0));

        // Insert a third — p=128 (unpinned, newer) should be evicted,
        // NOT p=0 (pinned, older).
        pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        assert!(pool.contains(&dom, 0), "pinned p=0 must survive eviction");
        assert!(!pool.contains(&dom, 128), "unpinned p=128 should be evicted");
        assert!(pool.contains(&dom, 256));
    }

    // ── LRU: access refreshes recency ───────────────────────────────────

    #[test]
    fn lru_access_refreshes_recency() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru-recency");

        pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Access p=0 to make it more recent than p=128.
        let _ = pool.get(&dom, 0);

        // Insert a third — p=128 (now oldest) should be evicted.
        pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        assert!(pool.contains(&dom, 0), "recently accessed p=0 survives");
        assert!(!pool.contains(&dom, 128), "oldest p=128 evicted");
    }

    // ── Domain isolation: different domains don't share state ───────────

    #[test]
    fn domain_isolation() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom_a = test_domain("isolation-a");
        let dom_b = test_domain("isolation-b");

        pool.insert(dom_a.clone(), 128, HostBlob { bytes: 4096 });

        // dom_b has no checkpoint at p=128.
        assert!(!pool.contains(&dom_b, 128));
        assert!(pool.contains(&dom_a, 128));

        // Planning with dom_b should miss.
        let err = plan_resume(
            &pool,
            &dom_b,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("different domain should miss");

        assert_eq!(err, MissReason::NoCheckpoint);

        // Planning with dom_a should succeed.
        let plan = plan_resume(
            &pool,
            &dom_a,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("same domain should hit");

        assert_eq!(plan.boundary, 128);
    }

    // ── Domain isolation: eviction in one domain doesn't affect another ─

    #[test]
    fn domain_isolation_eviction() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom_a = test_domain("iso-evict-a");
        let dom_b = test_domain("iso-evict-b");

        pool.insert(dom_a.clone(), 128, HostBlob { bytes: 4096 });
        pool.insert(dom_b.clone(), 128, HostBlob { bytes: 4096 });

        // Inserting a third entry evicts the oldest unpinned (dom_a, 128).
        pool.insert(dom_a.clone(), 256, HostBlob { bytes: 4096 });

        assert!(!pool.contains(&dom_a, 128), "dom_a p=128 evicted");
        assert!(pool.contains(&dom_b, 128), "dom_b p=128 must survive");
    }

    // ── Page alignment: non-aligned boundary is rejected ────────────────

    #[test]
    fn non_aligned_boundary_rejected() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("align");

        let id = pool.insert(dom.clone(), 100, HostBlob { bytes: 4096 });
        assert_eq!(id, CheckpointId::NONE, "non-page-aligned boundary must be rejected");
        assert!(!pool.contains(&dom, 100));
        assert_eq!(pool.total_bytes(), 0);
    }

    // ── p=0 is a valid capture boundary ──────────────────────────────────

    #[test]
    fn p0_is_valid_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("p0");

        let id = pool.insert(dom.clone(), 0, HostBlob { bytes: 0 });
        assert_ne!(id, CheckpointId::NONE);
        assert!(pool.contains(&dom, 0));
    }

    // ── CheckpointId is monotonic ────────────────────────────────────────

    #[test]
    fn checkpoint_ids_are_monotonic() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("monotonic");

        let id1 = pool.insert(dom.clone(), 0, HostBlob { bytes: 100 });
        let id2 = pool.insert(dom.clone(), 128, HostBlob { bytes: 100 });
        let id3 = pool.insert(dom.clone(), 256, HostBlob { bytes: 100 });

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert_eq!(id1, CheckpointId(1));
    }

    // ── Byte accounting is accurate ──────────────────────────────────────

    #[test]
    fn byte_accounting() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("bytes");

        pool.insert(dom.clone(), 0, HostBlob { bytes: 1000 });
        assert_eq!(pool.total_bytes(), 1000);

        pool.insert(dom.clone(), 128, HostBlob { bytes: 2000 });
        assert_eq!(pool.total_bytes(), 3000);

        pool.evict(&dom, 0);
        assert_eq!(pool.total_bytes(), 2000);
    }

    // ── DrafterDecision is passed through, not inferred ──────────────────

    #[test]
    fn drafter_decision_is_input() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("drafter");

        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Default Ar.
        let plan_ar = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .unwrap();
        assert_eq!(plan_ar.bundle.drafter, DrafterDecision::Ar);

        // Admission can choose Checkpoint.
        let plan_ckpt = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Checkpoint,
        )
        .unwrap();
        assert_eq!(plan_ckpt.bundle.drafter, DrafterDecision::Checkpoint);

        // Or Reseed.
        let plan_reseed = plan_resume(
            &pool,
            &dom,
            200,
            &lookup(128),
            DrafterDecision::Reseed,
        )
        .unwrap();
        assert_eq!(plan_reseed.bundle.drafter, DrafterDecision::Reseed);
    }

    // ── ResumePlan refuses incomplete bundle ─────────────────────────────

    #[test]
    fn resume_plan_refuses_incomplete_bundle() {
        let incomplete = ResumeBundle {
            attention_pages: false,
            dn_matrices_scales: true,
            conv_rings: true,
            ef_residual: true,
            drafter: DrafterDecision::Ar,
        };
        let err = ResumePlan::new(128, incomplete, 4096, LastTokenHandling::SuffixRecompute);
        assert!(err.is_err());
    }

    // ── Exact match with only p=0 checkpoint ─────────────────────────────

    #[test]
    fn exact_match_falls_back_to_p0() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("fallback-p0");

        // Only a checkpoint at p=0.
        pool.insert(dom.clone(), 0, HostBlob { bytes: 0 });

        // Prompt of exactly 128 tokens, lookup says 128 resumable.
        // But no checkpoint at 128 → NoCheckpoint (p=0 checkpoint doesn't
        // help for a 128-token prompt with resumable=128).
        let err = plan_resume(
            &pool,
            &dom,
            128,
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("no checkpoint at 128");

        assert_eq!(err, MissReason::NoCheckpoint);
    }

    // ── Resumable between page boundaries rounds down ────────────────────

    #[test]
    fn resumable_rounds_down_to_page_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("round");

        // Checkpoint at p=128 only.
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Lookup says 200 resumable (not page-aligned). Should find p=128.
        let plan = plan_resume(
            &pool,
            &dom,
            300,
            &lookup(200),
            DrafterDecision::Ar,
        )
        .unwrap();

        assert_eq!(plan.boundary, 128);
    }
}
