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

/// Secondary index key: `(domain, boundary_p)`.
///
/// WARNING: `(domain, boundary)` is NOT a unique identity — two different
/// token prefixes in one domain can each hold a checkpoint at the same
/// boundary count (page-alignment makes that the common case, not the edge
/// case). The pool's PRIMARY key is [`CheckpointId`]; this index exists only
/// for dedup-on-capture and eviction bookkeeping. Lookups that drive a
/// restore must resolve through `by_id` using the id the radix index
/// recorded on the matching token path, never through this map alone.
type CheckpointKey = (CacheDomain, u64);

#[derive(Debug)]
struct CheckpointEntry<B> {
    /// `(domain, boundary)` this entry was captured under (secondary index).
    key: CheckpointKey,
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
/// Keyed by [`CheckpointId`]. Entries are captured only at page-aligned
/// boundaries. When the pool exceeds `max_bytes`, the oldest
/// **unpinned** checkpoint is evicted until the pool fits. Pinned
/// checkpoints survive eviction.
///
/// Generic over `B: CheckpointBlob` so host tests can use a byte-counting
/// test double without GPU device buffers. The GPU-backed capture and
/// restore paths use `B = DeltaNetSnapshot`.
pub struct QwenCheckpointPool<B: CheckpointBlob> {
    entries: HashMap<CheckpointId, CheckpointEntry<B>>,
    /// `(domain, boundary) -> id` of the NEWEST capture at that key.
    /// Secondary index only — see the [`CheckpointKey`] warning.
    by_boundary: HashMap<CheckpointKey, CheckpointId>,
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
            by_boundary: HashMap::new(),
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

    /// Whether a blob of `bytes` could be inserted without exceeding the
    /// ceiling — accounting for the entry it would displace at `(domain, p)`
    /// (the newest unpinned capture under that boundary key) and for
    /// evicting every unpinned entry. Lets `capture_checkpoint` refuse
    /// BEFORE paying for the GPU allocation and device-to-device copy.
    pub fn can_afford(&self, domain: &CacheDomain, p: u64, bytes: u64) -> bool {
        let replaced_id = self.by_boundary.get(&(domain.clone(), p)).copied();
        let replaced = replaced_id
            .and_then(|id| self.entries.get(&id))
            .filter(|e| !e.pinned)
            .map(|e| e.blob.bytes_len())
            .unwrap_or(0);
        let unpinned_bytes: u64 = self
            .entries
            .iter()
            .filter(|(id, e)| !e.pinned && Some(**id) != replaced_id)
            .map(|(_, e)| e.blob.bytes_len())
            .sum();
        let floor = self.total_bytes.saturating_sub(replaced + unpinned_bytes);
        floor.saturating_add(bytes) <= self.max_bytes
    }

    /// Verify `p` is a valid capture boundary: `p == 0` or `p` is a
    /// multiple of [`PAGE_TOKENS`].
    fn is_aligned(p: u64) -> bool {
        p == 0 || p % PAGE_TOKENS as u64 == 0
    }

    /// Insert a captured checkpoint blob for `domain` at boundary `p`.
    ///
    /// `p` must be page-aligned (`p % 128 == 0` or `p == 0`); otherwise the
    /// entry is **not** inserted and [`CheckpointId::NONE`] is returned
    /// (spec §4.5: "a state at the end of a chunk cannot be relabelled as
    /// an earlier state").
    ///
    /// Every insert mints a FRESH [`CheckpointId`]. If an unpinned entry
    /// already exists at `(domain, p)` — a re-capture of the same prefix,
    /// or a DIFFERENT prefix colliding on the same boundary count — it is
    /// evicted and `by_boundary` repoints at the new id; the radix only
    /// keeps an id for the prefix that actually re-captured (its record is
    /// upserted on insert), so the displaced entry's old references simply
    /// stop matching and fall back to an earlier boundary or cold prefill.
    /// A pinned same-boundary entry is left in place (a restore may be
    /// mid-flight on it) — `by_boundary` still repoints, and the stale
    /// entry ages out through the normal LRU once unpinned.
    ///
    /// If the pool cannot afford the new capture, the oldest **unpinned**
    /// checkpoint is evicted repeatedly until the pool fits. If every
    /// remaining entry is pinned and the capture still does not fit, the NEW
    /// capture is dropped and [`CheckpointId::NONE`] is returned — the byte
    /// ceiling is a hard bound (spec §4.4, §9.1: "a cache-retention limit is
    /// a ceiling"), never oversubscribed; the caller simply publishes
    /// without a checkpoint and the boundary stays honestly unresumable.
    ///
    /// Returns the minted [`CheckpointId`] for the radix index to store, or
    /// [`CheckpointId::NONE`] when the capture was refused.
    ///
    /// GPU memory discipline: every blob this call displaces (a displaced
    /// same-boundary entry, LRU evictions, and a refused capture itself)
    /// is RETURNED to the caller. `DeviceBuffer` has no freeing `Drop`, so
    /// a blob dropped here would leak its device memory permanently while
    /// `total_bytes` is decremented as if freed. The caller routes each
    /// returned blob through `free_gpu`.
    pub fn insert(&mut self, domain: CacheDomain, p: u64, blob: B) -> (CheckpointId, Vec<B>) {
        let mut displaced: Vec<B> = Vec::new();
        if !Self::is_aligned(p) {
            displaced.push(blob);
            return (CheckpointId::NONE, displaced);
        }

        let bytes = blob.bytes_len();
        let key = (domain, p);

        // Capacity is judged against the POST-displacement floor: an
        // unpinned same-boundary occupant is doomed by this insert
        // regardless of how much room is needed, so evicting OTHER entries
        // to cover its bytes would be a spurious LRU eviction (the
        // pre-fix order did exactly that, then displaced anyway).
        // A pinned occupant stays resident — it does NOT credit the floor.
        // The floor is recomputed every iteration: each eviction shrinks
        // total_bytes, and the doomed occupant may itself be the victim the
        // loop evicts (its credit then drops to zero honestly).
        //
        // Fit FIRST: evict before touching the same-boundary occupant.
        // (The pre-fix order destroyed the occupant when the capture was
        // later refused — losing a good checkpoint to a failed insert.)
        loop {
            let doomed_bytes = self
                .by_boundary
                .get(&key)
                .and_then(|id| self.entries.get(id))
                .filter(|e| !e.pinned)
                .map(|e| e.blob.bytes_len())
                .unwrap_or(0);
            let floor = self.total_bytes.saturating_sub(doomed_bytes);
            if floor + bytes <= self.max_bytes {
                break;
            }
            match self.find_oldest_unpinned_id() {
                Some(evict_id) => {
                    if let Some(blob) = self.evict_internal(evict_id) {
                        displaced.push(blob);
                    }
                }
                None => {
                    // Everything left is pinned and the ceiling cannot be
                    // honored. Refuse the capture (handing the blob back for
                    // the caller to free) rather than exceeding the ceiling.
                    displaced.push(blob);
                    return (CheckpointId::NONE, displaced);
                }
            }
        }

        // Displace the previous occupant of this boundary key — UNLESS it
        // is pinned (a plan is mid-restore on it). Either way the boundary
        // repoints at the new capture; the radix records the new id only
        // for the prefix that produced it.
        if let Some(old_id) = self.by_boundary.get(&key).copied() {
            if self.entries.get(&old_id).is_some_and(|e| !e.pinned) {
                if let Some(blob) = self.evict_internal(old_id) {
                    displaced.push(blob);
                }
            }
        }
        self.evicted.remove(&key);

        let id = CheckpointId(self.next_id);
        self.next_id += 1;
        self.lru_clock += 1;
        self.total_bytes += bytes;
        self.by_boundary.insert(key.clone(), id);
        self.entries.insert(
            id,
            CheckpointEntry {
                key,
                blob,
                pinned: false,
                lru_stamp: self.lru_clock,
            },
        );

        (id, displaced)
    }

    /// Find the id of the oldest (smallest `lru_stamp`) unpinned entry.
    fn find_oldest_unpinned_id(&self) -> Option<CheckpointId> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.pinned)
            .min_by_key(|(_, e)| e.lru_stamp)
            .map(|(k, _)| *k)
    }

    /// Remove an entry by id, accounting bytes and recording its boundary
    /// key as evicted. Returns the removed blob for the caller to free on
    /// the GPU — never dropped here (see [`Self::insert`] for the no-`Drop`
    /// rationale).
    fn evict_internal(&mut self, id: CheckpointId) -> Option<B> {
        self.entries.remove(&id).map(|entry| {
            self.total_bytes = self.total_bytes.saturating_sub(entry.blob.bytes_len());
            if self.by_boundary.get(&entry.key) == Some(&id) {
                self.by_boundary.remove(&entry.key);
            }
            self.evicted.insert(entry.key);
            entry.blob
        })
    }

    /// Explicitly evict the checkpoint `id`.
    ///
    /// Returns the removed blob (for the caller to `free_gpu`), or `None`
    /// when no entry existed.
    pub fn evict(&mut self, id: CheckpointId) -> Option<B> {
        self.evict_internal(id)
    }

    /// Check whether checkpoint `id` exists. THE identity-preserving check —
    /// a hit means the blob captured by THIS prefix is still resident.
    pub fn contains_id(&self, id: CheckpointId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Check whether ANY checkpoint exists at `(domain, p)` — boundary-keyed
    /// bookkeeping for metrics/policy, NOT a restore-path identity check.
    pub fn contains(&self, domain: &CacheDomain, p: u64) -> bool {
        self.by_boundary
            .get(&(domain.clone(), p))
            .is_some_and(|id| self.entries.contains_key(id))
    }

    /// Get the [`CheckpointId`] currently registered at `(domain, p)`, if
    /// present. This is the NEWEST capture under that key — callers that
    /// need identity (restore) must use the id the radix recorded instead.
    pub fn id_of(&self, domain: &CacheDomain, p: u64) -> Option<CheckpointId> {
        self.by_boundary.get(&(domain.clone(), p)).copied()
    }

    /// Borrow the blob `id`, refreshing its LRU stamp.
    pub fn get(&mut self, id: CheckpointId) -> Option<&B> {
        if let Some(entry) = self.entries.get_mut(&id) {
            self.lru_clock += 1;
            entry.lru_stamp = self.lru_clock;
            Some(&entry.blob)
        } else {
            None
        }
    }

    /// Borrow the blob `id` without refreshing LRU (read-only).
    pub fn peek(&self, id: CheckpointId) -> Option<&B> {
        self.entries.get(&id).map(|e| &e.blob)
    }

    /// Pin checkpoint `id` so it survives LRU eviction.
    ///
    /// Returns `true` if the entry was found and pinned.
    pub fn pin(&mut self, id: CheckpointId) -> bool {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpin checkpoint `id`, making it eligible for LRU eviction again.
    ///
    /// Returns `true` if the entry was found and unpinned.
    pub fn unpin(&mut self, id: CheckpointId) -> bool {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.pinned = false;
            true
        } else {
            false
        }
    }

    /// Whether checkpoint `id` is pinned.
    pub fn is_pinned(&self, id: CheckpointId) -> bool {
        self.entries.get(&id).map(|e| e.pinned).unwrap_or(false)
    }

    /// Drain and return all stored blobs, clearing the pool. Used by the
    /// serve engine's `free_gpu` to explicitly free each `DeltaNetSnapshot`'s
    /// device buffers on shutdown (the pool itself has no `Drop` impl that
    /// touches the GPU).
    pub fn drain_blobs(&mut self) -> Vec<B> {
        self.total_bytes = 0;
        self.by_boundary.clear();
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

/// The largest `(boundary, checkpoint)` among the lookup's recorded
/// candidates whose pool entry is still resident, with `boundary > 0` and
/// `boundary <= max_p`.
///
/// Candidates are radix-recorded `(boundary, id)` pairs ON THE MATCHED
/// TOKEN PATH — selecting by boundary alone would be unsound: different
/// prefixes legitimately collide on the same page-aligned boundary count,
/// and `by_boundary` would then resolve to whatever prefix captured last.
/// Ids for evicted pool entries are skipped here (a stale radix record is
/// expected after pool eviction — `plan_resume` falls through to the next
/// valid candidate rather than declaring a miss).
///
/// Does NOT return `p = 0` — the initial state does not require a
/// checkpoint and is handled separately by the caller.
fn find_largest_checkpoint<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    candidates: &[(u64, CheckpointId)],
    max_p: u64,
) -> Option<(u64, CheckpointId)> {
    candidates
        .iter()
        .filter(|(p, id)| *p > 0 && *p <= max_p && pool.contains_id(*id))
        .max_by_key(|(p, _)| *p)
        .copied()
}

/// The largest resident candidate with `boundary < below_p` (strict) — the
/// exact-match fallback that must never restore `S_prompt_len`.
fn find_largest_checkpoint_below<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    candidates: &[(u64, CheckpointId)],
    below_p: u64,
) -> Option<(u64, CheckpointId)> {
    candidates
        .iter()
        .filter(|(p, id)| *p > 0 && *p < below_p && pool.contains_id(*id))
        .max_by_key(|(p, _)| *p)
        .copied()
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
    pool: &mut QwenCheckpointPool<B>,
    domain: &CacheDomain,
    prompt_len: u64,
    lookup: &PrefixLookup,
    drafter: DrafterDecision,
) -> Result<ResumePlan, MissReason> {
    let plan = plan_resume_inner(pool, domain, prompt_len, lookup, drafter)?;
    // Pin the chosen checkpoint so an unrelated capture cannot evict it
    // between this plan and the caller's restore. p == 0 is the initial
    // state — no entry exists to pin.
    if plan.checkpoint != CheckpointId::NONE {
        pool.pin(plan.checkpoint);
    }
    Ok(plan)
}

fn plan_resume_inner<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    prompt_len: u64,
    lookup: &PrefixLookup,
    drafter: DrafterDecision,
) -> Result<ResumePlan, MissReason> {
    let resumable = lookup.resumable_tokens;

    // Find the largest radix-recorded checkpoint candidate whose pool
    // entry is still resident, p <= resumable.
    let best = find_largest_checkpoint(pool, &lookup.checkpoint_candidates, resumable);

    match best {
        Some((p, _)) if prompt_len == p && prompt_len > 0 => {
            // Exact match: select an earlier boundary, never restore S_prompt_len.
            // Prefer an earlier checkpoint; fall back to the initial state p=0.
            let earlier = find_largest_checkpoint_below(pool, &lookup.checkpoint_candidates, p);
            let (ep, eid) = earlier.unwrap_or((0, CheckpointId::NONE));
            let byte_cost = if eid == CheckpointId::NONE {
                0
            } else {
                pool.peek(eid).map(|b| b.bytes_len()).unwrap_or(0)
            };
            let bundle = complete_bundle(drafter);
            ResumePlan::new(ep, eid, bundle, byte_cost, LastTokenHandling::EarlierBoundary)
                .map_err(|e| match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                })
        }
        Some((p, id)) => {
            // Normal: p < prompt_len (or prompt_len == 0 with p > 0 — shouldn't
            // normally happen, but SuffixRecompute is still safe).
            let byte_cost = pool.peek(id).map(|b| b.bytes_len()).unwrap_or(0);
            let bundle = complete_bundle(drafter);
            ResumePlan::new(p, id, bundle, byte_cost, LastTokenHandling::SuffixRecompute)
                .map_err(|e| match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                })
        }
        None => {
            // No checkpoint at any page-aligned boundary <= resumable.
            if resumable == 0 && prompt_len == 0 {
                // Empty prompt: p=0, no underflow (spec §4.5).
                let bundle = complete_bundle(drafter);
                ResumePlan::new(0, CheckpointId::NONE, bundle, 0, LastTokenHandling::SuffixRecompute)
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
/// Returns the minted [`CheckpointId`], [`CheckpointId::NONE`] when the
/// pool ceiling cannot cover the capture even after evicting every
/// unpinned entry, or an error if snapshot allocation or the
/// device-to-device copy fails.
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

    // Pre-check the pool ceiling BEFORE allocating the snapshot and paying
    // for the device-to-device copy: a capture the pool cannot afford is a
    // soft refusal, indistinguishable from a hard failure only after the
    // work is already spent.
    let bytes = DeltaNetSnapshot::bytes_for(state);
    if !pool.can_afford(domain, p, bytes) {
        return Ok(CheckpointId::NONE);
    }

    let mut snap = DeltaNetSnapshot::new_for(gpu, state)?;
    snap.save_from(state, gpu)?;

    let (id, displaced) = pool.insert(domain.clone(), p, snap);
    // Displaced blobs (LRU evictions / same-key replacement / a ceiling
    // refusal of this very capture) own device memory with no freeing
    // `Drop` — free them here or they leak for the process lifetime.
    for blob in displaced {
        blob.free_gpu(gpu);
    }
    Ok(id)
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
    id: CheckpointId,
    dst: &mut DeltaNetSnapshot,
) -> HipResult<()> {
    let src = pool
        .peek(id)
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

    /// Build a `PrefixLookup` with the given `resumable_tokens` and the
    /// checkpoint candidates the radix would have recorded for the path —
    /// each `(boundary, id)` must use the id `pool.insert` minted, never a
    /// re-derived `(domain, boundary)` guess.
    fn lookup(resumable: u64, candidates: &[(u64, CheckpointId)]) -> PrefixLookup {
        PrefixLookup {
            matched_tokens: resumable,
            resident_kv_tokens: resumable,
            resumable_tokens: resumable,
            checkpoint_candidates: candidates.to_vec(),
        }
    }

    const PAGE: u64 = PAGE_TOKENS as u64; // 128

    // ── A7: structural — capture at p=128, plan 200-token prompt ────────

    #[test]
    fn a7_capture_at_128_plan_200() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7");

        // Capture at p=128 (page-aligned).
        let (id, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert_ne!(id, CheckpointId::NONE, "insert should mint a nonzero id");

        // Plan for a 200-token prompt: p=128 < 200 → SuffixRecompute.
        let plan = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
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
        let (id128, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Prompt of exactly 128 tokens: must NOT resume at p=128.
        // Should select p=0 (initial state) with EarlierBoundary.
        let plan = plan_resume(
            &mut pool,
            &dom,
            128,
            &lookup(128, &[(128, id128)]),
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
        let (id128, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        let (id256, _) = pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        // Prompt of exactly 256 tokens: should select p=128, not p=256.
        let plan = plan_resume(
            &mut pool,
            &dom,
            256,
            &lookup(256, &[(128, id128), (256, id256)]),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.last_token, LastTokenHandling::EarlierBoundary);
    }

    // ── A7: empty prompt does not underflow ──────────────────────────────

    #[test]
    fn a7_empty_prompt_no_underflow() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-empty");

        // No checkpoints, empty prompt, resumable=0.
        let plan = plan_resume(
            &mut pool,
            &dom,
            0,
            &lookup(0, &[]),
            DrafterDecision::Ar,
        )
        .expect("empty prompt should not underflow");

        assert_eq!(plan.boundary, 0);
        assert_eq!(plan.last_token, LastTokenHandling::SuffixRecompute);
    }

    // ── A8: missing checkpoint → NoCheckpoint error ─────────────────────

    #[test]
    fn a8_missing_checkpoint_is_no_checkpoint() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-missing");

        // Lookup claims 128 resumable tokens, but no checkpoint in pool.
        let err = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[]),
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
        let (id, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert!(pool.evict(id).is_some());

        // Lookup still claims 128 resumable, but checkpoint was evicted.
        let err = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
            DrafterDecision::Ar,
        )
        .expect_err("should be a miss");

        assert_eq!(err, MissReason::Evicted);
    }

    // ── A8: not a silent Hit ─────────────────────────────────────────────

    #[test]
    fn a8_missing_is_not_silent_hit() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-silent");

        // resumable > 0 but no checkpoint → must error, not return a plan.
        let result = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(64, &[]),
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
        let (id, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Plan for 200-token prompt.
        let plan = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
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

        let (id128, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        let (id256, _) = pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });

        let plan = plan_resume(
            &mut pool,
            &dom,
            256, // exact match
            &lookup(256, &[(128, id128), (256, id256)]),
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

        let (id0, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        let (id128, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.total_bytes(), 8192);

        // Insert a third — should evict the oldest (p=0).
        let (id256, _) = pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2, "should still have 2 entries after eviction");
        assert!(!pool.contains(&dom, 0), "oldest (p=0) should be evicted");
        assert!(pool.contains(&dom, 128));
        assert!(pool.contains(&dom, 256));
        assert_ne!(id0, CheckpointId::NONE);
        assert_ne!(id128, CheckpointId::NONE);
        assert_ne!(id256, CheckpointId::NONE);
    }

    /// GPU-memory discipline: every blob displaced by an insert (LRU
    /// eviction, same-key replacement, ceiling refusal) is RETURNED to the
    /// caller — nothing is dropped inside the pool, because a dropped
    /// `DeltaNetSnapshot` leaks its device buffers (no freeing `Drop`).
    #[test]
    fn displaced_blobs_are_returned_never_dropped() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("displaced");

        // LRU eviction returns the evicted blob.
        let _ = pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        let (id128, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        let (_id256, displaced) = pool.insert(dom.clone(), 256, HostBlob { bytes: 4096 });
        assert_eq!(displaced.len(), 1, "evicted blob must be handed back");
        assert_eq!(displaced[0].bytes, 4096);
        assert!(!pool.contains(&dom, 0), "p=0 was the one evicted");

        // Same-key replacement returns the replaced blob.
        let (id256, _) = pool.insert(dom.clone(), 256, HostBlob { bytes: 2048 });
        assert_eq!(displaced.len(), 1, "replaced blob must be handed back");
        assert_eq!(displaced[0].bytes, 4096);
        assert_eq!(pool.total_bytes(), 4096 + 2048);

        // Ceiling refusal with everything pinned returns the refused blob.
        pool.pin(id128);
        pool.pin(id256);
        let (id, displaced) = pool.insert(dom.clone(), 384, HostBlob { bytes: 8192 });
        assert_eq!(id, CheckpointId::NONE, "refused capture reports NONE");
        assert_eq!(
            displaced.len(),
            1,
            "the refused capture's blob must be handed back for freeing"
        );
        assert_eq!(displaced[0].bytes, 8192);

        // Unaligned boundary: same contract.
        let (id, displaced) = pool.insert(dom.clone(), 100, HostBlob { bytes: 4096 });
        assert_eq!(id, CheckpointId::NONE);
        assert_eq!(displaced.len(), 1);
    }

    // ── LRU byte bound: pinned checkpoints survive eviction ─────────────

    #[test]
    fn lru_pinned_survives_eviction() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru-pinned");

        let (id0, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        let _ = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Pin the oldest (p=0).
        assert!(pool.pin(id0));
        assert!(pool.is_pinned(id0));

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

        let (id0, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Access p=0 to make it more recent than p=128.
        let _ = pool.get(id0);

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

        let (id_a, _) = pool.insert(dom_a.clone(), 128, HostBlob { bytes: 4096 });

        // dom_b has no checkpoint at p=128.
        assert!(!pool.contains(&dom_b, 128));
        assert!(pool.contains(&dom_a, 128));

        // Planning with dom_b should miss.
        let err = plan_resume(
            &mut pool,
            &dom_b,
            200,
            &lookup(128, &[]),
            DrafterDecision::Ar,
        )
        .expect_err("different domain should miss");

        assert_eq!(err, MissReason::NoCheckpoint);

        // Planning with dom_a should succeed.
        let plan = plan_resume(
            &mut pool,
            &dom_a,
            200,
            &lookup(128, &[(128, id_a)]),
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

        let (id, _) = pool.insert(dom.clone(), 100, HostBlob { bytes: 4096 });
        assert_eq!(id, CheckpointId::NONE, "non-page-aligned boundary must be rejected");
        assert!(!pool.contains(&dom, 100));
        assert_eq!(pool.total_bytes(), 0);
    }

    // ── p=0 is a valid capture boundary ──────────────────────────────────

    #[test]
    fn p0_is_valid_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("p0");

        let (id, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 0 });
        assert_ne!(id, CheckpointId::NONE);
        assert!(pool.contains(&dom, 0));
    }

    // ── CheckpointId is monotonic ────────────────────────────────────────

    #[test]
    fn checkpoint_ids_are_monotonic() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("monotonic");

        let (id1, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 100 });
        let (id2, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 100 });
        let (id3, _) = pool.insert(dom.clone(), 256, HostBlob { bytes: 100 });

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert_eq!(id1, CheckpointId(1));
    }

    // ── Byte accounting is accurate ──────────────────────────────────────

    #[test]
    fn byte_accounting() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("bytes");

        let (id0, _) = pool.insert(dom.clone(), 0, HostBlob { bytes: 1000 });
        assert_eq!(pool.total_bytes(), 1000);

        pool.insert(dom.clone(), 128, HostBlob { bytes: 2000 });
        assert_eq!(pool.total_bytes(), 3000);

        let _ = pool.evict(id0);
        assert_eq!(pool.total_bytes(), 2000);
    }

    // ── DrafterDecision is passed through, not inferred ──────────────────

    #[test]
    fn drafter_decision_is_input() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("drafter");

        let (id, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Default Ar.
        let plan_ar = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
            DrafterDecision::Ar,
        )
        .unwrap();
        assert_eq!(plan_ar.bundle.drafter, DrafterDecision::Ar);

        // Admission can choose Checkpoint.
        let plan_ckpt = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
            DrafterDecision::Checkpoint,
        )
        .unwrap();
        assert_eq!(plan_ckpt.bundle.drafter, DrafterDecision::Checkpoint);

        // Or Reseed.
        let plan_reseed = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id)]),
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
        let err = ResumePlan::new(128, CheckpointId::NONE, incomplete, 4096, LastTokenHandling::SuffixRecompute);
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
            &mut pool,
            &dom,
            128,
            &lookup(128, &[]),
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
        let (id, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });

        // Lookup says 200 resumable (not page-aligned). Should find p=128.
        let plan = plan_resume(
            &mut pool,
            &dom,
            300,
            &lookup(200, &[(128, id)]),
            DrafterDecision::Ar,
        )
        .unwrap();

        assert_eq!(plan.boundary, 128);

    // ── Boundary collision across prefixes: id-keyed restore is safe ────

    #[test]
    fn same_boundary_collision_never_crosses_conversations() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("collision");

        // Prefix A checkpoints at 128; a DIFFERENT prefix B in the same
        // domain later captures at the same boundary count. Under the old
        // `(domain, boundary)` keying, B's insert overwrote A's entry while
        // the radix still mapped A's path to (dom, 128) — a lookup for A's
        // prompt then restored B's recurrent state: cross-conversation
        // corruption.
        let (id_a, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        let (id_b, displaced) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert_ne!(id_a, id_b, "every capture mints a fresh id");
        assert_eq!(displaced.len(), 1, "the displaced A blob is handed back to free");

        // A's radix record points at id_a — the entry is gone, so the plan
        // must miss honestly, never fall through to B's blob at the same
        // boundary.
        let err = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id_a)]),
            DrafterDecision::Ar,
        )
        .expect_err("A's evicted checkpoint must not resolve to B's");
        assert_eq!(err, MissReason::Evicted);

        // B's radix record resolves to its own checkpoint.
        let plan = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id_b)]),
            DrafterDecision::Ar,
        )
        .expect("B's checkpoint is resident");
        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.checkpoint, id_b);

        // Boundary bookkeeping still reports a checkpoint at (dom, 128).
        assert!(pool.contains(&dom, 128));
        assert_eq!(pool.id_of(&dom, 128), Some(id_b));
    }

    // ── Pinned same-boundary entry survives; boundary repoints fresh ────

    #[test]
    fn pinned_same_boundary_entry_coexists_with_fresh_capture() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("pinned-collision");

        let (id_a, _) = pool.insert(dom.clone(), 128, HostBlob { bytes: 4096 });
        assert!(pool.pin(id_a), "pin A (a plan is mid-restore on it)");

        // Re-capture at the same boundary while A is pinned: A stays
        // resident, by_boundary repoints at the fresh id.
        let (id_b, displaced) = pool.insert(dom.clone(), 128, HostBlob { bytes: 2048 });
        assert!(displaced.is_empty(), "pinned A is not displaced");
        assert!(pool.contains_id(id_a), "pinned A survives the collision");
        assert!(pool.contains_id(id_b));
        assert_eq!(pool.id_of(&dom, 128), Some(id_b));
        assert_eq!(pool.total_bytes(), 4096 + 2048);

        // A's plan still restores A's blob by id.
        let plan = plan_resume(
            &mut pool,
            &dom,
            200,
            &lookup(128, &[(128, id_a)]),
            DrafterDecision::Ar,
        )
        .expect("pinned A remains restorable by id");
        assert_eq!(plan.checkpoint, id_a);
    }
    }
}
