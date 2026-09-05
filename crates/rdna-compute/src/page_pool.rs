// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// PagePool — paged KV cache allocator (vLLM-style PagedAttention).
//
// Replaces SlotPool's fixed per-slot slabs with a shared free-list of
// physical pages. Each session owns a BlockTable (Vec<u32>) mapping
// logical page index -> physical page index. Pages are allocated on
// demand and freed when a session ends, enabling:
//
//   - Dynamic memory allocation (no pre-reserved per-slot capacity)
//   - Prefix sharing across sessions (shared physical pages, refcounted)
//   - Over-subscription without host swap (more sessions than fixed
//     slabs would allow, since short sessions consume fewer pages)
//
// The block table is uploaded to the GPU as a flat i32 array and its
// device address is stored in KvSlotDesc.block_table. The attention
// kernels use kv_offset_for_k/v() to translate logical positions to
// physical byte offsets via the block table — no kernel is touched.
//
// ── Wave 2: ownership, COW, and deferred reclaim (spec §4.3 C3, §4.4 C4)
//
// Each physical page carries a `PageMeta` tracking:
//   - `PageState`: Free → Private → Sealed → CacheOnly → ReclaimPending → Free
//   - Checked refcounts separated by `LeaseClass` (table, cache, in-flight)
//   - A per-page `generation` so stale `PageHandle`s fail before dereference
//
// Copy-on-write (`CowPlan`/`plan_cow`/`commit_cow`/`abort_cow`) lets a
// session rewrite a shared (Sealed) page by reserving a private
// destination and copying the valid prefix — raw bytes, no dequant/requant
// — before rebinding the block table. Failed reservations release only
// newly reserved resources (spec §4.3).
//
// Deferred reclaim: when all table/cache refs drop to zero but in-flight
// device reads remain, the page enters `ReclaimPending` and is NOT
// immediately reusable. `drain_completed()` is the synchronous step-fence
// that frees confirmed-complete pages. The upgrade seam to HIP-event-based
// completion is documented at that method.

use crate::kv_slots::{preflight_alloc, KvSlotDesc, R9700_VRAM_BYTES};

/// Page size in tokens. Must match PAGE_TOKENS in slot_pool.rs and the
/// tile size the flash path walks KV in.
pub const PAGE_TOKENS: usize = 128;

/// A physical page index in the shared arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

// =========================================================================
// Wave 2 — physical ownership types (spec §4.3 C3, §4.4 C4)
//
// rdna-compute owns these *physical* types; hipfire-runtime's
// serve_contract.rs owns the matching *logical* types (CacheDomain,
// StepTicket, ReleaseDisposition, …).
// =========================================================================

/// A validated handle to a physical page (spec §4.3). The host validates
/// `generation` (and `epoch`) before table upload; a stale handle — the
/// page was freed and re-allocated to another session — fails before
/// dereference, preventing use-after-free on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageHandle {
    /// Physical page index.
    pub phys: u32,
    /// Pool allocation epoch (bumped on device reload; spec §4.1
    /// "allocation epoch").  In this implementation the epoch is always 0
    /// and never bumped — the field and check exist so the upgrade path is
    /// non-breaking.
    pub epoch: u32,
    /// Per-page generation.  Bumped each time the page transitions
    /// Free → allocated and again on Free → re-alloc, so a handle from a
    /// prior allocation cycle mismatches and is rejected.
    pub generation: u32,
}

/// Lifecycle state of a physical page (spec §4.3 C3).
///
/// ```text
///   Free ──alloc──▶ Private ──seal/share──▶ Sealed
///    ▲                  │                       │
///    │                  │ release (rc→0)        │ release (rc→0)
///    │                  ▼                       │
///    │                 Free              CacheOnly
///    │                                    │
///    │                              release (rc→0, inflight>0)
///    │                                    │
///    │                                    ▼
///    └──────────── drain_completed ◀── ReclaimPending
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageState {
    /// In the free list; no refs; generation already bumped.
    Free,
    /// Owned by exactly one block table; mutable (writes allowed).
    Private,
    /// Shared (≥2 table refs) or cache-pinned; immutable — any write
    /// intention must trigger copy-on-write (spec §4.3).
    Sealed,
    /// No active table ref but cache-owned; still immutable, still
    /// resident.  Can be re-attached (table ref added) or evicted.
    CacheOnly,
    /// All table/cache refs released but in-flight device/transport reads
    /// remain; NOT in the free list until `drain_completed` confirms
    /// completion (spec §4.4).
    ReclaimPending,
}

/// Lease class identifying *why* a page is pinned (spec §4.3: "Separate
/// active/table refs, cache ownership, in-flight/transfer leases; physical
/// bytes charged once").  The total refcount is the sum across classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseClass {
    /// A live session's block table references this page.
    Table,
    /// The prefix cache owns this page (it may outlive any single session).
    Cache,
    /// A device kernel or transport reader is still accessing this page;
    /// reuse is deferred until completion (spec §4.4).
    InFlight,
}

/// Per-page metadata (internal).  Replaces the flat `refcounts: Vec<u32>`
/// from Wave 1 with checked, class-separated refcounts and the state
/// machine.
#[derive(Debug, Clone, Copy)]
struct PageMeta {
    state: PageState,
    generation: u32,
    table_refs: u32,
    cache_refs: u32,
    inflight_refs: u32,
}

impl PageMeta {
    const fn free() -> Self {
        Self {
            state: PageState::Free,
            generation: 0,
            table_refs: 0,
            cache_refs: 0,
            inflight_refs: 0,
        }
    }

    /// Total references across all lease classes.
    fn total_refs(&self) -> u32 {
        self.table_refs + self.cache_refs + self.inflight_refs
    }
}

/// One copy-on-write operation planned by [`PagePool::plan_cow`].
///
/// The caller must raw-copy `valid_prefix_tokens` positions' worth of K
/// and V bytes from `src_phys` to `dst_phys` (spec §4.3: "copy valid
/// prefix in every K/V layer arena … preserve encoded bytes/quant/
/// strides, no dequant/requant").  `k_copy_bytes` / `v_copy_bytes` are
/// pre-computed for the caller's convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowCopy {
    /// Logical page index in the block table being rebound.
    pub logical_page: usize,
    /// Source physical page (Sealed or CacheOnly — being COWed away from).
    pub src_phys: u32,
    /// Destination physical page (Private — the COW copy).
    pub dst_phys: u32,
    /// Valid tokens before the write start in this page (0 when the write
    /// begins at the page boundary).
    pub valid_prefix_tokens: usize,
    /// K-arena bytes to copy (`valid_prefix_tokens × k_per_pos_bytes`).
    pub k_copy_bytes: usize,
    /// V-arena bytes to copy (`valid_prefix_tokens × v_per_pos_bytes`).
    pub v_copy_bytes: usize,
}

/// A copy-on-write plan produced by [`PagePool::plan_cow`] and consumed by
/// [`PagePool::commit_cow`] or [`PagePool::abort_cow`].
///
/// `plan_cow` reserves private destination pages for every Sealed or
/// CacheOnly page in the write interval **without** modifying the block
/// table.  The caller executes the GPU memcpys, then calls `commit_cow`
/// to rebind the table and adjust refcounts, or `abort_cow` to release
/// the reserved pages on failure.  This ordering ensures a failed
/// reservation leaves the original table and pool state intact (spec §4.3:
/// "transactional failure releases only newly reserved resources").
#[derive(Debug, Clone)]
pub struct CowPlan {
    copies: Vec<CowCopy>,
}

impl CowPlan {
    /// The individual copy operations this plan requires.
    pub fn copies(&self) -> &[CowCopy] {
        &self.copies
    }

    /// True when no COW is needed (the write interval touches no
    /// Sealed/CacheOnly pages).
    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }
}

// =========================================================================
// BlockTable — unchanged from Wave 1
// =========================================================================

/// A session's block table: logical page index -> physical page index.
/// Uploaded to the GPU as a flat i32 array; its device address goes into
/// KvSlotDesc.block_table.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical -> physical page mapping.
    pages: Vec<u32>,
    /// Number of live tokens (not pages). The last page may be partially
    /// filled; `live_tokens` determines how many positions the kernel reads.
    live_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            live_tokens: 0,
        }
    }

    /// Number of pages currently allocated.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Number of live tokens (logical KV length).
    pub fn live_tokens(&self) -> usize {
        self.live_tokens
    }

    /// Physical page index for logical page `lp`, or None if out of range.
    pub fn physical(&self, lp: usize) -> Option<u32> {
        self.pages.get(lp).copied()
    }

    /// Raw page indices for GPU upload.
    pub fn page_indices(&self) -> &[u32] {
        &self.pages
    }

    pub(crate) fn push_page(&mut self, page: u32) {
        self.pages.push(page);
    }

    /// Rebind logical page `lp` to physical page `phys`.  Used by
    /// `commit_cow` after the GPU memcpy is issued.
    pub(crate) fn set_page(&mut self, lp: usize, phys: u32) {
        if lp < self.pages.len() {
            self.pages[lp] = phys;
        }
    }

    /// Truncate to exactly `n_pages` pages, returning the freed pages.
    /// Used when trimming a session's KV (e.g. on eviction).
    fn truncate(&mut self, n_pages: usize) -> Vec<u32> {
        let freed: Vec<u32> = self.pages.drain(n_pages..).collect();
        self.live_tokens = self.live_tokens.min(n_pages * PAGE_TOKENS);
        freed
    }

    /// Set the live token count. Must be <= num_pages * PAGE_TOKENS.
    pub(crate) fn set_live_tokens(&mut self, tokens: usize) {
        debug_assert!(
            tokens <= self.pages.len() * PAGE_TOKENS,
            "live_tokens {} exceeds capacity {} ({} pages * {} tokens)",
            tokens,
            self.pages.len() * PAGE_TOKENS,
            self.pages.len(),
            PAGE_TOKENS
        );
        self.live_tokens = tokens;
    }

    /// Host-side copy of the table (pages + live_tokens). Used by
    /// `SlotPool::share_prefix` to hand `PagePool::share_prefix` a stable
    /// snapshot of the source without overlapping field borrows.
    pub(crate) fn snapshot(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            live_tokens: self.live_tokens,
        }
    }
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// PagePool
// =========================================================================

/// Free-list page allocator for a shared KV arena.
///
/// The arena is a single contiguous buffer per layer (K and V each).
/// `PagePool` tracks which pages are free and which are allocated to
/// which session. Pages are `PAGE_TOKENS * per_pos_bytes` bytes each.
///
/// Prefix sharing: when two sessions share the same token prefix, they
/// share the physical pages for that prefix. Shared pages are refcounted,
/// sealed (immutable), and only freed when the last session releases them.
#[derive(Debug)]
pub struct PagePool {
    /// Total number of physical pages in the arena.
    n_pages: usize,
    /// Per-position strides in bytes (n_kv_heads * (head_dim/32) * 34 for
    /// Q8_0). Equal on q8/bf16; DIFFERENT on the rotated-K tiers, where the
    /// K page is `PAGE_TOKENS × k_per_pos_bytes` and the V page
    /// `PAGE_TOKENS × v_per_pos_bytes`. One block table maps both arenas —
    /// a logical page resolves to the same physical index for K and V.
    k_per_pos_bytes: usize,
    v_per_pos_bytes: usize,
    /// Free list of physical page indices. Pages are popped from the end.
    /// Does NOT include ReclaimPending pages.
    free_pages: Vec<u32>,
    /// Per-page metadata (state, generation, class-separated refcounts).
    /// Index is physical page index.
    page_meta: Vec<PageMeta>,
    /// Pages pending reclaim: table/cache refs released but in-flight
    /// device reads remain. Freed by `drain_completed`.
    reclaim_pending: Vec<u32>,
    /// Pool allocation epoch (spec §4.1). Always 0 in this implementation;
    /// the field and `PageHandle.epoch` check exist so a future device-reload
    /// upgrade is non-breaking.
    epoch: u32,
}

impl PagePool {
    /// Create a page pool with `n_pages` physical pages, each holding
    /// `PAGE_TOKENS` positions of `per_pos_bytes` bytes.
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(n_pages: usize, per_pos_bytes: usize) -> Result<Self, String> {
        Self::new_with_strides(n_pages, per_pos_bytes, per_pos_bytes)
    }

    /// [`PagePool::new`] with independent K/V strides (the rotated-K tiers).
    pub fn new_with_strides(
        n_pages: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Result<Self, String> {
        assert!(n_pages > 0, "n_pages must be positive");
        assert!(k_per_pos_bytes > 0, "k_per_pos_bytes must be positive");
        assert!(v_per_pos_bytes > 0, "v_per_pos_bytes must be positive");

        let page_bytes = (PAGE_TOKENS * k_per_pos_bytes)
            .checked_add(PAGE_TOKENS * v_per_pos_bytes)
            .ok_or_else(|| "PagePool: arena size overflows u64".to_string())?;
        let total = (page_bytes as u64)
            .checked_mul(n_pages as u64)
            .ok_or_else(|| "PagePool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "PagePool arena")?;

        let free_pages: Vec<u32> = (0..n_pages as u32).rev().collect();
        Ok(Self {
            n_pages,
            k_per_pos_bytes,
            v_per_pos_bytes,
            free_pages,
            page_meta: vec![PageMeta::free(); n_pages],
            reclaim_pending: Vec::new(),
            epoch: 0,
        })
    }

    /// Total number of physical pages.
    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    /// Per-position stride in bytes. Equal-stride pools only — see
    /// [`PagePool::k_per_pos_bytes`] / [`PagePool::v_per_pos_bytes`].
    pub fn per_pos_bytes(&self) -> usize {
        debug_assert_eq!(
            self.k_per_pos_bytes, self.v_per_pos_bytes,
            "per_pos_bytes() is the EQUAL-stride accessor; this pool carries \
             distinct K/V strides"
        );
        self.k_per_pos_bytes
    }

    /// K-arena per-position stride in bytes.
    pub fn k_per_pos_bytes(&self) -> usize {
        self.k_per_pos_bytes
    }

    /// V-arena per-position stride in bytes.
    pub fn v_per_pos_bytes(&self) -> usize {
        self.v_per_pos_bytes
    }

    /// Bytes per K page (PAGE_TOKENS * k_per_pos_bytes).
    pub fn k_page_bytes(&self) -> usize {
        PAGE_TOKENS * self.k_per_pos_bytes
    }

    /// Bytes per V page (PAGE_TOKENS * v_per_pos_bytes).
    pub fn v_page_bytes(&self) -> usize {
        PAGE_TOKENS * self.v_per_pos_bytes
    }

    /// Bytes per page (PAGE_TOKENS * per_pos_bytes). Equal-stride pools only.
    pub fn page_bytes(&self) -> usize {
        self.k_page_bytes()
    }

    /// Bytes in ONE arena (K or V). The pool holds two of these.
    /// Equal-stride pools only — see `k_arena_bytes`/`v_arena_bytes`.
    pub fn arena_bytes(&self) -> usize {
        self.k_arena_bytes()
    }

    /// Bytes of the K arena.
    pub fn k_arena_bytes(&self) -> usize {
        self.n_pages * self.k_page_bytes()
    }

    /// Bytes of the V arena.
    pub fn v_arena_bytes(&self) -> usize {
        self.n_pages * self.v_page_bytes()
    }

    /// Number of free pages available for allocation (excludes
    /// ReclaimPending).
    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Maximum token capacity of the pool (all pages allocated).
    pub fn max_tokens(&self) -> usize {
        self.n_pages * PAGE_TOKENS
    }

    // ── Wave 2: per-page metadata accessors ───────────────────────────

    /// Total refcount (all lease classes) for physical page `phys`.
    pub(crate) fn refcount(&self, phys: u32) -> u32 {
        self.page_meta[phys as usize].total_refs()
    }

    /// Current [`PageState`] of physical page `phys`.
    pub fn page_state(&self, phys: u32) -> PageState {
        self.page_meta[phys as usize].state
    }

    /// Current generation of physical page `phys`.  A [`PageHandle`]
    /// carrying a mismatched generation is stale.
    pub fn page_generation(&self, phys: u32) -> u32 {
        self.page_meta[phys as usize].generation
    }

    /// Pool allocation epoch (spec §4.1).
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Number of pages in the ReclaimPending queue.
    pub fn reclaim_pending_count(&self) -> usize {
        self.reclaim_pending.len()
    }

    // ── Wave 2: handle validation ─────────────────────────────────────

    /// Validate that `handle` refers to the current allocation of its
    /// physical page (spec §4.3: "stale handles fail before dereference").
    ///
    /// Returns `Err` if the epoch changed (device reload), the generation
    /// changed (page was freed and re-allocated), or the page is Free
    /// (use-after-free).
    pub fn validate_handle(&self, handle: &PageHandle) -> Result<(), String> {
        if handle.epoch != self.epoch {
            return Err(format!(
                "PageHandle: epoch mismatch (handle={}, pool={}) — device reloaded?",
                handle.epoch, self.epoch
            ));
        }
        let meta = &self.page_meta[handle.phys as usize];
        if meta.generation != handle.generation {
            return Err(format!(
                "PageHandle: generation mismatch for phys {} (handle={}, \
                 current={}) — stale handle",
                handle.phys, handle.generation, meta.generation
            ));
        }
        if meta.state == PageState::Free {
            return Err(format!(
                "PageHandle: phys {} is Free (generation={}) — use-after-free",
                handle.phys, handle.generation
            ));
        }
        Ok(())
    }

    // ── Wave 2: internal refcount helpers ─────────────────────────────

    /// Decrement the table refcount for `phys`, transitioning state when
    /// it reaches zero.  Returns `Err` on underflow (spec §4.3: checked
    /// arithmetic, no wraparound).
    fn dec_table_ref(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        if meta.table_refs == 0 {
            return Err(format!(
                "dec_table_ref: table_refs underflow for phys {} (state={:?})",
                phys, meta.state
            ));
        }
        meta.table_refs -= 1;
        if meta.table_refs == 0 {
            if meta.cache_refs > 0 {
                meta.state = PageState::CacheOnly;
            } else if meta.inflight_refs > 0 {
                meta.state = PageState::ReclaimPending;
                self.reclaim_pending.push(phys);
            } else {
                meta.state = PageState::Free;
                meta.generation = meta.generation.wrapping_add(1);
                self.free_pages.push(phys);
            }
        }
        Ok(())
    }

    /// Free a just-reserved page back to the free list (used by COW
    /// rollback).  Bumps generation so any handle obtained during the
    /// failed reservation is stale.
    fn release_reserved(&mut self, phys: u32) {
        let meta = &mut self.page_meta[phys as usize];
        meta.state = PageState::Free;
        meta.table_refs = 0;
        meta.generation = meta.generation.wrapping_add(1);
        self.free_pages.push(phys);
    }

    // ── Allocation ────────────────────────────────────────────────────

    /// Allocate `n_pages` fresh pages and append them to `table`.
    /// Returns the number of pages actually allocated (may be less than
    /// requested if the pool is nearly full).
    pub fn alloc_pages(&mut self, table: &mut BlockTable, n_pages: usize) -> usize {
        let can_alloc = n_pages.min(self.free_pages.len());
        for _ in 0..can_alloc {
            let page = self.free_pages.pop().expect("free_pages non-empty");
            let meta = &mut self.page_meta[page as usize];
            meta.state = PageState::Private;
            meta.generation = meta.generation.wrapping_add(1);
            meta.table_refs = 1;
            table.push_page(page);
        }
        can_alloc
    }

    /// Allocate `n_pages` fresh pages and append them to `table`, returning
    /// [`PageHandle`]s carrying the current generation for each page (spec
    /// §4.3).  Unlike [`alloc_pages`], this returns `Err` on insufficient
    /// free pages rather than silently allocating fewer.
    pub fn alloc_pages_checked(
        &mut self,
        table: &mut BlockTable,
        n_pages: usize,
    ) -> Result<Vec<PageHandle>, String> {
        if n_pages > self.free_pages.len() {
            return Err(format!(
                "PagePool: need {} pages but only {} free",
                n_pages,
                self.free_pages.len()
            ));
        }
        let mut handles = Vec::with_capacity(n_pages);
        for _ in 0..n_pages {
            let page = self.free_pages.pop().expect("free_pages non-empty");
            let meta = &mut self.page_meta[page as usize];
            meta.state = PageState::Private;
            meta.generation = meta.generation.wrapping_add(1);
            meta.table_refs = 1;
            handles.push(PageHandle {
                phys: page,
                epoch: self.epoch,
                generation: meta.generation,
            });
            table.push_page(page);
        }
        Ok(handles)
    }

    // ── Prefix sharing (existing contract preserved) ──────────────────

    /// Share the first `n_pages` pages of `src` with `dst` by pushing the
    /// same physical indices into `dst` and incrementing their refcounts.
    /// Used for prefix sharing: the shared prefix pages persist until the
    /// last session releases them.
    ///
    /// After sharing, the shared pages are marked [`PageState::Sealed`]
    /// (immutable).  Any subsequent write intention against a Sealed page
    /// must go through copy-on-write ([`plan_cow`]/[`commit_cow`]).
    ///
    /// # The copy-on-write-free contract (must hold or KV corrupts silently)
    ///
    /// There is NO copy-on-write in this *sharing* path. Sharing is only
    /// safe at a page-aligned fork point, which this method enforces on
    /// both sides:
    ///
    /// - **Every shared page must be FULL in `src`** (`n_pages *
    ///   PAGE_TOKENS <= src.live_tokens()`). A full page is never written
    ///   again by append-only growth — src's next token goes to a fresh
    ///   page — so src cannot mutate bytes dst is reading.
    /// - **`dst` must have an empty table.** The shared pages BECOME dst's
    ///   prefix; the caller then sets dst's live length to the fork point
    ///   (`n_pages * PAGE_TOKENS`), so dst's own writes also land in fresh
    ///   pages only. Sharing into a non-empty table would interleave two
    ///   unrelated mappings.
    ///
    /// Sharing a partially-filled page, or letting either session's write
    /// frontier sit inside a shared page, lets one session's tokens land in
    /// the other's cache with no error raised.
    ///
    /// Panics if `src` has fewer than `n_pages` pages.
    ///
    /// [`plan_cow`]: PagePool::plan_cow
    /// [`commit_cow`]: PagePool::commit_cow
    pub fn share_prefix(
        &mut self,
        src: &BlockTable,
        dst: &mut BlockTable,
        n_pages: usize,
    ) -> Result<(), String> {
        if n_pages > src.num_pages() {
            return Err(format!(
                "share_prefix: src has {} pages but {} requested",
                src.num_pages(),
                n_pages
            ));
        }
        if n_pages * PAGE_TOKENS > src.live_tokens() {
            return Err(format!(
                "share_prefix: sharing {} pages covers {} tokens but src has \
                 only {} live — refusing to share a partially-filled page \
                 without copy-on-write (would corrupt src on its next append)",
                n_pages,
                n_pages * PAGE_TOKENS,
                src.live_tokens()
            ));
        }
        if dst.num_pages() != 0 {
            return Err(format!(
                "share_prefix: dst already holds {} pages — sharing is only \
                 defined into an empty table (fork point must be the start \
                 of dst's KV)",
                dst.num_pages()
            ));
        }
        for lp in 0..n_pages {
            let phys = src.physical(lp).expect("src page exists");
            let meta = &mut self.page_meta[phys as usize];
            meta.table_refs = meta
                .table_refs
                .checked_add(1)
                .ok_or_else(|| format!("share_prefix: refcount overflow for phys {}", phys))?;
            // Shared pages are immutable (spec §4.3: "cached pages immutable
            // even with one owner").
            meta.state = PageState::Sealed;
            dst.push_page(phys);
        }
        Ok(())
    }

    /// Increment the table refcount for physical page `phys` and seal it
    /// if currently Private (spec §4.3: shared pages are immutable).
    ///
    /// Uses checked arithmetic — returns `Err` on overflow or if the page
    /// is Free.
    pub fn refcount_inc(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("refcount_inc: phys {} is Free", phys));
        }
        meta.table_refs = meta
            .table_refs
            .checked_add(1)
            .ok_or_else(|| format!("refcount_inc: overflow for phys {}", phys))?;
        if meta.state == PageState::Private {
            meta.state = PageState::Sealed;
        }
        Ok(())
    }

    // ── Release / free ────────────────────────────────────────────────

    /// Free all pages in `table`, decrementing table refcounts with
    /// checked arithmetic.  Pages whose refs reach zero are returned to
    /// the free list (or to the ReclaimPending queue if in-flight refs
    /// remain).  Returns `Err` if any page had a zero table refcount
    /// (underflow / duplicate release); the table is still cleared.
    pub fn release_table(&mut self, table: &mut BlockTable) -> Result<(), String> {
        let mut had_error = false;
        for &phys in table.page_indices() {
            if let Err(_) = self.dec_table_ref(phys) {
                had_error = true;
            }
        }
        table.pages.clear();
        table.live_tokens = 0;
        if had_error {
            Err("release_table: refcount underflow on one or more pages".to_string())
        } else {
            Ok(())
        }
    }

    /// Free the last `n_pages` pages from `table`, decrementing table
    /// refcounts with checked arithmetic.  Used when trimming a session's
    /// KV (e.g. sliding window eviction).  Returns `Err` on underflow.
    pub fn free_pages_from_tail(
        &mut self,
        table: &mut BlockTable,
        n_pages: usize,
    ) -> Result<(), String> {
        let freed = table.truncate(table.num_pages().saturating_sub(n_pages));
        for phys in freed {
            self.dec_table_ref(phys)?;
        }
        Ok(())
    }

    /// Ensure `table` has enough pages to hold `n_tokens` positions,
    /// allocating new pages as needed. Returns the number of new pages
    /// allocated.
    pub fn ensure_capacity(
        &mut self,
        table: &mut BlockTable,
        n_tokens: usize,
    ) -> Result<usize, String> {
        let needed_pages = n_tokens.div_ceil(PAGE_TOKENS);
        let current_pages = table.num_pages();
        if needed_pages <= current_pages {
            table.set_live_tokens(n_tokens);
            return Ok(0);
        }
        let to_alloc = needed_pages - current_pages;
        if to_alloc > self.free_pages.len() {
            return Err(format!(
                "PagePool: need {} more pages but only {} free ({} tokens requested, {} pages allocated)",
                to_alloc,
                self.free_pages.len(),
                n_tokens,
                current_pages
            ));
        }
        let allocated = self.alloc_pages(table, to_alloc);
        table.set_live_tokens(n_tokens);
        Ok(allocated)
    }

    // ── Seal ──────────────────────────────────────────────────────────

    /// Mark a full page as [`PageState::Sealed`] (immutable).  Idempotent
    /// on already-Sealed pages.  Returns `Err` if the page is Free,
    /// CacheOnly, or ReclaimPending (spec §4.3).
    pub fn seal(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        match meta.state {
            PageState::Private => {
                meta.state = PageState::Sealed;
                Ok(())
            }
            PageState::Sealed => Ok(()),
            _ => Err(format!(
                "seal: phys {} is {:?}, cannot seal",
                phys, meta.state
            )),
        }
    }

    // ── Copy-on-write (spec §4.3 C3) ──────────────────────────────────

    /// Inspect the write interval `[write_start, write_end)` and reserve
    /// private destination pages for every [`PageState::Sealed`] or
    /// [`PageState::CacheOnly`] page in that interval.  Returns a
    /// [`CowPlan`] describing the copies the caller must perform.
    ///
    /// **Does NOT modify the block table** — the caller executes the GPU
    /// memcpys, then calls [`commit_cow`] to rebind the table, or
    /// [`abort_cow`] to release the reserved pages on failure.
    ///
    /// If any reservation fails (OOM), all reserved pages are released
    /// and `Err` is returned — the original table and pool state are
    /// untouched (spec §4.3: "transactional failure releases only newly
    /// reserved resources").
    ///
    /// [`commit_cow`]: PagePool::commit_cow
    /// [`abort_cow`]: PagePool::abort_cow
    pub fn plan_cow(
        &mut self,
        table: &BlockTable,
        write_start: usize,
        write_end: usize,
    ) -> Result<CowPlan, String> {
        if write_end < write_start {
            return Err("plan_cow: write_end < write_start".to_string());
        }
        if write_end == write_start {
            return Ok(CowPlan {
                copies: Vec::new(),
            });
        }

        let first_lp = write_start / PAGE_TOKENS;
        let last_lp = (write_end - 1) / PAGE_TOKENS;

        let mut copies = Vec::new();
        let mut reserved: Vec<u32> = Vec::new();

        for lp in first_lp..=last_lp {
            let phys = match table.physical(lp) {
                Some(p) => p,
                None => continue, // page not yet allocated — no COW needed
            };
            let state = self.page_meta[phys as usize].state;
            match state {
                PageState::Sealed | PageState::CacheOnly => {
                    // Reserve a private destination.
                    let dst = match self.free_pages.pop() {
                        Some(p) => p,
                        None => {
                            // OOM — rollback all reservations.
                            for &d in &reserved {
                                self.release_reserved(d);
                            }
                            return Err(format!(
                                "plan_cow: OOM — need COW page for logical {} \
                                 (phys {}, state={:?}) but no free pages",
                                lp, phys, state
                            ));
                        }
                    };
                    let dst_meta = &mut self.page_meta[dst as usize];
                    dst_meta.state = PageState::Private;
                    dst_meta.generation = dst_meta.generation.wrapping_add(1);
                    dst_meta.table_refs = 1;

                    // Valid prefix: tokens in this page before write_start.
                    let page_start = lp * PAGE_TOKENS;
                    let valid_prefix_tokens =
                        write_start.saturating_sub(page_start).min(PAGE_TOKENS);

                    copies.push(CowCopy {
                        logical_page: lp,
                        src_phys: phys,
                        dst_phys: dst,
                        valid_prefix_tokens,
                        k_copy_bytes: valid_prefix_tokens * self.k_per_pos_bytes,
                        v_copy_bytes: valid_prefix_tokens * self.v_per_pos_bytes,
                    });
                    reserved.push(dst);
                }
                _ => {} // Private — mutable, no COW needed
            }
        }

        Ok(CowPlan { copies })
    }

    /// Rebind the block table entries per `plan` and adjust refcounts on
    /// the old (Sealed/CacheOnly) pages.  Call this **after** the GPU
    /// memcpys described by the plan have been issued (spec §4.3: "rebind
    /// table only after copy ordering established").
    ///
    /// Old pages whose table refs reach zero transition to Free,
    /// CacheOnly, or ReclaimPending as appropriate.
    pub fn commit_cow(&mut self, table: &mut BlockTable, plan: &CowPlan) -> Result<(), String> {
        for c in plan.copies() {
            // Rebind the table entry to the private copy.
            table.set_page(c.logical_page, c.dst_phys);
            // Decrement the old page's table ref (this table no longer
            // references it).
            self.dec_table_ref(c.src_phys)?;
        }
        Ok(())
    }

    /// Release all pages reserved by `plan_cow` without rebinding the
    /// table.  Call this when the GPU memcpy failed or the step was
    /// cancelled (spec §4.3: "transactional failure releases only newly
    /// reserved resources").
    pub fn abort_cow(&mut self, plan: &CowPlan) {
        for c in plan.copies() {
            self.release_reserved(c.dst_phys);
        }
    }

    // ── Deferred reclaim (spec §4.4 C4) ───────────────────────────────

    /// Add an in-flight lease to `phys`, preventing immediate reuse even
    /// after all table/cache refs are released (spec §4.4: "rc==0 ≠ GPU
    /// completion; free/reuse requires all leases released").
    ///
    /// Returns `Err` if the page is Free or on overflow.
    pub fn add_inflight_ref(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("add_inflight_ref: phys {} is Free", phys));
        }
        meta.inflight_refs = meta
            .inflight_refs
            .checked_add(1)
            .ok_or_else(|| format!("add_inflight_ref: overflow for phys {}", phys))?;
        Ok(())
    }

    /// Release an in-flight lease from `phys`.  The page is NOT freed
    /// here — it remains [`PageState::ReclaimPending`] until
    /// [`drain_completed`] confirms completion (spec §4.4).
    ///
    /// Returns `Err` on underflow.
    ///
    /// [`drain_completed`]: PagePool::drain_completed
    pub fn release_inflight_ref(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        if meta.inflight_refs == 0 {
            return Err(format!(
                "release_inflight_ref: underflow for phys {}",
                phys
            ));
        }
        meta.inflight_refs -= 1;
        Ok(())
    }

    /// Synchronous step-fence drain: free every [`PageState::ReclaimPending`]
    /// page whose in-flight refs have all been released (spec §4.4:
    /// "drain_completed(now/lease-proof)").
    ///
    /// Returns the physical page indices that were freed.
    ///
    /// # HIP-event upgrade seam
    ///
    /// This initial implementation uses a **synchronous fence**: the
    /// caller is expected to have completed a `hipStreamSynchronize` (or
    /// equivalent) before calling `drain_completed`, so any page with
    /// `inflight_refs == 0` is safe to free.  The upgrade path is:
    ///
    /// 1. `add_inflight_ref` records the HIP event/stream associated with
    ///    the device read (stored alongside `inflight_refs` in `PageMeta`).
    /// 2. `drain_completed` queries `hipEventQuery` for each pending page
    ///    and frees only those whose event is signaled — no global sync
    ///    needed.
    /// 3. A `drain_completed_timeout` variant may wait on events with a
    ///    deadline for back-pressure.
    ///
    /// The current API (`add_inflight_ref` / `release_inflight_ref` /
    /// `drain_completed`) is structured so this upgrade is non-breaking:
    /// only the internal completion check changes.
    pub fn drain_completed(&mut self) -> Vec<u32> {
        let mut freed = Vec::new();
        let mut still_pending = Vec::new();
        for &phys in &self.reclaim_pending {
            if self.page_meta[phys as usize].inflight_refs == 0 {
                let meta = &mut self.page_meta[phys as usize];
                meta.state = PageState::Free;
                meta.generation = meta.generation.wrapping_add(1);
                self.free_pages.push(phys);
                freed.push(phys);
            } else {
                still_pending.push(phys);
            }
        }
        self.reclaim_pending = still_pending;
        freed
    }

    // ── Cache leases ──────────────────────────────────────────────────

    /// Add a cache ownership lease to `phys`.  A page with a cache lease
    /// is immutable: if it was Private, it transitions to Sealed (spec
    /// §4.3: "cached pages immutable even with one owner").
    ///
    /// Returns `Err` if the page is Free or on overflow.
    pub fn add_cache_ref(&mut self, phys: u32) -> Result<(), String> {
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("add_cache_ref: phys {} is Free", phys));
        }
        meta.cache_refs = meta
            .cache_refs
            .checked_add(1)
            .ok_or_else(|| format!("add_cache_ref: overflow for phys {}", phys))?;
        if meta.state == PageState::Private {
            meta.state = PageState::Sealed;
        }
        Ok(())
    }

    /// Release a cache ownership lease from `phys`.  If all refs are now
    /// zero, the page is freed (or enters ReclaimPending if in-flight refs
    /// remain).  Returns `Err` on underflow.
    pub fn release_cache_ref(&mut self, phys: u32) -> Result<(), String> {
        let need_free = {
            let meta = &mut self.page_meta[phys as usize];
            if meta.cache_refs == 0 {
                return Err(format!(
                    "release_cache_ref: underflow for phys {}",
                    phys
                ));
            }
            meta.cache_refs -= 1;
            meta.table_refs == 0 && meta.cache_refs == 0 && meta.inflight_refs == 0
        };
        if need_free {
            let meta = &mut self.page_meta[phys as usize];
            meta.state = PageState::Free;
            meta.generation = meta.generation.wrapping_add(1);
            self.free_pages.push(phys);
        }
        Ok(())
    }

    // ── Descriptor construction (unchanged from Wave 1) ───────────────

    /// Build a KvSlotDesc for a session with the given block table.
    /// `block_table_dev_addr` is the GPU address of the uploaded page
    /// index array (i32 per page). The descriptor is in paged mode.
    pub fn make_desc(&self, table: &BlockTable, block_table_dev_addr: u64) -> KvSlotDesc {
        KvSlotDesc {
            block_table: block_table_dev_addr,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len: table.live_tokens() as i32,
            page_tokens: PAGE_TOKENS as i32,
        }
    }

    /// Build a legacy-mode KvSlotDesc (contiguous slab at `base`).
    /// Used for backward compatibility with non-paged code paths.
    pub fn make_legacy_desc(base: u64, seq_len: usize) -> KvSlotDesc {
        KvSlotDesc {
            block_table: 0,
            // Equal K/V bases: this constructor is for equal-stride legacy
            // arenas (Q8_0 / BF16), where one slab offset serves both.
            legacy_k_base: base,
            legacy_v_base: base,
            seq_len: seq_len as i32,
            page_tokens: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests (updated for Wave 2 API) ───────────────────────

    #[test]
    fn alloc_and_release_roundtrip() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        assert_eq!(pool.free_pages(), 16);

        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, 4);
        assert_eq!(allocated, 4);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 12);

        // Each allocated page should have refcount 1 and be Private
        for &phys in table.page_indices() {
            assert_eq!(pool.refcount(phys), 1);
            assert_eq!(pool.page_state(phys), PageState::Private);
        }

        pool.release_table(&mut table).unwrap();
        assert_eq!(table.num_pages(), 0);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn ensure_capacity_allocates_on_demand() {
        let mut pool = PagePool::new(32, 1088).unwrap();
        let mut table = BlockTable::new();

        // 300 tokens needs 3 pages (ceil(300/128) = 3)
        let n = pool.ensure_capacity(&mut table, 300).unwrap();
        assert_eq!(n, 3);
        assert_eq!(table.num_pages(), 3);
        assert_eq!(table.live_tokens(), 300);

        // Growing to 500 tokens needs 4 pages (ceil(500/128) = 4)
        let n = pool.ensure_capacity(&mut table, 500).unwrap();
        assert_eq!(n, 1);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(table.live_tokens(), 500);

        // Shrinking to 200 tokens needs 2 pages — no alloc, just trim live_tokens
        let n = pool.ensure_capacity(&mut table, 200).unwrap();
        assert_eq!(n, 0);
        assert_eq!(table.num_pages(), 4); // pages not freed, just live_tokens reduced
        assert_eq!(table.live_tokens(), 200);
    }

    #[test]
    fn ensure_capacity_oom() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();

        // 4 pages * 128 = 512 tokens max
        let result = pool.ensure_capacity(&mut table, 600);
        assert!(result.is_err());
    }

    #[test]
    fn prefix_sharing_increments_refcount() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(400);

        // Share first 3 pages with table_b
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 3).unwrap();

        assert_eq!(table_b.num_pages(), 3);
        // Shared pages should have refcount 2 and be Sealed
        for lp in 0..3 {
            let phys = table_a.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 2);
            assert_eq!(pool.page_state(phys), PageState::Sealed);
        }
        // Non-shared page should still have refcount 1 and be Private
        let phys_4 = table_a.physical(3).unwrap();
        assert_eq!(pool.refcount(phys_4), 1);
        assert_eq!(pool.page_state(phys_4), PageState::Private);

        // Releasing table_a should free only the non-shared page
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.free_pages(), 16 - 3); // only 1 page freed, 3 still shared
        for lp in 0..3 {
            let phys = table_b.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1);
            // Still Sealed — "cached pages immutable even with one owner"
            assert_eq!(pool.page_state(phys), PageState::Sealed);
        }

        // Releasing table_b frees the remaining shared pages
        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn free_pages_from_tail() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 6);
        table.set_live_tokens(600);

        // Free last 2 pages
        pool.free_pages_from_tail(&mut table, 2).unwrap();
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 16 - 4);
    }

    #[test]
    fn share_prefix_refuses_partially_filled_tail_page() {
        // 3 pages held, but only 300 tokens live: page 2 is 44/128 full.
        // Sharing it without copy-on-write lets src's next append corrupt
        // dst's view of the prefix — must refuse.
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 3);
        src.set_live_tokens(300);

        let mut dst = BlockTable::new();
        let err = pool.share_prefix(&src, &mut dst, 3).unwrap_err();
        assert!(err.contains("partially-filled"), "unexpected: {err}");
        assert_eq!(dst.num_pages(), 0, "refused share must not touch dst");
        for lp in 0..3 {
            let phys = src.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1, "refcounts untouched");
        }
        // Sharing only the full pages is fine.
        pool.share_prefix(&src, &mut dst, 2).unwrap();
        assert_eq!(dst.num_pages(), 2);
    }

    #[test]
    fn share_prefix_refuses_nonempty_dst() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 2);
        src.set_live_tokens(256);

        let mut dst = BlockTable::new();
        pool.alloc_pages(&mut dst, 1);
        let err = pool.share_prefix(&src, &mut dst, 2).unwrap_err();
        assert!(err.contains("empty table"), "unexpected: {err}");
        assert_eq!(dst.num_pages(), 1, "refused share must not touch dst");
        // And the refused attempt must not have bumped refcounts.
        for lp in 0..2 {
            let phys = src.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1);
        }
    }

    #[test]
    fn release_table_after_failed_share_is_balanced() {
        // Edge case: a failed share must leave the pool so that full
        // alloc/release cycles still return every page.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 2);
        src.set_live_tokens(100); // partial tail page
        let mut dst = BlockTable::new();
        assert!(pool.share_prefix(&src, &mut dst, 2).is_err());

        pool.release_table(&mut src).unwrap();
        assert_eq!(pool.free_pages(), 8);
        pool.release_table(&mut dst).unwrap(); // empty — no-op
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn make_desc_paged_mode() {
        let pool = PagePool::new(16, 1088).unwrap();
        let mut table = BlockTable::new();
        // Can't mutate table through immutable pool, so test with manual table
        table.push_page(5);
        table.push_page(12);
        table.set_live_tokens(200);

        let desc = pool.make_desc(&table, 0xDEAD_BEEF);
        assert_eq!(desc.block_table, 0xDEAD_BEEF);
        assert_eq!(desc.legacy_k_base, 0);
        assert_eq!(desc.legacy_v_base, 0);
        assert_eq!(desc.seq_len, 200);
        assert_eq!(desc.page_tokens, PAGE_TOKENS as i32);
    }

    #[test]
    fn make_legacy_desc() {
        let desc = PagePool::make_legacy_desc(0x1000, 42);
        assert_eq!(desc.block_table, 0);
        assert_eq!(desc.legacy_k_base, 0x1000);
        assert_eq!(desc.legacy_v_base, 0x1000);
        assert_eq!(desc.seq_len, 42);
        assert_eq!(desc.page_tokens, 0);
    }

    // ── Wave 2: PageHandle / generation tests ──────────────────────────

    #[test]
    fn alloc_pages_checked_returns_valid_handles() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        let handles = pool.alloc_pages_checked(&mut table, 3).unwrap();
        assert_eq!(handles.len(), 3);
        for h in &handles {
            assert!(pool.validate_handle(h).is_ok());
            assert_eq!(pool.page_state(h.phys), PageState::Private);
        }
        assert_eq!(pool.free_pages(), 5);
    }

    #[test]
    fn stale_handle_after_release_fails_validation() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        let handles = pool.alloc_pages_checked(&mut table, 2).unwrap();

        // Release the table — pages are freed, generations bumped.
        pool.release_table(&mut table).unwrap();

        // All handles should now be stale (page is Free, generation changed).
        for h in &handles {
            assert!(
                pool.validate_handle(h).is_err(),
                "handle must be stale after release"
            );
        }
    }

    #[test]
    fn handle_still_valid_while_page_referenced() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table_a = BlockTable::new();
        let handles_a = pool.alloc_pages_checked(&mut table_a, 2);
        let handles_a = handles_a.unwrap();
        table_a.set_live_tokens(2 * PAGE_TOKENS);

        // Share with table_b
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();

        // Release table_a — shared pages still referenced by table_b
        pool.release_table(&mut table_a).unwrap();

        // Handles for shared pages should still be valid (page not freed,
        // generation unchanged — only table_refs decremented).
        for h in &handles_a {
            assert_eq!(pool.page_state(h.phys), PageState::Sealed);
            assert!(
                pool.validate_handle(h).is_ok(),
                "handle for still-referenced page should be valid"
            );
        }

        // Release table_b — now pages are freed
        pool.release_table(&mut table_b).unwrap();
        for h in &handles_a {
            assert!(
                pool.validate_handle(h).is_err(),
                "handle must be stale after all refs released"
            );
        }
    }

    // ── Wave 2: seal / PageState tests ────────────────────────────────

    #[test]
    fn seal_marks_private_page_as_sealed() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Private);

        pool.seal(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        // Idempotent
        pool.seal(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);
    }

    #[test]
    fn seal_refuses_free_page() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let err = pool.seal(0).unwrap_err();
        assert!(err.contains("Free"));
    }

    #[test]
    fn shared_page_is_sealed_and_stays_sealed_after_partial_release() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 2);
        table_a.set_live_tokens(2 * PAGE_TOKENS);

        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();

        let phys = table_a.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);
        assert_eq!(pool.refcount(phys), 2);

        // Release one table — page still Sealed (immutable even with one owner)
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.refcount(phys), 1);
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Free);
    }

    // ── Wave 2: COW tests (A2, A4) ────────────────────────────────────

    #[test]
    fn a2_branch_share_then_cow_isolation() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        // Branch A: 4 full pages (512 tokens)
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(4 * PAGE_TOKENS);

        // Share first 2 pages with branch B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();
        table_b.set_live_tokens(2 * PAGE_TOKENS);

        // Shared pages are Sealed and have identical physical indices
        for lp in 0..2 {
            let phys = table_a.physical(lp).unwrap();
            assert_eq!(pool.page_state(phys), PageState::Sealed);
            assert_eq!(table_a.physical(lp), table_b.physical(lp));
        }

        // Both branches append: allocate new private pages
        pool.alloc_pages(&mut table_a, 1);
        table_a.set_live_tokens(5 * PAGE_TOKENS);
        pool.alloc_pages(&mut table_b, 1);
        table_b.set_live_tokens(3 * PAGE_TOKENS);

        // Private tails differ and are Private
        assert_ne!(table_a.physical(4), table_b.physical(2));
        assert_eq!(
            pool.page_state(table_a.physical(4).unwrap()),
            PageState::Private
        );
        assert_eq!(
            pool.page_state(table_b.physical(2).unwrap()),
            PageState::Private
        );

        // Branch B rewinds and wants to write page 0 (Sealed → COW needed)
        let plan = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].logical_page, 0);
        assert_eq!(plan.copies()[0].src_phys, table_b.physical(0).unwrap());
        assert_eq!(plan.copies()[0].valid_prefix_tokens, 0); // write from pos 0

        // Commit COW: rebind table B's page 0 to the private copy
        pool.commit_cow(&mut table_b, &plan).unwrap();

        // Branch B's page 0 is now private (different from A's)
        assert_ne!(table_b.physical(0), table_a.physical(0));
        assert_eq!(
            pool.page_state(table_b.physical(0).unwrap()),
            PageState::Private
        );

        // Branch A's page 0 is still the original shared page (unchanged)
        let original_phys = table_a.physical(0).unwrap();
        assert_eq!(pool.page_state(original_phys), PageState::Sealed);
        assert_eq!(pool.refcount(original_phys), 1); // only A references it now

        // Page conservation: 16 total
        // A: [orig0, orig1, orig2, orig3, new_a] = 5 pages
        // B: [cow0, orig1, new_b] = 3 pages
        // Unique: orig0, orig1, orig2, orig3, new_a, cow0, new_b = 7
        // Free: 16 - 7 = 9
        assert_eq!(pool.free_pages(), 9);
    }

    #[test]
    fn cow_plan_with_valid_prefix() {
        // Write starts mid-page: the valid prefix (tokens before write_start
        // in that page) must be copied.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // Seal page 0 so COW is triggered
        pool.seal(table.physical(0).unwrap()).unwrap();

        // Write at positions [64, 128) — page 0 has 64 valid prefix tokens
        let plan = pool.plan_cow(&table, 64, 128).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].valid_prefix_tokens, 64);
        assert_eq!(plan.copies()[0].k_copy_bytes, 64 * 1088);
        assert_eq!(plan.copies()[0].v_copy_bytes, 64 * 1088);
    }

    #[test]
    fn cow_plan_skips_private_pages() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // No pages are Sealed — plan should be empty
        let plan = pool.plan_cow(&table, 0, 2 * PAGE_TOKENS).unwrap();
        assert!(plan.is_empty());
    }

    #[test]
    fn cow_abort_releases_reserved_pages() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);
        pool.seal(table.physical(0).unwrap()).unwrap();

        let free_before = pool.free_pages();
        let plan = pool.plan_cow(&table, 0, PAGE_TOKENS).unwrap();
        assert_eq!(pool.free_pages(), free_before - 1); // one page reserved

        // Abort — reserved page released
        pool.abort_cow(&plan);
        assert_eq!(pool.free_pages(), free_before);

        // Table unchanged
        assert_eq!(table.num_pages(), 2);
    }

    #[test]
    fn a4_cow_oom_leaves_tables_intact() {
        let mut pool = PagePool::new(8, 1088).unwrap();

        // Branch A: 4 full pages
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(4 * PAGE_TOKENS);

        // Share all 4 pages with branch B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 4).unwrap();
        table_b.set_live_tokens(4 * PAGE_TOKENS);

        // Exhaust remaining pages with branch C
        let mut table_c = BlockTable::new();
        pool.alloc_pages(&mut table_c, 4);
        table_c.set_live_tokens(4 * PAGE_TOKENS);
        assert_eq!(pool.free_pages(), 0);

        // COW on table_b fails — no free pages
        let err = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap_err();
        assert!(err.contains("OOM"), "unexpected: {err}");

        // Table B is unchanged (plan_cow doesn't modify the table on failure)
        assert_eq!(table_b.num_pages(), 4);
        for lp in 0..4 {
            assert_eq!(
                table_b.physical(lp),
                table_a.physical(lp),
                "table B must be unchanged after failed COW"
            );
        }

        // Pool state unchanged — no pages leaked
        assert_eq!(pool.free_pages(), 0);

        // Unrelated request: release C, then COW succeeds
        pool.release_table(&mut table_c).unwrap();
        assert_eq!(pool.free_pages(), 4);

        let plan = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap();
        pool.commit_cow(&mut table_b, &plan).unwrap();
        assert_ne!(table_b.physical(0), table_a.physical(0));
        assert_eq!(
            pool.page_state(table_b.physical(0).unwrap()),
            PageState::Private
        );
    }

    // ── Wave 2: checked refcount / underflow tests (A3) ───────────────

    #[test]
    fn a3_release_underflow_detected() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        let phys0 = table.physical(0).unwrap();

        // Release normally
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.refcount(phys0), 0);
        assert_eq!(pool.page_state(phys0), PageState::Free);

        // Construct a stale table referencing the freed page
        let mut stale_table = BlockTable::new();
        stale_table.push_page(phys0);

        // Releasing the stale table must detect underflow
        let result = pool.release_table(&mut stale_table);
        assert!(result.is_err(), "releasing freed page must error");
        assert!(result.unwrap_err().contains("underflow"));
    }

    #[test]
    fn a3_property_page_conservation_and_stale_handle_rejection() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        // Allocate 4 pages for table A
        let mut table_a = BlockTable::new();
        let handles_a = pool.alloc_pages_checked(&mut table_a, 4).unwrap();
        table_a.set_live_tokens(4 * PAGE_TOKENS);
        assert_eq!(pool.free_pages(), 12);

        // Share first 2 pages with table B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();
        assert_eq!(pool.refcount(handles_a[0].phys), 2);
        assert_eq!(pool.page_state(handles_a[0].phys), PageState::Sealed);

        // Page conservation invariant
        let n_alloc: usize = (0..pool.n_pages())
            .filter(|&i| pool.page_state(i as u32) != PageState::Free)
            .count();
        assert_eq!(n_alloc + pool.free_pages(), pool.n_pages());

        // Release table A — shared pages survive, private pages freed
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.page_state(handles_a[0].phys), PageState::Sealed); // shared
        assert_eq!(pool.page_state(handles_a[2].phys), PageState::Free); // freed

        // Stale handle for freed page fails
        assert!(pool.validate_handle(&handles_a[2]).is_err());

        // Handle for still-referenced page is valid
        assert!(pool.validate_handle(&handles_a[0]).is_ok());

        // Release table B — all pages freed
        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.free_pages(), 16);

        // All handles stale
        for h in &handles_a {
            assert!(pool.validate_handle(h).is_err());
        }

        // Final conservation check
        let n_alloc: usize = (0..pool.n_pages())
            .filter(|&i| pool.page_state(i as u32) != PageState::Free)
            .count();
        assert_eq!(n_alloc + pool.free_pages(), pool.n_pages());
        assert_eq!(pool.reclaim_pending_count(), 0);
    }

    #[test]
    fn a3_checked_refcount_no_overflow() {
        let mut pool = PagePool::new(2, 1088).unwrap();
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 1);
        table_a.set_live_tokens(PAGE_TOKENS);

        // Share the same page many times — refcount increments
        let mut tables = Vec::new();
        for _ in 0..10 {
            let mut dst = BlockTable::new();
            pool.share_prefix(&table_a, &mut dst, 1).unwrap();
            tables.push(dst);
        }
        assert_eq!(pool.refcount(table_a.physical(0).unwrap()), 11);

        // Release all — page eventually freed
        pool.release_table(&mut table_a).unwrap();
        for mut t in tables {
            pool.release_table(&mut t).unwrap();
        }
        assert_eq!(pool.free_pages(), 2);
    }

    // ── Wave 2: deferred reclaim tests (A5) ───────────────────────────

    #[test]
    fn a5_reuse_while_reclaim_pending_fails_until_drain() {
        let mut pool = PagePool::new(4, 1088).unwrap();

        // Allocate a page
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        let gen = pool.page_generation(phys);
        assert_eq!(pool.page_state(phys), PageState::Private);

        // Pin with an in-flight lease (simulating a GPU read in progress)
        pool.add_inflight_ref(phys).unwrap();

        // Release the table ref — page goes to ReclaimPending (not Free)
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);
        assert_eq!(pool.free_pages(), 3); // NOT in free list
        assert_eq!(pool.reclaim_pending_count(), 1);

        // Allocation must not reuse the reclaim-pending page
        let mut table2 = BlockTable::new();
        pool.alloc_pages(&mut table2, 1);
        assert_ne!(
            table2.physical(0).unwrap(),
            phys,
            "must not reuse reclaim-pending page"
        );
        assert_eq!(pool.free_pages(), 2);

        // Release the new table
        pool.release_table(&mut table2).unwrap();
        assert_eq!(pool.free_pages(), 3);

        // Release the in-flight ref — page still ReclaimPending
        pool.release_inflight_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);
        assert_eq!(pool.free_pages(), 3);

        // Drain — page freed
        let freed = pool.drain_completed();
        assert_eq!(freed, vec![phys]);
        assert_eq!(pool.page_state(phys), PageState::Free);
        assert_eq!(pool.free_pages(), 4);
        assert_eq!(pool.reclaim_pending_count(), 0);

        // Now the page is allocatable again
        let mut table3 = BlockTable::new();
        pool.alloc_pages(&mut table3, 1);
        assert_eq!(pool.free_pages(), 3);

        // Generation was bumped — old handle is stale
        let new_gen = pool.page_generation(phys);
        assert_ne!(new_gen, gen, "generation must change after free/realloc");
        let old_handle = PageHandle {
            phys,
            epoch: 0,
            generation: gen,
        };
        assert!(
            pool.validate_handle(&old_handle).is_err(),
            "stale handle with old generation must fail"
        );
    }

    #[test]
    fn drain_completed_skips_pages_with_active_inflight() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();

        // Two in-flight refs
        pool.add_inflight_ref(phys).unwrap();
        pool.add_inflight_ref(phys).unwrap();

        // Release table — ReclaimPending
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // Release one in-flight — still pending (one ref remains)
        pool.release_inflight_ref(phys).unwrap();
        let freed = pool.drain_completed();
        assert!(freed.is_empty(), "page with active in-flight must not drain");
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // Release last in-flight — now drainable
        pool.release_inflight_ref(phys).unwrap();
        let freed = pool.drain_completed();
        assert_eq!(freed, vec![phys]);
        assert_eq!(pool.page_state(phys), PageState::Free);
    }

    #[test]
    fn inflight_underflow_detected() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let err = pool.release_inflight_ref(0).unwrap_err();
        assert!(err.contains("underflow"));
    }

    // ── Wave 2: cache lease tests ─────────────────────────────────────

    #[test]
    fn cache_ref_makes_private_page_sealed() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Private);

        pool.add_cache_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        // Release table ref — page goes to CacheOnly (not Free)
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::CacheOnly);
        assert_eq!(pool.free_pages(), 3);

        // Release cache ref — page freed
        pool.release_cache_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Free);
        assert_eq!(pool.free_pages(), 4);
    }

    #[test]
    fn cow_on_cacheonly_page() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // Add cache ref and release table — page becomes CacheOnly
        let phys0 = table.physical(0).unwrap();
        pool.add_cache_ref(phys0).unwrap();
        // Page is now Sealed (Private + cache ref)
        assert_eq!(pool.page_state(phys0), PageState::Sealed);

        // plan_cow should detect Sealed page and plan a copy
        let plan = pool.plan_cow(&table, 0, PAGE_TOKENS).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].src_phys, phys0);

        // Commit COW
        pool.commit_cow(&mut table, &plan).unwrap();
        assert_ne!(table.physical(0).unwrap(), phys0);
    }
}