// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotPool — owns the per-slot KV descriptors that SP1's batched attention
// kernels read.
//
// Two modes:
// - **Legacy** (default): fixed-size per-slot slabs in a contiguous arena.
//   `KvSlotDesc.block_table = 0`, `page_tokens = 0`, `legacy_{k,v}_base` = slab offset.
// - **Paged**: pages allocated on demand from a shared `PagePool`. Each slot
//   owns a `BlockTable` (logical→physical page mapping). `KvSlotDesc.block_table`
//   points to the GPU-uploaded page index array, `page_tokens = PAGE_TOKENS`.
//   Enables dynamic allocation, prefix sharing, and over-subscription.

use crate::kv_slots::{preflight_alloc, KvSlotDesc, R9700_VRAM_BYTES};
use crate::page_pool::{BlockTable, PagePool, PAGE_TOKENS as POOL_PAGE_TOKENS};

/// Slab capacities round up to this, so a future page size divides them.
/// Matches the tile size the flash path walks KV in.
const PAGE_TOKENS: usize = 128;

/// Index of a slot within its pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotId(pub usize);

// `Debug` is required so tests can call `.unwrap_err()` on
// `Result<SlotPool, String>` (`unwrap_err` requires `T: Debug` because it
// formats the Ok value into the panic message on the failure path).
#[derive(Debug)]
pub struct SlotPool {
    descs: Vec<KvSlotDesc>,
    in_use: Vec<bool>,
    cap_tokens: usize,
    /// Per-position strides in bytes. These DIFFER on the rotated-K tiers
    /// (asym{2,3,4}/fwht{2,3,4} store K packed per head + a 4-byte header,
    /// V at Q8_0) and are equal on q8/bf16. Every slab offset in the K
    /// arena scales by the K stride, the V arena by the V stride; the
    /// descriptors carry the two separately (`legacy_k_base`/`legacy_v_base`).
    k_per_pos_bytes: usize,
    v_per_pos_bytes: usize,
    dirty: bool,
    // ── Paged mode (None = legacy) ──────────────────────────────────────
    page_pool: Option<PagePool>,
    block_tables: Vec<Option<BlockTable>>,
    /// Per-slot dirty flag: block table changed since last upload.
    block_tables_dirty: Vec<bool>,
}

/// True when this pool is operating in paged mode.
fn _assert_page_tokens_match() {
    // Compile-time check that both constants agree.
    const _: () = assert!(POOL_PAGE_TOKENS == PAGE_TOKENS);
}

impl SlotPool {
    /// Build a pool of `n_slots` fixed-size slabs.
    ///
    /// `per_pos_bytes` is the per-position stride, uniform across slots
    /// (`n_kv_heads * (head_dim/32) * 34` for Q8_0) and shared by the K and
    /// V arenas. For the rotated-K tiers call [`SlotPool::new_with_strides`]
    /// instead — their K and V strides differ.
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(n_slots: usize, cap_tokens: usize, per_pos_bytes: usize) -> Result<Self, String> {
        Self::new_with_strides(n_slots, cap_tokens, per_pos_bytes, per_pos_bytes)
    }

    /// [`SlotPool::new`] with independent K/V strides — the form the
    /// rotated-K tiers need. The K arena holds `n_slots` slabs of
    /// `cap × k_per_pos_bytes`; the V arena the same at `v_per_pos_bytes`.
    /// Descriptors carry matching `legacy_k_base`/`legacy_v_base` so every
    /// kernel resolves each arena through its own stride.
    pub fn new_with_strides(
        n_slots: usize,
        cap_tokens: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(k_per_pos_bytes > 0, "k_per_pos_bytes must be positive");
        assert!(v_per_pos_bytes > 0, "v_per_pos_bytes must be positive");
        let cap = cap_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let k_slab_bytes = (cap * k_per_pos_bytes) as u64;
        let v_slab_bytes = (cap * v_per_pos_bytes) as u64;
        let total = k_slab_bytes
            .checked_add(v_slab_bytes)
            .ok_or_else(|| "SlotPool: arena size overflows u64".to_string())?
            .checked_mul(n_slots as u64)
            .ok_or_else(|| "SlotPool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "SlotPool arena")?;

        let descs = (0..n_slots)
            .map(|i| {
                let k_base = i as u64 * k_slab_bytes;
                let v_base = i as u64 * v_slab_bytes;
                KvSlotDesc {
                    // Legacy contiguous mode: block_table = 0, page_tokens = 0.
                    block_table: 0,
                    legacy_k_base: k_base,
                    legacy_v_base: v_base,
                    seq_len: 0,
                    page_tokens: 0,
                }
            })
            .collect();

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: cap,
            k_per_pos_bytes,
            v_per_pos_bytes,
            dirty: true,
            page_pool: None,
            block_tables: vec![None; n_slots],
            block_tables_dirty: vec![false; n_slots],
        })
    }

    /// Build a paged pool with `n_slots` slots drawing from a shared arena
    /// of `n_pages` physical pages. Each slot's KV is allocated on demand
    /// via the `PagePool`, enabling dynamic allocation, prefix sharing,
    /// and over-subscription.
    ///
    /// `cap_tokens` is the per-slot maximum (admission limit), not a
    /// pre-allocation. The actual arena is `n_pages * PAGE_TOKENS *
    /// per_pos_bytes` bytes per layer (K or V). K and V share the stride.
    pub fn new_paged(
        n_slots: usize,
        cap_tokens: usize,
        per_pos_bytes: usize,
        n_pages: usize,
    ) -> Result<Self, String> {
        Self::new_paged_with_strides(n_slots, cap_tokens, per_pos_bytes, per_pos_bytes, n_pages)
    }

    /// [`SlotPool::new_paged`] with independent K/V strides — one block
    /// table still maps both arenas (a logical page resolves to the same
    /// physical page for K and V), but each arena's page is
    /// `PAGE_TOKENS ×` its own stride.
    pub fn new_paged_with_strides(
        n_slots: usize,
        cap_tokens: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
        n_pages: usize,
    ) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(k_per_pos_bytes > 0, "k_per_pos_bytes must be positive");
        assert!(v_per_pos_bytes > 0, "v_per_pos_bytes must be positive");
        let cap = cap_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let page_pool = PagePool::new_with_strides(n_pages, k_per_pos_bytes, v_per_pos_bytes)?;

        // In paged mode, descriptors start in legacy mode (block_table = 0).
        // They switch to paged mode when the slot acquires pages and the
        // block table is uploaded to the GPU.
        let descs = (0..n_slots)
            .map(|_| KvSlotDesc {
                block_table: 0,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: 0,
                page_tokens: 0,
            })
            .collect();

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: cap,
            k_per_pos_bytes,
            v_per_pos_bytes,
            dirty: true,
            page_pool: Some(page_pool),
            block_tables: vec![None; n_slots],
            block_tables_dirty: vec![false; n_slots],
        })
    }

    /// Take a free slot, or `None` when the pool is full. Admission control
    /// lives in SP4; this only reports capacity.
    ///
    /// In paged mode, a fresh `BlockTable` is created for the slot.
    pub fn acquire(&mut self) -> Option<SlotId> {
        let i = self.in_use.iter().position(|&u| !u)?;
        self.in_use[i] = true;
        self.reset(SlotId(i));
        // In paged mode, give the slot a fresh block table.
        if self.page_pool.is_some() {
            self.block_tables[i] = Some(BlockTable::new());
            self.block_tables_dirty[i] = true;
        }
        Some(SlotId(i))
    }

    /// Return a slot to the pool. Resets its length so a later `acquire`
    /// cannot inherit the previous occupant's history.
    ///
    /// In paged mode, the slot's pages are freed back to the `PagePool`.
    pub fn release(&mut self, id: SlotId) {
        self.reset(id);
        // In paged mode, free the block table's pages.
        if let Some(pool) = self.page_pool.as_mut() {
            if let Some(bt) = self.block_tables[id.0].as_mut() {
                pool.release_table(bt);
            }
            self.block_tables[id.0] = None;
            self.block_tables_dirty[id.0] = false;
        }
        self.in_use[id.0] = false;
    }

    /// Zero a slot's logical length. The slab bytes are left alone — every
    /// read is bounded by `seq_len`, so stale bytes are unreachable.
    ///
    /// In paged mode, this truncates the block table to 0 pages (freeing
    /// all pages) and resets the descriptor to legacy mode.
    pub fn reset(&mut self, id: SlotId) {
        if self.descs[id.0].seq_len != 0 {
            self.descs[id.0].seq_len = 0;
            self.dirty = true;
        }
        if let Some(pool) = self.page_pool.as_mut() {
            if let Some(bt) = self.block_tables[id.0].as_mut() {
                if bt.num_pages() > 0 {
                    pool.release_table(bt);
                    self.block_tables_dirty[id.0] = true;
                }
            }
            // Reset descriptor to legacy mode until pages are allocated.
            self.descs[id.0].block_table = 0;
            self.descs[id.0].page_tokens = 0;
            self.descs[id.0].legacy_k_base = 0;
            self.descs[id.0].legacy_v_base = 0;
        }
    }

    /// Set a slot's logical KV length. Enforces `seq_len <= cap` host-side,
    /// because SP1 removed the device asserts (they shipped in release and
    /// cost 64 B/lane of scratch).
    ///
    /// In paged mode, this ensures the block table has enough pages to hold
    /// `seq_len` tokens, allocating new pages from the `PagePool` as needed.
    /// The descriptor's `seq_len` is updated; the block table GPU upload and
    /// descriptor paged-mode activation happen in the forward path's
    /// `upload_block_tables`.
    pub fn set_seq_len(&mut self, id: SlotId, seq_len: usize) -> Result<(), String> {
        if seq_len > self.cap_tokens {
            return Err(format!(
                "SlotPool: slot {} seq_len {} exceeds cap {}",
                id.0, seq_len, self.cap_tokens
            ));
        }
        if let Some(pool) = self.page_pool.as_mut() {
            let bt = self
                .block_tables[id.0]
                .as_mut()
                .ok_or_else(|| format!("SlotPool: slot {} has no block table", id.0))?;
            pool.ensure_capacity(bt, seq_len)?;
            self.block_tables_dirty[id.0] = true;
        }
        if self.descs[id.0].seq_len != seq_len as i32 {
            self.descs[id.0].seq_len = seq_len as i32;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn descriptors(&self) -> &[KvSlotDesc] {
        &self.descs
    }

    /// True when the descriptor table or any block table has changed since
    /// the last `mark_uploaded`. Callers skip the device upload when clean.
    pub fn descriptors_dirty(&self) -> bool {
        self.dirty || self.block_tables_dirty.iter().any(|&d| d)
    }

    pub fn mark_uploaded(&mut self) {
        self.dirty = false;
        self.block_tables_dirty.iter_mut().for_each(|d| *d = false);
    }

    /// Per-slot token capacity (uniform across all slots).
    pub fn cap_tokens(&self) -> usize {
        self.cap_tokens
    }

    /// Per-position stride in bytes. Equal-stride pools (q8/bf16) only —
    /// returns the shared stride. On rotated-K tiers use
    /// [`SlotPool::k_per_pos_bytes`] / [`SlotPool::v_per_pos_bytes`].
    pub fn per_pos_bytes(&self) -> usize {
        debug_assert_eq!(
            self.k_per_pos_bytes, self.v_per_pos_bytes,
            "per_pos_bytes() is the EQUAL-stride accessor; this pool carries \
             distinct K/V strides — use k_/v_per_pos_bytes()"
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

    /// Bytes in ONE arena (K or V) for an equal-stride pool. The pool holds
    /// two of these. Rotated-K tiers must use [`SlotPool::k_arena_bytes`] /
    /// [`SlotPool::v_arena_bytes`] instead.
    ///
    /// In legacy mode: `n_slots * cap_tokens * per_pos_bytes`.
    /// In paged mode: `n_pages * PAGE_TOKENS * per_pos_bytes`.
    pub fn arena_bytes(&self) -> usize {
        self.k_arena_bytes()
    }

    /// Bytes of the K arena.
    pub fn k_arena_bytes(&self) -> usize {
        if let Some(pool) = &self.page_pool {
            pool.k_arena_bytes()
        } else {
            self.descs.len() * self.cap_tokens * self.k_per_pos_bytes
        }
    }

    /// Bytes of the V arena.
    pub fn v_arena_bytes(&self) -> usize {
        if let Some(pool) = &self.page_pool {
            pool.v_arena_bytes()
        } else {
            self.descs.len() * self.cap_tokens * self.v_per_pos_bytes
        }
    }

    // ── Paged-mode accessors ───────────────────────────────────────────

    /// True when this pool is operating in paged mode.
    pub fn is_paged(&self) -> bool {
        self.page_pool.is_some()
    }

    /// Get the block table for `slot`, if in paged mode.
    pub fn block_table(&self, slot: SlotId) -> Option<&BlockTable> {
        self.block_tables.get(slot.0).and_then(|bt| bt.as_ref())
    }

    /// True when `slot`'s block table has changed since last upload.
    pub fn block_table_dirty(&self, slot: SlotId) -> bool {
        self.block_tables_dirty.get(slot.0).copied().unwrap_or(false)
    }

    /// Activate paged mode for `slot`: set the descriptor's `block_table`
    /// to the GPU device address of the uploaded page index array, and
    /// `page_tokens` to `PAGE_TOKENS`. Called by `forward_batch_slots`
    /// after uploading the block table to the GPU.
    pub fn activate_paged_desc(&mut self, slot: SlotId, block_table_dev_addr: u64) {
        let bt = self
            .block_tables
            .get(slot.0)
            .and_then(|bt| bt.as_ref())
            .expect("activate_paged_desc: slot has no block table");
        self.descs[slot.0].block_table = block_table_dev_addr;
        self.descs[slot.0].page_tokens = PAGE_TOKENS as i32;
        self.descs[slot.0].legacy_k_base = 0;
        self.descs[slot.0].legacy_v_base = 0;
        self.descs[slot.0].seq_len = bt.live_tokens() as i32;
        self.dirty = true;
    }

    /// Share a prefix of `n_pages` pages from `src` slot into `dst` slot's
    /// block table. Used for prefix sharing across sessions.
    ///
    /// Delegates the refcounting AND the safety contract to
    /// [`PagePool::share_prefix`]: sharing is only defined at a page-aligned
    /// fork point into an EMPTY dst whose shared pages are all full in src —
    /// there is no copy-on-write, so violating that lets two sessions append
    /// into the same physical page with no error raised. After sharing, set
    /// dst's live length to `n_pages * PAGE_TOKENS` (its fork point); further
    /// growth allocates fresh pages only.
    pub fn share_prefix(
        &mut self,
        src: SlotId,
        dst: SlotId,
        n_pages: usize,
    ) -> Result<(), String> {
        // Snapshot src's table (a small Vec<u32>) so the PagePool call sees a
        // stable copy instead of fighting the field borrows.
        let src_snap = self
            .block_tables
            .get(src.0)
            .and_then(|bt| bt.as_ref())
            .ok_or_else(|| "share_prefix: src slot has no block table".to_string())?
            .snapshot();
        let dst_bt = self
            .block_tables
            .get_mut(dst.0)
            .and_then(|bt| bt.as_mut())
            .ok_or_else(|| "share_prefix: dst slot has no block table".to_string())?;
        let pool = self
            .page_pool
            .as_mut()
            .ok_or_else(|| "share_prefix: pool is not in paged mode".to_string())?;
        pool.share_prefix(&src_snap, dst_bt, n_pages)?;
        self.block_tables_dirty[dst.0] = true;
        self.dirty = true;
        Ok(())
    }

    /// Number of free pages in the page pool (paged mode only).
    pub fn free_pages(&self) -> usize {
        self.page_pool.as_ref().map(|p| p.free_pages()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PPB: usize = 1088; // Q8_0 bytes/position at n_kv_heads=2, head_dim=256

    #[test]
    fn slabs_are_cap_aligned_and_non_overlapping() {
        let p = SlotPool::new(4, 300, PPB).unwrap();
        let d = p.descriptors();
        assert_eq!(d.len(), 4);
        // cap rounds up to a multiple of PAGE_TOKENS (128) so a future page size divides it
        assert_eq!(p.cap_tokens(), 384);
        for i in 1..4 {
            assert_eq!(d[i - 1].legacy_k_base, d[i - 1].legacy_v_base);
            let prev_end = d[i - 1].legacy_k_base + (p.cap_tokens() as u64) * PPB as u64;
            assert_eq!(
                d[i].legacy_k_base,
                prev_end,
                "slab {i} must start where {} ended",
                i - 1
            );
        }
    }

    #[test]
    fn q8_abi_uses_shared_legacy_base() {
        // Q8_0 ABI: the flash-prefill kernel uses ONE shared slab offset.
        // The Q8 pool honours it by keeping legacy_k_base == legacy_v_base.
        let p = SlotPool::new(3, 256, PPB).unwrap();
        for d in p.descriptors() {
            assert_eq!(d.block_table, 0, "SlotPool must use legacy mode");
            assert_eq!(d.page_tokens, 0, "SlotPool must use legacy mode");
        }
    }

    #[test]
    fn acquire_release_reuses_slots_and_bounds_count() {
        let mut p = SlotPool::new(2, 128, PPB).unwrap();
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        assert!(p.acquire().is_none(), "pool of 2 must not hand out a third");
        p.release(a);
        let c = p.acquire().unwrap();
        assert_eq!(c.0, a.0, "released slot must be reused");
        p.release(b);
        p.release(c);
    }

    #[test]
    fn set_seq_len_enforces_the_cap_invariant() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        let id = p.acquire().unwrap();
        assert!(p.set_seq_len(id, 128).is_ok());
        let e = p.set_seq_len(id, 129).unwrap_err();
        assert!(e.contains("cap"), "unexpected message: {e}");
    }

    #[test]
    fn release_resets_seq_len_so_reuse_cannot_inherit_history() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        let id = p.acquire().unwrap();
        p.set_seq_len(id, 100).unwrap();
        p.release(id);
        let id2 = p.acquire().unwrap();
        assert_eq!(
            p.descriptors()[id2.0].seq_len,
            0,
            "reused slot must start empty"
        );
    }

    #[test]
    fn dirty_flag_tracks_descriptor_changes() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
        let id = p.acquire().unwrap();
        p.set_seq_len(id, 10).unwrap();
        assert!(
            p.descriptors_dirty(),
            "a seq_len change must dirty the table"
        );
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
    }

    #[test]
    fn oversized_pool_is_refused_not_allocated() {
        // 8 slots x 4M tokens x 1088 B/pos x 2 (K and V) = ~69.6 GB, over the
        // 32 GiB target budget.
        //
        // Was 1M tokens, which is ~17.4 GB once K and V are counted -- under
        // the budget, so `new` correctly returned Ok and the `unwrap_err` here
        // panicked. The test's comment said "8.7 TB", off by 1000x; the
        // refusal it is checking was never actually being exercised.
        let e = SlotPool::new(8, 4_000_000, PPB).unwrap_err();
        assert!(e.contains("budget") || e.contains("GiB"), "unexpected: {e}");
    }

    // ── Rotated-K tier stride tests (asym3 is the standing case) ──────

    const K_PPB_ASYM3: usize = 400; // n_kv_heads=1, head_dim=256: 4 + 256*3/8
    const V_PPB_Q8: usize = 1088; // n_kv_heads=1, head_dim=256: 8 * 34

    #[test]
    fn stride_split_legacy_desc_bases_follow_each_arena() {
        // Slot 1's K slab starts where slot 0's K slab ENDED (400 B/pos),
        // likewise V (1088 B/pos) — the bases must NOT share one stride.
        let p = SlotPool::new_with_strides(3, 128, K_PPB_ASYM3, V_PPB_Q8).unwrap();
        let d = p.descriptors();
        assert_eq!(d[0].legacy_k_base, 0);
        assert_eq!(d[1].legacy_k_base, (128 * K_PPB_ASYM3) as u64);
        assert_eq!(d[2].legacy_k_base, 2 * (128 * K_PPB_ASYM3) as u64);
        assert_eq!(d[0].legacy_v_base, 0);
        assert_eq!(d[1].legacy_v_base, (128 * V_PPB_Q8) as u64);
        assert_eq!(d[2].legacy_v_base, 2 * (128 * V_PPB_Q8) as u64);
        assert_ne!(d[1].legacy_k_base, d[1].legacy_v_base);
        assert_eq!(p.k_arena_bytes(), 3 * 128 * K_PPB_ASYM3);
        assert_eq!(p.v_arena_bytes(), 3 * 128 * V_PPB_Q8);
        assert_eq!(p.k_per_pos_bytes(), K_PPB_ASYM3);
        assert_eq!(p.v_per_pos_bytes(), V_PPB_Q8);
    }

    #[test]
    fn stride_split_equal_strides_match_legacy_ctor() {
        // Equal strides must produce exactly what `new` always produced.
        let a = SlotPool::new(2, 256, PPB).unwrap();
        let b = SlotPool::new_with_strides(2, 256, PPB, PPB).unwrap();
        assert_eq!(a.descriptors(), b.descriptors());
        assert_eq!(a.arena_bytes(), b.arena_bytes());
    }

    #[test]
    fn stride_split_paged_sized_per_arena() {
        let p = SlotPool::new_paged_with_strides(2, 256, K_PPB_ASYM3, V_PPB_Q8, 8).unwrap();
        // 8 pages * 128 tokens * stride, per arena.
        assert_eq!(p.k_arena_bytes(), 8 * 128 * K_PPB_ASYM3);
        assert_eq!(p.v_arena_bytes(), 8 * 128 * V_PPB_Q8);
        // Provisioning is stride-agnostic.
        let mut p = p;
        let slot = p.acquire().unwrap();
        p.set_seq_len(slot, 256).unwrap();
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 2);
    }

    // ── Paged mode tests ──────────────────────────────────────────────

    #[test]
    fn paged_pool_acquires_with_fresh_block_table() {
        let mut p = SlotPool::new_paged(2, 256, PPB, 16).unwrap();
        assert!(p.is_paged());
        let slot = p.acquire().unwrap();
        assert!(p.block_table(slot).is_some(), "acquired slot must have a block table");
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 0);
        assert_eq!(p.block_table(slot).unwrap().live_tokens(), 0);
    }

    #[test]
    fn paged_set_seq_len_allocates_pages() {
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let slot = p.acquire().unwrap();
        // 300 tokens needs 3 pages (ceil(300/128) = 3)
        p.set_seq_len(slot, 300).unwrap();
        let bt = p.block_table(slot).unwrap();
        assert_eq!(bt.num_pages(), 3);
        assert_eq!(bt.live_tokens(), 300);
        assert_eq!(p.descriptors()[slot.0].seq_len, 300);
        // Free pages should be 16 - 3 = 13
        assert_eq!(p.free_pages(), 13);
    }

    #[test]
    fn paged_set_seq_len_grows_and_shrinks() {
        let mut p = SlotPool::new_paged(1, 1024, PPB, 32).unwrap();
        let slot = p.acquire().unwrap();
        // Grow to 500 tokens (4 pages)
        p.set_seq_len(slot, 500).unwrap();
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 4);
        // Grow to 800 tokens (7 pages)
        p.set_seq_len(slot, 800).unwrap();
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 7);
        // Shrink to 200 tokens — pages stay allocated, live_tokens reduced
        p.set_seq_len(slot, 200).unwrap();
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 7);
        assert_eq!(p.block_table(slot).unwrap().live_tokens(), 200);
    }
    #[test]
    fn paged_release_frees_pages() {
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let slot = p.acquire().unwrap();
        p.set_seq_len(slot, 300).unwrap();
        assert_eq!(p.free_pages(), 13);
        p.release(slot);
        assert_eq!(p.free_pages(), 16, "release must free all pages");
        assert!(p.block_table(slot).is_none(), "release must clear block table");
    }

    #[test]
    fn paged_oom_when_pages_exhausted() {
        let mut p = SlotPool::new_paged(1, 4096, PPB, 4).unwrap();
        let slot = p.acquire().unwrap();
        // 4 pages * 128 = 512 tokens max
        p.set_seq_len(slot, 512).unwrap();
        // 513 tokens needs 5 pages — only 4 available
        let err = p.set_seq_len(slot, 513).unwrap_err();
        assert!(err.contains("free"), "unexpected: {err}");
    }

    #[test]
    fn paged_share_prefix_between_slots() {
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let slot_a = p.acquire().unwrap();
        p.set_seq_len(slot_a, 300).unwrap(); // 3 pages
        let slot_b = p.acquire().unwrap();
        // Share first 2 pages from A to B
        p.share_prefix(slot_a, slot_b, 2).unwrap();
        let bt_b = p.block_table(slot_b).unwrap();
        assert_eq!(bt_b.num_pages(), 2);
        // Shared pages should have the same physical indices
        let bt_a = p.block_table(slot_a).unwrap();
        assert_eq!(bt_a.physical(0), bt_b.physical(0));
        assert_eq!(bt_a.physical(1), bt_b.physical(1));
        // Free pages: 16 - 3 (A) - 0 (B new) = 13 (B shares A's pages)
        assert_eq!(p.free_pages(), 13);
        // Release A — shared pages should survive
        p.release(slot_a);
        assert_eq!(p.free_pages(), 16 - 2, "only non-shared page freed");
        // Release B — now all pages freed
        p.release(slot_b);
        assert_eq!(p.free_pages(), 16);
    }

    #[test]
    fn paged_activate_paged_desc_sets_block_table_addr() {
        let mut p = SlotPool::new_paged(1, 256, PPB, 8).unwrap();
        let slot = p.acquire().unwrap();
        p.set_seq_len(slot, 128).unwrap();
        // Simulate GPU upload: activate paged descriptor with a fake address
        p.activate_paged_desc(slot, 0xCAFE_BABE);
        let desc = p.descriptors()[slot.0];
        assert_eq!(desc.block_table, 0xCAFE_BABE);
        assert_eq!(desc.page_tokens, PAGE_TOKENS as i32);
        assert_eq!(desc.legacy_k_base, 0);
        assert_eq!(desc.legacy_v_base, 0);
        assert_eq!(desc.seq_len, 128);
    }

    #[test]
    fn paged_release_resets_to_legacy_mode() {
        let mut p = SlotPool::new_paged(1, 256, PPB, 8).unwrap();
        let slot = p.acquire().unwrap();
        p.set_seq_len(slot, 128).unwrap();
        p.activate_paged_desc(slot, 0xCAFE_BABE);
        p.release(slot);
        // After release, descriptor should be back in legacy mode
        let desc = p.descriptors()[slot.0];
        assert_eq!(desc.block_table, 0);
        assert_eq!(desc.page_tokens, 0);
        assert_eq!(desc.seq_len, 0);
    }

    #[test]
    fn paged_dirty_flag_tracks_block_table_changes() {
        let mut p = SlotPool::new_paged(1, 256, PPB, 8).unwrap();
        let slot = p.acquire().unwrap();
        p.mark_uploaded();
        // set_seq_len should dirty the block table
        p.set_seq_len(slot, 128).unwrap();
        assert!(p.descriptors_dirty(), "block table change must dirty");
        assert!(p.block_table_dirty(slot), "slot block table must be dirty");
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
        assert!(!p.block_table_dirty(slot));
    }

    #[test]
    fn paged_arena_bytes_uses_page_pool() {
        let p = SlotPool::new_paged(2, 256, PPB, 16).unwrap();
        // 16 pages * 128 tokens * 1088 bytes = 2,228,224 bytes per arena
        assert_eq!(p.arena_bytes(), 16 * 128 * PPB);
    }

    // ── Paged-mode regression tests (branch bug fixes) ─────────────────

    #[test]
    fn paged_share_prefix_refuses_partially_filled_page() {
        // 300 live tokens = 3 pages, but page 2 holds only 44 tokens.
        // Sharing all 3 would let src's next append land inside a page dst
        // is also reading — the missing-copy-on-write corruption. Must err.
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        p.set_seq_len(a, 300).unwrap();
        let err = p.share_prefix(a, b, 3).unwrap_err();
        assert!(
            err.contains("partially-filled"),
            "unexpected: {err}"
        );
        // The refused share must not have mutated dst or the refcounts.
        assert_eq!(p.block_table(b).unwrap().num_pages(), 0);
        assert_eq!(p.free_pages(), 13);
    }

    #[test]
    fn paged_share_prefix_at_exact_page_boundary_is_allowed() {
        // 256 live tokens = exactly 2 full pages: the one safe fork point.
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        p.set_seq_len(a, 256).unwrap();
        p.share_prefix(a, b, 2).unwrap();
        assert_eq!(p.block_table(b).unwrap().num_pages(), 2);
        // Fork dst's live length to the fork point, as the caller must.
        p.set_seq_len(b, 256).unwrap();
        // Both sessions append; shared full pages are immutable so neither
        // can corrupt the other.
        p.set_seq_len(a, 300).unwrap();
        p.set_seq_len(b, 400).unwrap();
        let a_pages = p.block_table(a).unwrap().page_indices().to_vec();
        let b_pages = p.block_table(b).unwrap().page_indices().to_vec();
        assert_eq!(&a_pages[..2], &b_pages[..2], "prefix stays shared");
        assert_ne!(a_pages[2], b_pages[2], "growth must be private");
        assert_eq!(a_pages.len(), 3);
        assert_eq!(b_pages.len(), 4);
    }

    #[test]
    fn paged_share_prefix_refuses_nonempty_dst() {
        let mut p = SlotPool::new_paged(2, 512, PPB, 16).unwrap();
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        p.set_seq_len(a, 256).unwrap();
        p.set_seq_len(b, 128).unwrap();
        let err = p.share_prefix(a, b, 2).unwrap_err();
        assert!(err.contains("empty table"), "unexpected: {err}");
        // Refused share must not touch dst's mapping.
        assert_eq!(p.block_table(b).unwrap().num_pages(), 1);
    }

    #[test]
    fn paged_write_frontier_provisioning_keeps_table_covered() {
        // The forward path provisions set_seq_len(last_pos + 1) BEFORE the
        // step's KV write; advance_slot_seq_lens then sets the same value.
        // Simulate the engine loop across page boundaries and verify every
        // position the "kernel" would translate is inside the allocated
        // table — before provisioning existed, a step writing at pos 128
        // indexed block_table[1] of a one-entry table.
        let mut p = SlotPool::new_paged(1, 4096, PPB, 8).unwrap();
        let slot = p.acquire().unwrap();
        let mut kv_len = 0usize;
        for m in [1usize, 127, 1, 5, 128, 3] {
            let last_pos = kv_len + m - 1;
            // Provision (what forward now does before its kernels).
            p.set_seq_len(slot, last_pos + 1).unwrap();
            let bt = p.block_table(slot).unwrap();
            let needed_pages = (last_pos + 1).div_ceil(PAGE_TOKENS);
            assert!(
                bt.num_pages() >= needed_pages,
                "step writing positions [{}, {}] needs {needed_pages} pages, \
                 table holds {}",
                kv_len,
                last_pos,
                bt.num_pages()
            );
            for lp in 0..=last_pos / PAGE_TOKENS {
                assert!(
                    bt.physical(lp).is_some(),
                    "position {lp:?} page {lp} missing from table"
                );
            }
            // Post-step advance: same value, must stay idempotent.
            p.set_seq_len(slot, last_pos + 1).unwrap();
            assert_eq!(p.block_table(slot).unwrap().num_pages(), needed_pages);
            kv_len = last_pos + 1;
        }
    }

    #[test]
    fn paged_provision_oom_fails_before_any_write() {
        // 2 pages = 256 tokens. A step whose write frontier reaches token
        // 256 must fail AT PROVISION TIME — before any KV is written — not
        // corrupt a neighbour page.
        let mut p = SlotPool::new_paged(1, 4096, PPB, 2).unwrap();
        let slot = p.acquire().unwrap();
        p.set_seq_len(slot, 128).unwrap(); // page 0 live
        let err = p.set_seq_len(slot, 257).unwrap_err();
        assert!(err.contains("free"), "unexpected: {err}");
        // The refused provision leaves the table and length untouched.
        assert_eq!(p.block_table(slot).unwrap().num_pages(), 1);
        assert_eq!(p.descriptors()[slot.0].seq_len, 128);
    }
}
