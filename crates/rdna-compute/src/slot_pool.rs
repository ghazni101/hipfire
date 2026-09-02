// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotPool — owns the per-slot KV descriptors that SP1's batched attention
// kernels read.
//
// Two modes:
// - **Legacy** (default): fixed-size per-slot slabs in a contiguous arena.
//   `KvSlotDesc.block_table = 0`, `page_tokens = 0`, `legacy_base` = slab offset.
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
    per_pos_bytes: usize,
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
    /// (`n_kv_heads * (head_dim/32) * 34` for Q8_0).
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(n_slots: usize, cap_tokens: usize, per_pos_bytes: usize) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(per_pos_bytes > 0, "per_pos_bytes must be positive");
        let cap = cap_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let slab_bytes = (cap * per_pos_bytes) as u64;
        // K and V are separate arenas of identical layout, hence x2.
        let total = slab_bytes
            .checked_mul(n_slots as u64)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| "SlotPool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "SlotPool arena")?;

        let descs = (0..n_slots)
            .map(|i| {
                let base = i as u64 * slab_bytes;
                KvSlotDesc {
                    // Legacy contiguous mode: block_table = 0, page_tokens = 0.
                    // Q8_0 ABI: legacy_base serves as both k_base and v_base.
                    block_table: 0,
                    legacy_base: base,
                    seq_len: 0,
                    page_tokens: 0,
                }
            })
            .collect();

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: cap,
            per_pos_bytes,
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
    /// per_pos_bytes` bytes per layer (K or V).
    pub fn new_paged(
        n_slots: usize,
        cap_tokens: usize,
        per_pos_bytes: usize,
        n_pages: usize,
    ) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(per_pos_bytes > 0, "per_pos_bytes must be positive");
        let cap = cap_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let page_pool = PagePool::new(n_pages, per_pos_bytes)?;

        // In paged mode, descriptors start in legacy mode (block_table = 0).
        // They switch to paged mode when the slot acquires pages and the
        // block table is uploaded to the GPU.
        let descs = (0..n_slots)
            .map(|_| KvSlotDesc {
                block_table: 0,
                legacy_base: 0,
                seq_len: 0,
                page_tokens: 0,
            })
            .collect();

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: cap,
            per_pos_bytes,
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
            self.descs[id.0].legacy_base = 0;
        }
    }

    /// Set a slot's logical KV length. Enforces `seq_len <= cap` host-side,
    /// because SP1 removed the device asserts (they shipped in release and
    /// cost 64 B/lane of scratch).
    ///
    /// In paged mode, this ensures the block table has enough pages to hold
    /// `seq_len` tokens, allocating new pages from the `PagePool` as needed.
    /// The descriptor's `seq_len` is updated; the block table GPU upload and
    /// descriptor paged-mode activation happen in `mark_block_table_uploaded`.
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

    /// Per-position stride in bytes.
    pub fn per_pos_bytes(&self) -> usize {
        self.per_pos_bytes
    }

    /// Bytes in ONE arena (K or V). The pool holds two of these.
    /// In legacy mode: `n_slots * cap_tokens * per_pos_bytes`.
    /// In paged mode: `n_pages * PAGE_TOKENS * per_pos_bytes`.
    pub fn arena_bytes(&self) -> usize {
        if let Some(pool) = &self.page_pool {
            pool.arena_bytes()
        } else {
            self.descs.len() * self.cap_tokens * self.per_pos_bytes
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
        self.descs[slot.0].legacy_base = 0;
        self.descs[slot.0].seq_len = bt.live_tokens() as i32;
        self.dirty = true;
    }

    /// Share a prefix of `n_pages` pages from `src` slot into `dst` slot's
    /// block table. Used for prefix sharing across sessions.
    pub fn share_prefix(
        &mut self,
        src: SlotId,
        dst: SlotId,
        n_pages: usize,
    ) -> Result<(), String> {
        // Extract src page indices first to avoid overlapping borrows.
        let src_pages: Vec<u32> = {
            let src_bt = self
                .block_tables
                .get(src.0)
                .and_then(|bt| bt.as_ref())
                .ok_or_else(|| "share_prefix: src slot has no block table".to_string())?;
            if n_pages > src_bt.num_pages() {
                return Err(format!(
                    "share_prefix: src has {} pages but {} requested",
                    src_bt.num_pages(),
                    n_pages
                ));
            }
            src_bt.page_indices()[..n_pages].to_vec()
        };
        // Now do the mutable work.
        let pool = self
            .page_pool
            .as_mut()
            .ok_or_else(|| "share_prefix: pool is not in paged mode".to_string())?;
        let dst_bt = self
            .block_tables
            .get_mut(dst.0)
            .and_then(|bt| bt.as_mut())
            .ok_or_else(|| "share_prefix: dst slot has no block table".to_string())?;
        for &phys in &src_pages {
            pool.refcount_inc(phys);
            dst_bt.push_page(phys);
        }
        self.block_tables_dirty[dst.0] = true;
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
            let prev_end = d[i - 1].legacy_base + (p.cap_tokens() as u64) * PPB as u64;
            assert_eq!(
                d[i].legacy_base,
                prev_end,
                "slab {i} must start where {} ended",
                i - 1
            );
        }
    }

    #[test]
    fn q8_abi_uses_shared_legacy_base() {
        // Q8_0 ABI: the flash-prefill kernel uses ONE shared slab offset.
        // In the new paged layout, legacy_base serves as both k_base and v_base.
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
}
