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

use crate::kv_slots::{preflight_alloc, KvSlotDesc, R9700_VRAM_BYTES};

/// Page size in tokens. Must match PAGE_TOKENS in slot_pool.rs and the
/// tile size the flash path walks KV in.
pub const PAGE_TOKENS: usize = 128;

/// A physical page index in the shared arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

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

    /// Append a physical page to the end of the block table.
    fn push_page(&mut self, page: u32) {
        self.pages.push(page);
    }

    /// Truncate to exactly `n_pages` pages, returning the freed pages.
    /// Used when trimming a session's KV (e.g. on eviction).
    fn truncate(&mut self, n_pages: usize) -> Vec<u32> {
        let freed: Vec<u32> = self.pages.drain(n_pages..).collect();
        self.live_tokens = self.live_tokens.min(n_pages * PAGE_TOKENS);
        freed
    }

    /// Set the live token count. Must be <= num_pages * PAGE_TOKENS.
    fn set_live_tokens(&mut self, tokens: usize) {
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
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Free-list page allocator for a shared KV arena.
///
/// The arena is a single contiguous buffer per layer (K and V each).
/// `PagePool` tracks which pages are free and which are allocated to
/// which session. Pages are `PAGE_TOKENS * per_pos_bytes` bytes each.
///
/// Prefix sharing: when two sessions share the same token prefix, they
/// share the physical pages for that prefix. Shared pages are refcounted
/// and only freed when the last session releases them.
#[derive(Debug)]
pub struct PagePool {
    /// Total number of physical pages in the arena.
    n_pages: usize,
    /// Per-position stride in bytes (n_kv_heads * (head_dim/32) * 34 for Q8_0).
    per_pos_bytes: usize,
    /// Free list of physical page indices. Pages are popped from the end.
    free_pages: Vec<u32>,
    /// Reference count per physical page. 0 = free, >0 = allocated.
    /// Index is physical page index.
    refcounts: Vec<u32>,
}

impl PagePool {
    /// Create a page pool with `n_pages` physical pages, each holding
    /// `PAGE_TOKENS` positions of `per_pos_bytes` bytes.
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(n_pages: usize, per_pos_bytes: usize) -> Result<Self, String> {
        assert!(n_pages > 0, "n_pages must be positive");
        assert!(per_pos_bytes > 0, "per_pos_bytes must be positive");

        let page_bytes = PAGE_TOKENS * per_pos_bytes;
        // K and V are separate arenas of identical layout, hence x2.
        let total = (page_bytes as u64)
            .checked_mul(n_pages as u64)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| "PagePool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "PagePool arena")?;

        let free_pages: Vec<u32> = (0..n_pages as u32).rev().collect();
        Ok(Self {
            n_pages,
            per_pos_bytes,
            free_pages,
            refcounts: vec![0; n_pages],
        })
    }

    /// Total number of physical pages.
    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    /// Per-position stride in bytes.
    pub fn per_pos_bytes(&self) -> usize {
        self.per_pos_bytes
    }

    /// Bytes per page (PAGE_TOKENS * per_pos_bytes).
    pub fn page_bytes(&self) -> usize {
        PAGE_TOKENS * self.per_pos_bytes
    }

    /// Bytes in ONE arena (K or V). The pool holds two of these.
    pub fn arena_bytes(&self) -> usize {
        self.n_pages * self.page_bytes()
    }

    /// Number of free pages available for allocation.
    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Maximum token capacity of the pool (all pages allocated).
    pub fn max_tokens(&self) -> usize {
        self.n_pages * PAGE_TOKENS
    }

    /// Allocate `n_pages` fresh pages and append them to `table`.
    /// Returns the number of pages actually allocated (may be less than
    /// requested if the pool is nearly full).
    pub fn alloc_pages(&mut self, table: &mut BlockTable, n_pages: usize) -> usize {
        let can_alloc = n_pages.min(self.free_pages.len());
        for _ in 0..can_alloc {
            let page = self.free_pages.pop().expect("free_pages non-empty");
            self.refcounts[page as usize] = 1;
            table.push_page(page);
        }
        can_alloc
    }

    /// Share `n_pages` pages from `src` table into `dst` table by
    /// incrementing refcounts. Both tables now reference the same physical
    /// pages. Used for prefix sharing: the shared prefix pages are
    /// refcounted so they persist until the last session releases them.
    ///
    /// Panics if `src` has fewer than `n_pages` pages.
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
        for lp in 0..n_pages {
            let phys = src.physical(lp).expect("src page exists");
            self.refcounts[phys as usize] =
                self.refcounts[phys as usize].saturating_add(1);
            dst.push_page(phys);
        }
        Ok(())
    }

    /// Free all pages in `table`, decrementing refcounts. Pages with
    /// refcount reaching 0 are returned to the free list.
    pub fn release_table(&mut self, table: &mut BlockTable) {
        for &phys in table.page_indices() {
            let rc = &mut self.refcounts[phys as usize];
            if *rc > 0 {
                *rc -= 1;
                if *rc == 0 {
                    self.free_pages.push(phys);
                }
            }
        }
        table.pages.clear();
        table.live_tokens = 0;
    }

    /// Free the last `n_pages` pages from `table`, decrementing refcounts.
    /// Used when trimming a session's KV (e.g. sliding window eviction).
    pub fn free_pages_from_tail(&mut self, table: &mut BlockTable, n_pages: usize) {
        let freed = table.truncate(table.num_pages().saturating_sub(n_pages));
        for phys in freed {
            let rc = &mut self.refcounts[phys as usize];
            if *rc > 0 {
                *rc -= 1;
                if *rc == 0 {
                    self.free_pages.push(phys);
                }
            }
        }
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

    /// Build a KvSlotDesc for a session with the given block table.
    /// `block_table_dev_addr` is the GPU address of the uploaded page
    /// index array (i32 per page). The descriptor is in paged mode.
    pub fn make_desc(&self, table: &BlockTable, block_table_dev_addr: u64) -> KvSlotDesc {
        KvSlotDesc {
            block_table: block_table_dev_addr,
            legacy_base: 0,
            seq_len: table.live_tokens() as i32,
            page_tokens: PAGE_TOKENS as i32,
        }
    }

    /// Build a legacy-mode KvSlotDesc (contiguous slab at `base`).
    /// Used for backward compatibility with non-paged code paths.
    pub fn make_legacy_desc(base: u64, seq_len: usize) -> KvSlotDesc {
        KvSlotDesc {
            block_table: 0,
            legacy_base: base,
            seq_len: seq_len as i32,
            page_tokens: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_release_roundtrip() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        assert_eq!(pool.free_pages(), 16);

        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, 4);
        assert_eq!(allocated, 4);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 12);

        // Each allocated page should have refcount 1
        for &phys in table.page_indices() {
            assert_eq!(pool.refcounts[phys as usize], 1);
        }

        pool.release_table(&mut table);
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
        // Shared pages should have refcount 2
        for lp in 0..3 {
            let phys = table_a.physical(lp).unwrap();
            assert_eq!(pool.refcounts[phys as usize], 2);
        }
        // Non-shared page should still have refcount 1
        let phys_4 = table_a.physical(3).unwrap();
        assert_eq!(pool.refcounts[phys_4 as usize], 1);

        // Releasing table_a should free only the non-shared page
        pool.release_table(&mut table_a);
        assert_eq!(pool.free_pages(), 16 - 3); // only 1 page freed, 3 still shared
        for lp in 0..3 {
            let phys = table_b.physical(lp).unwrap();
            assert_eq!(pool.refcounts[phys as usize], 1);
        }

        // Releasing table_b frees the remaining shared pages
        pool.release_table(&mut table_b);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn free_pages_from_tail() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 6);
        table.set_live_tokens(600);

        // Free last 2 pages
        pool.free_pages_from_tail(&mut table, 2);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 16 - 4);
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
        assert_eq!(desc.legacy_base, 0);
        assert_eq!(desc.seq_len, 200);
        assert_eq!(desc.page_tokens, PAGE_TOKENS as i32);
    }

    #[test]
    fn make_legacy_desc() {
        let desc = PagePool::make_legacy_desc(0x1000, 42);
        assert_eq!(desc.block_table, 0);
        assert_eq!(desc.legacy_base, 0x1000);
        assert_eq!(desc.seq_len, 42);
        assert_eq!(desc.page_tokens, 0);
    }
}
