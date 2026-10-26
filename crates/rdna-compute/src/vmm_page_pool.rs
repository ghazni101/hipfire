// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// VMM-backed physical mapping for the shared paged KV PagePool.
//
// The logical page pool still owns page ids, refcounts, COW, and prefix
// sharing. This mapper owns only the virtual-arena metadata and physical
// granule mapping; the kernel-visible owner tensors live in the caller
// (`VmmPageArenas`) so the engine can free them exactly once.
//
//   logical page  ->  PagePool block table  ->  physical page id
//   physical page ->  granule = id / P, within = id % P
//   arena offset  =  granule * granule_bytes + within * page_bytes
//
// Granules are mapped in every K/V layer arena when the first page in the
// granule becomes resident, and unmapped when the last resident page leaves.
// Mapping is synchronized at page-pool transition boundaries (admission,
// provisioning, COW, release), never inside a forward.

use crate::kv_slots::VmmPagedKvSlotDesc;
use crate::page_pool::{PageState, PAGE_TOKENS};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::DeviceBuffer;

/// Geometry shared by the descriptor packer and the COW/snapshot copy paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmmPageLayout {
    pub page_tokens: usize,
    pub page_bytes: usize,
    pub granule_bytes: usize,
    pub pages_per_granule: u32,
    pub n_pages: usize,
    pub n_granules: usize,
}

/// Granule packing that does not depend on how many logical pages the pool
/// owns. Auto-sizing uses this so the page count is bounded by committed
/// physical bytes (one whole granule in every K and V arena), not by the
/// slightly smaller logical page size. vLLM and SGLang size the block pool
/// the same way: the allocator's byte cost is the physical block, padding
/// included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmmGranuleGeometry {
    pub page_bytes: usize,
    pub granule_bytes: usize,
    pub pages_per_granule: u32,
}

impl VmmGranuleGeometry {
    /// `requested_granule == 0` selects at least 2 MiB, aligned up to
    /// `granularity`. A page larger than that granule gets its own
    /// rounded-up granule. Several pages pack into one granule when they
    /// fit; the unused tail is part of the descriptor contract.
    pub fn plan(
        granularity: usize,
        page_tokens: usize,
        per_pos_bytes: usize,
        requested_granule: usize,
    ) -> Result<Self, String> {
        if granularity == 0 {
            return Err("VmmPageMapper: VMM granularity is zero".into());
        }
        if page_tokens == 0 || per_pos_bytes == 0 {
            return Err("VmmPageMapper: page geometry must be positive".into());
        }
        let page_bytes = page_tokens
            .checked_mul(per_pos_bytes)
            .ok_or_else(|| "VmmPageMapper: page bytes overflowed".to_string())?;
        let requested = if requested_granule == 0 {
            // One q8 page is ~136 KiB. A 4 KiB granule would map once per
            // page; 2 MiB packs several pages and stays a multiple of the
            // device granularity on gfx1100.
            granularity.max(2 * 1024 * 1024)
        } else {
            requested_granule
        };
        if requested == 0 || !requested.is_multiple_of(granularity) {
            return Err(format!(
                "VmmPageMapper: granule {requested} is not a multiple of VMM granularity {granularity}"
            ));
        }
        let page_granule = page_bytes.div_ceil(granularity) * granularity;
        let granule_bytes = requested.max(page_granule);
        let pages_per_granule = granule_bytes / page_bytes;
        if pages_per_granule == 0 {
            return Err(format!(
                "VmmPageMapper: page {page_bytes} B does not fit granule {granule_bytes} B"
            ));
        }
        if pages_per_granule > u32::MAX as usize {
            return Err("VmmPageMapper: pages_per_granule exceeds u32".into());
        }
        Ok(Self {
            page_bytes,
            granule_bytes,
            pages_per_granule: pages_per_granule as u32,
        })
    }

    /// Physical bytes of one granule group: `granule_bytes` in every K arena
    /// and every V arena.
    pub fn group_bytes(&self, n_fa_layers: usize) -> Result<usize, String> {
        if n_fa_layers == 0 {
            return Err("VmmPageMapper: n_fa_layers must be positive".into());
        }
        let arenas = n_fa_layers
            .checked_mul(2)
            .ok_or_else(|| "VmmPageMapper: arena count overflowed".to_string())?;
        self.granule_bytes
            .checked_mul(arenas)
            .ok_or_else(|| "VmmPageMapper: physical granule cost overflowed".to_string())
    }

    /// Logical pages whose granule groups fit in `usable_bytes`.
    ///
    /// A partially filled last granule still commits a whole group, so the
    /// count is `groups * pages_per_granule` unless `cap_pages` cuts it.
    /// Returns an error when even one group does not fit: handing back a
    /// single page would OOM on the first map.
    pub fn pages_fitting(
        &self,
        n_fa_layers: usize,
        usable_bytes: usize,
        cap_pages: usize,
    ) -> Result<usize, String> {
        if cap_pages == 0 {
            return Err("VmmPageMapper: page cap is zero".into());
        }
        let group = self.group_bytes(n_fa_layers)?;
        if group == 0 {
            return Err("VmmPageMapper: granule group costs nothing".into());
        }
        let groups = usable_bytes / group;
        if groups == 0 {
            return Err(format!(
                "VmmPageMapper: one granule group costs {group} B but only {usable_bytes} B are available"
            ));
        }
        let packed = groups.saturating_mul(self.pages_per_granule as usize);
        Ok(packed.min(cap_pages))
    }

    /// Physical bytes committed when all `n_pages` are resident. The last
    /// partial granule rounds up to a full group.
    pub fn physical_bytes(&self, n_fa_layers: usize, n_pages: usize) -> Result<usize, String> {
        if n_pages == 0 {
            return Err("VmmPageMapper: n_pages is zero".into());
        }
        let groups = n_pages.div_ceil(self.pages_per_granule as usize);
        self.group_bytes(n_fa_layers)?
            .checked_mul(groups)
            .ok_or_else(|| "VmmPageMapper: physical pool size overflowed".to_string())
    }
}

/// Kernel-visible owner tensors. The caller owns and frees these exactly
/// once; the mapper holds only their base addresses and sizes.
pub struct VmmPageArenas {
    pub k: Vec<GpuTensor>,
    pub v: Vec<GpuTensor>,
}

impl VmmPageArenas {
    pub fn k_arenas(&self) -> &[GpuTensor] {
        &self.k
    }

    pub fn v_arenas(&self) -> &[GpuTensor] {
        &self.v
    }

    pub fn arenas(&self) -> impl Iterator<Item = &GpuTensor> {
        self.k.iter().chain(self.v.iter())
    }
}

/// One VMM arena per full-attention layer for K and V, with page-granular
/// mapping. Physical memory is committed only when `sync` sees the first
/// resident page in a granule.
pub struct VmmPageMapper {
    layout: VmmPageLayout,
    k_bases: Vec<(usize, usize)>,
    v_bases: Vec<(usize, usize)>,
    mapped_granules: Vec<bool>,
    mapped_bytes: usize,
    budget_bytes: usize,
    poisoned: bool,
    released: bool,
}

impl std::fmt::Debug for VmmPageMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmmPageMapper")
            .field("layout", &self.layout)
            .field("k_arenas", &self.k_bases.len())
            .field("v_arenas", &self.v_bases.len())
            .field(
                "mapped_granules",
                &self.mapped_granules.iter().filter(|&&m| m).count(),
            )
            .field("mapped_bytes", &self.mapped_bytes)
            .field("budget_bytes", &self.budget_bytes)
            .field("poisoned", &self.poisoned)
            .field("released", &self.released)
            .finish()
    }
}

fn alias(base: usize, size: usize) -> GpuTensor {
    // SAFETY: the caller owns a live registered VMM arena at `base` with at
    // least `size` reserved bytes. This is a non-owning view; it is never
    // freed and is only used for immediate map/unmap calls.
    GpuTensor {
        buf: unsafe { DeviceBuffer::from_raw(base as *mut std::ffi::c_void, size) },
        shape: vec![size],
        dtype: DType::Raw,
    }
}

impl VmmPageMapper {
    /// Reserve VA for `n_fa_layers` K and V arenas covering `n_pages`
    /// logical pages. `granule_bytes` must be a multiple of the device's VMM
    /// granularity; `0` selects the recommended granularity. A page larger
    /// than the requested granule gets its own rounded-up granule.
    ///
    /// `budget_bytes` is a hard cap on committed physical bytes across every
    /// K/V arena; `0` means no cap. No physical memory is committed here.
    /// The returned arenas are the owners and must be freed by the caller.
    pub fn new(
        gpu: &mut Gpu,
        n_fa_layers: usize,
        n_pages: usize,
        page_tokens: usize,
        per_pos_bytes: usize,
        granule_bytes: usize,
        budget_bytes: usize,
    ) -> Result<(Self, VmmPageArenas), String> {
        if n_fa_layers == 0 {
            return Err("VmmPageMapper: n_fa_layers must be positive".into());
        }
        if n_pages == 0 {
            return Err("VmmPageMapper: n_pages must be positive".into());
        }
        if n_pages > u32::MAX as usize {
            return Err("VmmPageMapper: n_pages exceeds the u32 page-id space".into());
        }
        let granularity = gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("VmmPageMapper: VMM granularity unavailable: {e}"))?;
        let geom =
            VmmGranuleGeometry::plan(granularity, page_tokens, per_pos_bytes, granule_bytes)?;
        let page_bytes = geom.page_bytes;
        let granule_bytes = geom.granule_bytes;
        let pages_per_granule = geom.pages_per_granule as usize;
        let n_granules = n_pages.div_ceil(pages_per_granule);
        let va_bytes = n_granules
            .checked_mul(granule_bytes)
            .ok_or_else(|| "VmmPageMapper: VA size overflowed".to_string())?;
        let granule_cost = geom.group_bytes(n_fa_layers)?;
        if budget_bytes != 0 && granule_cost > budget_bytes {
            return Err(format!(
                "VmmPageMapper: one granule group costs {granule_cost} B over the {budget_bytes} B budget"
            ));
        }

        let mut k = Vec::with_capacity(n_fa_layers);
        let mut v = Vec::with_capacity(n_fa_layers);
        for layer in 0..n_fa_layers {
            // SAFETY: no mapping is created by reserve; both owners are
            // registered and explicitly released by the caller.
            let kt = match unsafe { gpu.reserve_vmm_tensor(&[va_bytes], DType::Raw, &[]) } {
                Ok(t) => t,
                Err(e) => {
                    for t in k.drain(..).chain(v.drain(..)) {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(format!("VmmPageMapper: reserve K layer {layer}: {e}"));
                }
            };
            k.push(kt);
            // SAFETY: same reserve-only contract as K.
            let vt = match unsafe { gpu.reserve_vmm_tensor(&[va_bytes], DType::Raw, &[]) } {
                Ok(t) => t,
                Err(e) => {
                    for t in k.drain(..).chain(v.drain(..)) {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(format!("VmmPageMapper: reserve V layer {layer}: {e}"));
                }
            };
            v.push(vt);
        }
        let k_bases: Vec<(usize, usize)> = k
            .iter()
            .map(|t| (t.buf.as_ptr() as usize, va_bytes))
            .collect();
        let v_bases: Vec<(usize, usize)> = v
            .iter()
            .map(|t| (t.buf.as_ptr() as usize, va_bytes))
            .collect();

        Ok((
            Self {
                layout: VmmPageLayout {
                    page_tokens,
                    page_bytes,
                    granule_bytes,
                    pages_per_granule: pages_per_granule as u32,
                    n_pages,
                    n_granules,
                },
                k_bases,
                v_bases,
                mapped_granules: vec![false; n_granules],
                mapped_bytes: 0,
                budget_bytes,
                poisoned: false,
                released: false,
            },
            VmmPageArenas { k, v },
        ))
    }

    /// `PAGE_TOKENS` is the paged engine's logical page size; keep the
    /// constructor's `page_tokens` argument aligned with it for descriptor
    /// and offset compatibility.
    pub fn page_layout(&self) -> VmmPageLayout {
        self.layout
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Physical byte offset of page `phys_page` in every K/V arena.
    pub fn page_offset(&self, phys_page: u32) -> Option<usize> {
        let p = self.layout.pages_per_granule as usize;
        if phys_page as usize >= self.layout.n_pages {
            return None;
        }
        let within = phys_page as usize % p;
        Some((phys_page as usize / p) * self.layout.granule_bytes + within * self.layout.page_bytes)
    }

    /// Build the 48-byte descriptor for one slot. `block_table` is the GPU
    /// address of that slot's physical page-index array.
    pub fn descriptor(&self, block_table: u64, seq_len: usize) -> VmmPagedKvSlotDesc {
        VmmPagedKvSlotDesc {
            block_table,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len: seq_len.min(i32::MAX as usize) as i32,
            page_tokens: self.layout.page_tokens as i32,
            granule_bytes: self.layout.granule_bytes as u64,
            pages_per_granule: self.layout.pages_per_granule,
            _pad: 0,
        }
    }

    /// Validate a block table before upload. Every live logical page must be
    /// within the mapper's page space; the mapper also requires its granule
    /// to be mapped after `sync`.
    pub fn check_table(&self, table: &[u32], live_pages: usize) -> Result<(), String> {
        if live_pages > table.len() {
            return Err(format!(
                "VmmPageMapper: {live_pages} live pages but table has {}",
                table.len()
            ));
        }
        for (lp, &phys) in table[..live_pages].iter().enumerate() {
            if phys as usize >= self.layout.n_pages {
                return Err(format!(
                    "VmmPageMapper: logical page {lp} maps phys {phys} >= {}",
                    self.layout.n_pages
                ));
            }
            let granule = phys as usize / self.layout.pages_per_granule as usize;
            if !self.mapped_granules[granule] {
                return Err(format!(
                    "VmmPageMapper: logical page {lp} phys {phys} granule {granule} is unmapped"
                ));
            }
        }
        Ok(())
    }

    /// Refresh the owner descriptors' safe lengths after `sync` changed the
    /// mapped extent. Owners are caller-held (see [`VmmPageArenas`]).
    pub fn refresh_owners(&self, gpu: &mut Gpu, arenas: &mut VmmPageArenas) -> Result<(), String> {
        for tensor in arenas.k.iter_mut().chain(arenas.v.iter_mut()) {
            gpu.refresh_vmm_owner_size(tensor)
                .map_err(|e| format!("VmmPageMapper: refresh owner: {e}"))?;
        }
        Ok(())
    }

    /// Reconcile the mapped granules with the page pool's current states.
    /// `states[phys]` is authoritative: every state except `Free` keeps the
    /// page's granule mapped (`ReclaimPending` may still have device reads).
    ///
    /// Free granules are unmapped before any new granule is mapped. Doing it
    /// the other way charges soon-to-be-freed physical memory against the
    /// budget and can refuse a map that would have fit, the same failure
    /// mode as freeing a block only after the next allocation in a paged
    /// pool. On a mapping failure the granules mapped by this call are
    /// rolled back. An unmap that cannot be undone poisons the mapper: the
    /// bitmap must not claim a VA range is free while a segment is still
    /// mapped, or the next `map_at` fails closed with "overlaps" and the
    /// range is stuck.
    pub fn sync(&mut self, gpu: &mut Gpu, states: &[PageState]) -> Result<(), String> {
        if self.released {
            return Err("VmmPageMapper: sync after release".into());
        }
        if self.poisoned {
            return Err("VmmPageMapper: mapper is poisoned by a failed unmap".into());
        }
        if states.len() != self.layout.n_pages {
            return Err(format!(
                "VmmPageMapper: {} page states for {} pages",
                states.len(),
                self.layout.n_pages
            ));
        }
        let p = self.layout.pages_per_granule as usize;
        let mut used = vec![false; self.layout.n_granules];
        for (phys, &state) in states.iter().enumerate() {
            if state != PageState::Free {
                used[phys / p] = true;
            }
        }
        let arena_count = self.k_bases.len() + self.v_bases.len();
        let granule_cost = self
            .layout
            .granule_bytes
            .checked_mul(arena_count)
            .ok_or_else(|| "VmmPageMapper: granule cost overflowed".to_string())?;

        let needs_unmap = (0..self.layout.n_granules)
            .any(|granule| self.mapped_granules[granule] && !used[granule]);
        if needs_unmap {
            // Paged kernels run on the default stream. Wait for that stream
            // once, then unmap. `vmm_unmap_at` also device-synchronizes, so
            // a kernel that was not on the default stream still cannot race
            // the unmap.
            gpu.hip
                .stream_synchronize_default()
                .map_err(|e| format!("VmmPageMapper: stream sync before unmap: {e}"))?;
            for granule in 0..self.layout.n_granules {
                if self.mapped_granules[granule] && !used[granule] {
                    self.unmap_granule(gpu, granule, granule_cost)?;
                }
            }
        }

        let mut newly_mapped = Vec::new();
        for granule in 0..self.layout.n_granules {
            if !used[granule] || self.mapped_granules[granule] {
                continue;
            }
            if self.budget_bytes != 0
                && self.mapped_bytes.saturating_add(granule_cost) > self.budget_bytes
            {
                self.rollback_mapped(gpu, &newly_mapped)?;
                return Err(format!(
                    "VmmPageMapper: mapping granule {granule} would exceed {} B budget ({} mapped)",
                    self.budget_bytes, self.mapped_bytes
                ));
            }
            if let Err(e) = self.map_granule(gpu, granule, granule_cost) {
                self.rollback_mapped(gpu, &newly_mapped)?;
                return Err(e);
            }
            newly_mapped.push(granule);
        }
        Ok(())
    }

    fn map_granule(
        &mut self,
        gpu: &mut Gpu,
        granule: usize,
        granule_cost: usize,
    ) -> Result<(), String> {
        let offset = granule * self.layout.granule_bytes;
        let bases: Vec<(usize, usize)> = self
            .k_bases
            .iter()
            .chain(self.v_bases.iter())
            .copied()
            .collect();
        let mut staged = 0usize;
        for (arena_idx, &(base, size)) in bases.iter().enumerate() {
            let arena = alias(base, size);
            if let Err(e) = gpu.vmm_map_at(&arena, offset, self.layout.granule_bytes, &[]) {
                let mut staged_err = None;
                for &(base, size) in bases.iter().take(staged).rev() {
                    let arena = alias(base, size);
                    if let Err(ue) = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes) {
                        self.poisoned = true;
                        staged_err = Some(format!(
                            "VmmPageMapper: map granule {granule} arena {arena_idx}: {e}; \
                             staged unmap failed and the mapper is poisoned: {ue}"
                        ));
                        break;
                    }
                }
                if let Some(err) = staged_err {
                    return Err(err);
                }
                return Err(format!(
                    "VmmPageMapper: map granule {granule} arena {arena_idx}: {e}"
                ));
            }
            staged += 1;
        }
        let Some(next) = self.mapped_bytes.checked_add(granule_cost) else {
            // The maps landed. Drop them before returning so the unset bit
            // still matches the VA. An unmap failure poisons instead of
            // pretending the range is free.
            for &(base, size) in bases.iter().rev() {
                let arena = alias(base, size);
                if let Err(e) = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes) {
                    self.poisoned = true;
                    return Err(format!(
                        "VmmPageMapper: mapped bytes overflowed; unmap of granule {granule} failed and the mapper is poisoned: {e}"
                    ));
                }
            }
            return Err("VmmPageMapper: mapped bytes overflowed".into());
        };
        self.mapped_bytes = next;
        self.mapped_granules[granule] = true;
        Ok(())
    }

    fn unmap_granule(
        &mut self,
        gpu: &mut Gpu,
        granule: usize,
        granule_cost: usize,
    ) -> Result<(), String> {
        let offset = granule * self.layout.granule_bytes;
        for &(base, size) in self.k_bases.iter().chain(self.v_bases.iter()) {
            let arena = alias(base, size);
            if let Err(e) = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes) {
                // Leave the bit set. Some arenas in this granule may already
                // be unmapped; a cleared bit would let the next map overlap
                // the ones that survived.
                self.poisoned = true;
                return Err(format!(
                    "VmmPageMapper: unmap granule {granule} failed after partial teardown: {e}"
                ));
            }
        }
        self.mapped_granules[granule] = false;
        self.mapped_bytes = self.mapped_bytes.saturating_sub(granule_cost);
        Ok(())
    }

    fn rollback_mapped(&mut self, gpu: &mut Gpu, granules: &[usize]) -> Result<(), String> {
        let cost = self.layout.granule_bytes * (self.k_bases.len() + self.v_bases.len());
        for &granule in granules.iter().rev() {
            if !self.mapped_granules[granule] {
                continue;
            }
            let offset = granule * self.layout.granule_bytes;
            for &(base, size) in self.k_bases.iter().chain(self.v_bases.iter()) {
                let arena = alias(base, size);
                if let Err(e) = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes) {
                    self.poisoned = true;
                    return Err(format!(
                        "VmmPageMapper: rollback unmap of granule {granule} failed; mapper poisoned: {e}"
                    ));
                }
            }
            self.mapped_granules[granule] = false;
            self.mapped_bytes = self.mapped_bytes.saturating_sub(cost);
        }
        Ok(())
    }

    /// Clear the mapper's bookkeeping. The caller must free the
    /// [`VmmPageArenas`] owners (which unmaps every segment). Idempotent.
    pub fn release(&mut self) {
        self.mapped_granules.fill(false);
        self.mapped_bytes = 0;
        self.released = true;
    }
}

impl Drop for VmmPageMapper {
    fn drop(&mut self) {
        if !self.released && self.mapped_bytes != 0 {
            eprintln!(
                "[vmm-page-pool] VmmPageMapper dropped with {} mapped bytes; free the caller-owned arenas",
                self.mapped_bytes
            );
        }
    }
}

/// `PAGE_TOKENS` is the paged engine's page size; callers that build a
/// mapper from `PagePool` should pass it so descriptor and copy arithmetic
/// cannot drift.
pub const VMM_PAGE_TOKENS: usize = PAGE_TOKENS;

#[cfg(test)]
mod tests {
    use super::VmmGranuleGeometry;

    /// Qwen3.5-27B q8: 16 full-attention layers, 4 KV heads, D=256, 1088 B/token.
    /// A 2 MiB granule holds 15 pages and wastes 8 KiB. Sizing by the logical
    /// page (K+V across layers) asks for one more granule group than fits in
    /// `free - 512 MiB`.
    #[test]
    fn q8_27b_auto_pool_fits_the_granule_budget_logical_sizing_does_not() {
        let geom = VmmGranuleGeometry::plan(4096, 128, 1088, 0).unwrap();
        assert_eq!(geom.page_bytes, 139_264);
        assert_eq!(geom.granule_bytes, 2 * 1024 * 1024);
        assert_eq!(geom.pages_per_granule, 15);

        let layers = 16;
        let logical_page = layers * 128 * (1088 + 1088);
        assert_eq!(logical_page, 4_456_448);
        let free = 10usize << 30;
        let reserve = 512usize << 20;
        let usable = free - reserve;
        let logical_pages = usable / logical_page;
        let logical_cost = geom.physical_bytes(layers, logical_pages).unwrap();
        assert!(
            logical_cost > usable,
            "logical sizing must overshoot: cost {logical_cost} usable {usable} pages {logical_pages}"
        );

        let pages = geom.pages_fitting(layers, usable, usize::MAX).unwrap();
        let physical = geom.physical_bytes(layers, pages).unwrap();
        assert!(physical <= usable, "physical {physical} usable {usable}");
        assert!(pages < logical_pages);
        assert_eq!(pages % 15, 0);
    }

    #[test]
    fn pages_fitting_refuses_a_budget_smaller_than_one_group_and_honors_the_cap() {
        let geom = VmmGranuleGeometry::plan(4096, 128, 1088, 0).unwrap();
        let group = geom.group_bytes(16).unwrap();
        let err = geom.pages_fitting(16, group - 1, 100).unwrap_err();
        assert!(err.contains("granule group"), "{err}");
        assert_eq!(geom.pages_fitting(16, group, 10).unwrap(), 10);
        assert_eq!(geom.physical_bytes(16, 10).unwrap(), group);
        assert_eq!(geom.pages_fitting(16, group * 2, 1000).unwrap(), 30);
    }
}
