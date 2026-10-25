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
        if page_tokens == 0 || per_pos_bytes == 0 {
            return Err("VmmPageMapper: page geometry must be positive".into());
        }
        if n_pages > u32::MAX as usize {
            return Err("VmmPageMapper: n_pages exceeds the u32 page-id space".into());
        }
        let page_bytes = page_tokens
            .checked_mul(per_pos_bytes)
            .ok_or_else(|| "VmmPageMapper: page bytes overflowed".to_string())?;
        let granularity = gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("VmmPageMapper: VMM granularity unavailable: {e}"))?;
        let requested_granule = if granule_bytes == 0 {
            granularity
        } else {
            granule_bytes
        };
        if requested_granule == 0 || !requested_granule.is_multiple_of(granularity) {
            return Err(format!(
                "VmmPageMapper: granule {requested_granule} is not a multiple of VMM granularity {granularity}"
            ));
        }
        // A logical page is the unit of ownership, not a VMM granule: on this
        // engine a q8 page is ~136 KiB while the device's recommended granule
        // is 4 KiB. Reserve whole granules per page and never split a page
        // across mappings. When the requested granule is large enough and
        // divides evenly, pack several pages into it; otherwise one page per
        // granule (the tail of a non-dividing granule is unused, as documented
        // by the descriptor).
        let page_granule = page_bytes.div_ceil(granularity) * granularity;
        let granule_bytes = requested_granule.max(page_granule);
        // `page_granule` is divisible by `page_bytes` only when the rounded
        // value happens to divide; use floor packing otherwise. The unused
        // tail is already part of the descriptor contract.
        let pages_per_granule = granule_bytes / page_bytes;
        if pages_per_granule == 0 {
            return Err(format!(
                "VmmPageMapper: page {page_bytes} B does not fit granule {granule_bytes} B"
            ));
        }
        if pages_per_granule > u32::MAX as usize {
            return Err("VmmPageMapper: pages_per_granule exceeds u32".into());
        }
        let n_granules = n_pages.div_ceil(pages_per_granule);
        let va_bytes = n_granules
            .checked_mul(granule_bytes)
            .ok_or_else(|| "VmmPageMapper: VA size overflowed".to_string())?;
        let arena_count = n_fa_layers
            .checked_mul(2)
            .ok_or_else(|| "VmmPageMapper: arena count overflowed".to_string())?;
        let granule_cost = granule_bytes
            .checked_mul(arena_count)
            .ok_or_else(|| "VmmPageMapper: physical granule cost overflowed".to_string())?;
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
    /// Mapping/unmapping happens here, at a caller-controlled boundary, and
    /// never inside a forward. On any mapping failure the previously mapped
    /// granules from this call are rolled back.
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

        let mut newly_mapped = Vec::new();
        for granule in 0..self.layout.n_granules {
            if used[granule] == self.mapped_granules[granule] {
                continue;
            }
            if used[granule] {
                if self.budget_bytes != 0
                    && self.mapped_bytes.saturating_add(granule_cost) > self.budget_bytes
                {
                    self.rollback_mapped(gpu, &newly_mapped);
                    return Err(format!(
                        "VmmPageMapper: mapping granule {granule} would exceed {} B budget ({} mapped)",
                        self.budget_bytes, self.mapped_bytes
                    ));
                }
                let offset = granule * self.layout.granule_bytes;
                let bases: Vec<(usize, usize)> = self
                    .k_bases
                    .iter()
                    .chain(self.v_bases.iter())
                    .copied()
                    .collect();
                let mut staged = 0usize;
                let mut error = None;
                for (arena_idx, &(base, size)) in bases.iter().enumerate() {
                    let arena = alias(base, size);
                    match gpu.vmm_map_at(&arena, offset, self.layout.granule_bytes, &[]) {
                        Ok(()) => staged += 1,
                        Err(e) => {
                            error = Some(format!(
                                "VmmPageMapper: map granule {granule} arena {arena_idx}: {e}"
                            ));
                            break;
                        }
                    }
                }
                if let Some(e) = error {
                    // Undo this granule's partial mapping in reverse order.
                    for &(base, size) in bases.iter().take(staged).rev() {
                        let arena = alias(base, size);
                        let _ = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes);
                    }
                    self.rollback_mapped(gpu, &newly_mapped);
                    return Err(e);
                }
                self.mapped_granules[granule] = true;
                self.mapped_bytes = self
                    .mapped_bytes
                    .checked_add(granule_cost)
                    .ok_or_else(|| "VmmPageMapper: mapped bytes overflowed".to_string())?;
                newly_mapped.push(granule);
            } else {
                // No page in this granule is resident. Complete prior stream
                // work before releasing the mappings. Every paged kernel runs
                // on the Gpu's default stream; wait
                // on that stream instead of fencing the whole device. This is
                // still a completion fence, not a per-page event lease: unmap
                // happens only at a caller-controlled boundary after the
                // step's writes/reads have been submitted.
                gpu.hip
                    .stream_synchronize_default()
                    .map_err(|e| format!("VmmPageMapper: stream sync before unmap: {e}"))?;
                let offset = granule * self.layout.granule_bytes;
                for &(base, size) in self.k_bases.iter().chain(self.v_bases.iter()) {
                    let arena = alias(base, size);
                    if let Err(e) = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes) {
                        self.poisoned = true;
                        return Err(format!(
                            "VmmPageMapper: unmap granule {granule} failed after partial teardown: {e}"
                        ));
                    }
                }
                self.mapped_granules[granule] = false;
                self.mapped_bytes = self.mapped_bytes.saturating_sub(granule_cost);
            }
        }
        Ok(())
    }

    fn rollback_mapped(&mut self, gpu: &mut Gpu, granules: &[usize]) {
        for &granule in granules.iter().rev() {
            let offset = granule * self.layout.granule_bytes;
            for &(base, size) in self.k_bases.iter().chain(self.v_bases.iter()) {
                let arena = alias(base, size);
                let _ = gpu.vmm_unmap_at(&arena, offset, self.layout.granule_bytes);
            }
            self.mapped_granules[granule] = false;
            let cost = self.layout.granule_bytes * (self.k_bases.len() + self.v_bases.len());
            self.mapped_bytes = self.mapped_bytes.saturating_sub(cost);
        }
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
