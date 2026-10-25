# VMM-backed shared paged KV pool

**Status:** proposed (design only; no implementation on this branch).
**Date:** 2026-10-25.
**Branch:** `design/vmm-paged-kv-pool` (cut from `feat/multislot-vram-efficiency`).
**Related:**
[`2026-10-22-vmm-kv-multislot-engine.md`](2026-10-22-vmm-kv-multislot-engine.md),
[`2026-09-03-multislot-paged-mtp-vision.md`](../specs/2026-09-03-multislot-paged-mtp-vision.md),
[`2026-09-05-serving-cache-scheduler-spec.md`](../specs/2026-09-05-serving-cache-scheduler-spec.md).

## Summary

The multi-slot engine currently has two KV storage routes:

- **VMM:** one `KvCache` VMM owner per slot. Stable VA per request, contiguous
  prefix mapped at admission for `prompt + max_tokens + slack`, `_vmm`
  descriptor kernels. No shared pages, no GPU COW; the prefix cache is a
  host-snapshot LRU.
- **Paged:** one shared `PagePool` of 128-token physical pages, per-slot block
  tables, refcounts/COW/prefix sharing, `_paged` descriptor kernels. The K/V
  arenas are ordinary `zeros` allocations sized to the full page count, so the
  pool is physically preallocated at load (1361 pages / 6.07 GB on the 27B
  fixture, 383 MB free after load).

This design combines them: reserve VMM VA for the paged K/V arenas, commit
physical memory one **granule group** at a time as `PagePool` allocates pages,
and keep the existing block tables, page refcounts, COW, prefix cache and
admission/refusal behavior. The logical page stays 128 tokens; the kernel
translation changes from a contiguous page index to a granule-grouped page
index behind a new descriptor/header variant, so the arena-paged route and its
byte-identical code objects are untouched.

## Goals

1. Load-time KV cost is VA only. Physical KV is committed as pages are
   allocated, not as a full pool at load.
2. One shared pool for all slots, with the existing `PagePool` semantics:
   on-demand page allocation, refcounts, seal/CacheOnly, COW, radix prefix
   cache, page-demand reclaim, per-request overload refusal.
3. Preserve 128-token logical pages and the existing block-table ABI/behavior.
4. No mapping inside an engine step. Page mapping/unmapping happens at
   admission/provisioning/release boundaries, with a completion event before
   unmap.
5. Byte-identical outputs to the arena-paged route for the same batch and
   state, at every test width.
6. Reclaim physical memory when pages are released: a finished/evicted
   session returns its granules; a later request can reuse them.
7. Fail closed: map/allocation failure becomes a typed retryable refusal, not
   an engine-thread death or a partially visible page.

## Non-goals (phase 1)

- Cross-device or multi-GPU shared physical memory.
- Windows. VMM growth on Windows already maps the full reservation and is
  documented as unsafe for segmented growth; this backend starts Linux-only.
- Rotated/asym/native-bf16 KV tiers. Like the VMM route, VMM-paged admits q8
  (and fp8 where the VMM route already does); equal K/V strides are required
  by the single page layout. Other tiers keep the arena-paged route.
- Changing the logical `PagePool` sharing/COW policy, radix cache policy, or
  the admission/refusal contract.
- Partial GPU offload in auto mode. Explicit VA/page counts may be added
  later; phase 1 requires fully resident weights (the current auto-paged rule).
- Making VMM-paged the default. Phase 4 ends in a measured default decision,
  not a silent flip.

## Current constraints and the central arithmetic

### VMM primitive

`hip_bridge::VmmArena` (`crates/hip-bridge/src/vmm.rs`) reserves VA, maps
physical allocations with `map_next` / `map_next_granules`, and can share a
**prefix** of granules (`share_prefix` / `map_shared_prefix`). It has no
arbitrary `map_at(offset, size)` / `unmap_at(offset, size)` primitive, and
`map_shared_prefix` requires a never-mapped arena. Both are needed for a
page-granular pool; phase 0 adds them.

`Gpu::alloc_vmm_tensor`, `grow_vmm_tensor`, `vmm_granularity` and
`vmm_mapped_bytes` (`crates/rdna-compute/src/dispatch.rs`) are the existing
public surface. `KvCache` (`crates/saddle-core/src/kv.rs`) uses a contiguous
prefix mapping, not a page table.

### Page geometry (27B fixture)

Qwen3.8-27B, q8 KV:

- 16 full-attention layers
- 4 KV heads, head_dim 256
- K stride = V stride = 4 × 272 = 1088 B/token/layer
- 128-token page = 139,264 B/layer; K+V per page = 278,528 B/layer
- 2 MiB VMM granule = 2,097,152 B
- `P = floor(2 MiB / 139,264) = 15` pages per granule per K/V arena
- granule tail = 2,097,152 − 15 × 139,264 = 8,192 B unused
- one logical page, all 16 layers, K+V = 4,456,448 B (~4.25 MiB)

A 128-token page is **not** 2 MiB-aligned and does not divide the granule.
`P = 15` lets every page live entirely inside one granule:

```
page_offset(page) =
    (page / P) * granule_bytes
  + (page % P) * page_bytes
```

with `granule_bytes` = 2 MiB and `page_bytes` = `page_tokens × stride`. The
last `granule_bytes − P × page_bytes` bytes of every granule are unused. This
is the key to keeping 128-token pages, COW and prefix sharing while mapping
VMM granules.

### Arena layout and VA

One VMM arena per full-attention layer for K and one for V (same as the
current `k_arenas`/`v_arenas` shape). Each arena reserves:

```
granules = ceil(n_pages / P)
va_bytes = granules * granule_bytes
```

For 1361 pages: 91 granules = 182 MiB per K/V arena, 5.8 GiB VA total on the
27B geometry. VA costs no physical memory. Physical granules are mapped only
when a page in that granule is allocated; a granule is unmapped when its last
page ref is gone (after synchronization).

A page index is global across layers; all layer arenas use the same
granule/page layout, so the same `page_offset` applies in every K/V arena.

**Geometry refusals.** The backend requires:

- `P >= 1` (`page_tokens × stride <= granule_bytes`); a page must fit inside
  one granule. A larger page would need a multi-granule page layout, which is
  out of scope for phase 1.
- equal K and V strides, so one page index has one page offset in both arenas.
  q8 and the admitted fp8 tier satisfy this; rotated/asym tiers do not and
  stay on the arena-paged route.
- `ceil(n_pages / P) × granule_bytes` per K/V arena to fit the driver's VA
  reservation; a refused VA reservation fails the load closed before weights
  are touched.

The mapper computes `P` and `granule_bytes` once per load and exposes them to
descriptor packing. `P` is never inferred by the kernel from `stride`; the
kernel carries both constants so the arithmetic is explicit.

### Kernel translation

`kernels/src/kv_slot_desc_paged.h` is the only translation point today:

```
phys_page = block_table[pos / page_tokens]
offset = phys_page * page_tokens * stride + (pos % page_tokens) * stride
```

The new backend needs:

```
phys_page = block_table[pos / page_tokens]
granule   = phys_page / pages_per_granule
within    = phys_page % pages_per_granule
offset =
    granule * granule_bytes
  + within * page_tokens * stride
  + (pos % page_tokens) * stride
```

Add a new descriptor header `kernels/src/kv_slot_desc_vmm_paged.h` carrying
`granule_bytes` and `pages_per_granule` (0 = existing contiguous behavior),
plus a `_vmm_paged` symbol variant for each `_paged` kernel. The body and its
`kv_offset_for_k/v()` calls stay the same; the descriptor layout changes only
for the new variant. The arena-paged header/code objects are untouched. This
keeps the existing paged route byte-identical and gives one code path per
variant.

The host mirror (`crates/rdna-compute/src/kv_slots.rs`) and the symbol table
(`kv_slot_paged_symbol` in `crates/rdna-compute/src/attention.rs`) grow the
third variant. A module is selected as `_vmm_paged` only when the slot pool is
VMM-paged.

## Design

### 1. `VmmPageMapper` (new, in `crates/rdna-compute`)

Owns the per-layer K/V VMM arenas and the physical granule allocator.

State per K/V arena:

- `base_va`, `granule_bytes`, `pages_per_granule`, `n_granules`
- mapped-granule bitmap / free list
- mapped-granule count and physical-byte count
- a free list of **page indices** supplied by the logical `PagePool`
- granule occupancy count (number of referenced pages per granule)

State per logical page:

- `granule = page / P`, `within = page % P`, `offset`
- page refcount is owned by the logical `PagePool`; the mapper only needs the
  set of allocated pages to decide when a granule becomes empty.

API (all mapping happens on the engine thread, outside a step):

```rust
fn reserve_va(n_pages, page_bytes_k, page_bytes_v, n_layers, n_slots) -> Result<Self>
fn map_page(gpu, page_index) -> Result<()>       // maps its granule in every K/V arena
fn unmap_page_after(gpu, page_index, event)      // drops occupancy; unmaps empty granules
fn mapped_bytes(&self) -> usize
fn budget_bytes(&self) -> usize
fn page_layout(&self) -> PageLayout { granule_bytes, pages_per_granule, page_bytes_k/v }
fn check_slot(gpu, page_indices) -> Result<()>   // host validation before descriptor upload
```

- `reserve_va` uses one `VmmArena::reserve` per K/V per layer. No
  `hipMemCreate`.
- `map_page` computes the granule and, for each of the 32 arenas, maps the
  granule at `base_va + granule * granule_bytes` if not already mapped. A
  granule map may be batched per layer (contiguous granules) to reduce
  `hipMemMap` calls; start simple and measure.
- Physical failure (`hipMemCreate`/`hipMemMap` OOM) is returned before any
  page becomes visible. The mapper rolls back the granules it mapped for this
  call and does not increment occupancy.
- `unmap_page_after` takes the completion event recorded after the last step
  that read/wrote the page (the existing `PagePool` page leases/epochs can
  supply the sync point), decrements occupancy, and unmaps a granule when its
  count reaches zero. It must never unmap while any mapped stream may read.
- `budget_bytes` is the physical ceiling: free VRAM after weights/scratch
  minus the map reserve. It is checked before the first granule of a page so
  an over-budget request is a typed refusal.
- The mapper owns the VMM tensors. The engine passes their base pointers to
  the `_vmm_paged` kernels; the page offset table is still the block table.

### 2. Logical `PagePool` stays the source of truth

`crates/rdna-compute/src/page_pool.rs` keeps:

- `n_pages`, `free_pages`, `page_meta` (state/refcount/seal/generation)
- `BlockTable`, `alloc_pages`, `ensure_capacity`, `release_table`,
  `free_pages_from_tail`
- `share_prefix`, `share_published_pages`, seal/CacheOnly, COW plans
- logical page indices in block tables

Add a **mapper hook** so logical allocation/release drives physical mapping:

```rust
trait PageMapper {
    fn page_layout(&self) -> Option<PageLayout>; // None = arena contiguous
    fn map_pages(&mut self, gpu: &mut Gpu, pages: &[u32]) -> Result<(), String>;
    fn release_after(&mut self, gpu: &mut Gpu, pages: &[u32], completion: Option<&Event>);
    fn mapped_bytes(&self) -> usize;
}
```

`PagePool` keeps `Option<Box<dyn PageMapper>>`? Avoid `dyn` on the hot path:
the mapper is only used at allocation/release, not per token. A trait object is
acceptable; alternatively an enum `PageBackend { Arena, Vmm(VmmPageMapper) }`.

Allocation paths that currently only mutate metadata now need a `Gpu`:

- `SlotPool::set_seq_len(gpu, slot, seq_len)` and `provision_pages_for`
  (`PagePool::ensure_capacity` → `alloc_pages` → `map_pages`)
- `SlotPool::share_prefix` / `share_published_pages` (shared pages are already
  mapped; no new mapping unless the destination needs pages)
- `SlotPool::release` / `reset` / `release_table` (free pages; unmap after a
  completion event)
- `commit_cow` (new pages → map before the copies)
- `free_pages_from_tail` / sliding-window free

The engine already calls `set_seq_len` at admission and at each step's
provisioning point, and already turns a failure into a per-request refusal
(`serve_engine.rs` page-demand reclaim and `provision_failed` handling). The
`Gpu` parameter is threaded from those call sites. No mapping may occur inside
`forward_batch_slots_*`; the mapper must be finished before the forward.

### 3. Descriptor and kernel variant

New header `kernels/src/kv_slot_desc_vmm_paged.h`:

```c
struct KvSlotDesc {
    const int* __restrict__ block_table;
    unsigned long long legacy_k_base;   // unused (0)
    unsigned long long legacy_v_base;   // unused (0)
    int seq_len;
    int page_tokens;
    unsigned long long granule_bytes;   // 0 = contiguous paged
    unsigned int pages_per_granule;     // 0 = contiguous paged
};
```

`kv_offset_for_k/v` uses the granule formula when both constants are non-zero
and the block table is non-null; otherwise the existing contiguous formula.
This keeps one source body. The `_vmm_paged` module defines
`HIPFIRE_KV_SLOT_VMM_PAGED` and uses this header; the arena `_paged` module
keeps `kv_slot_desc_paged.h` and its exact code bytes.

Host mirror in `crates/rdna-compute/src/kv_slots.rs`; descriptor packers in
`forward_slots.rs` fill `granule_bytes = 0, pages_per_granule = 0` for the
arena route and the actual layout for VMM-paged.

Device-side validation (fail closed): every page index in the block table is
`< n_pages`; the mapper has mapped the granule; a page must never point at an
unmapped granule. The host validation is authoritative; the device check is a
last line of defence if a descriptor table can be trusted only at pack time.
Do not add a per-token branch on the hot path beyond the two constants.

Kernel families to add the variant for (all `_paged` symbols):

- `attention_flash_*_tile_batched_paged` (q8/bf16/fp8/f16/asym/fwht table)
- `attention_q8_0_flash_prefill_paged`, `..._wmma_paged`
- `attention_q8_0_kv_batched_paged`, fp8/bf16/f16 variants
- `kv_cache_write_q8_0_batched_paged`, fp8/bf16/f16 variants
- `kv_cache_write_asym_k_*_batched_paged` (not phase 1 if rotated tiers are
  refused; still add only the q8/fp8 VMM-paged symbols first)
- `attention_*_reduce_*` that consume descriptors

Phase 1 restricts to q8 (and fp8 where admitted), equal K/V strides, so the
single page offset applies to both arenas. Rotated tiers keep the arena route.

### 4. COW and sharing

Logical `PagePool` COW is unchanged:

1. `plan_cow` identifies shared/sealed pages in the write interval and
   reserves private page indices.
2. The engine maps the new pages (`mapper.map_pages`) before copies.
3. `execute_cow_copies` copies the page bytes from old to new page indices in
   every K/V arena. With VMM-paged, copy by `page_offset`, not a flat arena
   offset.
4. `commit_cow` rebinds the block table; the old page refs are dropped and
   their granules unmapped when empty (after the copy's completion event).

Page sharing is just two block tables referencing the same page index. Prefix
cache publication (`seal` + `add_cache_ref`) keeps the page/granule mapped;
radix eviction releases cache refs and unmaps empty granules. Refcounts,
generations, leases and `PageHandle` stay the logical source of truth.

The COW copy size is 32 × `page_bytes` = 4.45 MiB for a 128-token page on the
27B fixture. That is larger than the previous arena COW's physical copy but
proportional to the KV bytes; measure COW-heavy agentic workloads.

### 5. Admission, auto sizing and refusal

- Reserve VA at load for the logical maximum (`n_slots × ceil(cap/128)` pages
  or a lower configured page count). VA is ~5.8 GiB on the 27B fixture at full
  cap; physical is zero.
- The physical budget is post-weight/post-scratch free VRAM less the map
  reserve. The mapper charges whole granules: `page_cost ≈ granule_bytes ×
  2 × n_fa_layers / P`.
- Admission/provisioning asks for pages for the next frontier, maps them, and
  on failure emits the existing retryable overload 503; the engine stays
  healthy.
- Auto mode computes `n_pages` from the physical budget and the configured
  slot cap; explicit `serve.multi_slot_paged_pages` sets the logical pool
  ceiling. The mapper refuses physical granules above its budget even if the
  logical pool has free pages.
- Partial offload with auto sizing is refused, as in the arena-paged route;
  explicit page counts may be supported later.

### 6. Swap/restore, graphs, and release

- Swap capture reads K/V bytes by page index/offset; the snapshot is the same
  logical payload. Restore allocates pages, maps granules, and copies bytes
  before publishing `seq_len`.
- Graph capture must not map. Pre-provision the step's pages before capture;
  a graph is invalidated when a slot's block table/COW offsets change. The
  current paged route already falls back from graph capture; keep that until a
  capture test proves stable tables.
- On session close/evict/sweep: release the block table pages; refs drop;
  granules unmap after the last reader's event. A parked session's pages are
  freed (the snapshot holds the bytes), unlike VMM's retained prefix mapping.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Per-page 2 MiB VMM map | 14× physical waste per page; unacceptable. |
| Logical page = 1927 tokens (granule-aligned) | No kernel change, but prefix sharing/COW granularity becomes ~1.9k tokens; short prompts allocate 64 MiB; regresses the radix cache and multi-slot short requests. |
| Keep per-slot VMM owners + `share_prefix` | Shares only whole prefixes into never-mapped arenas; no page-granular refcounts/COW and no shared pool. |
| Two-level byte-offset page table | More flexible than the granule-group formula, but adds a table load per page and more host/device ABI; the granule-group formula stores no per-page offsets and keeps the existing block table. Revisit only if `P` packing becomes a measured problem. |
| Page-index × page_bytes with 2 MiB map at unaligned offset | VMM requires granule-aligned VA; not expressible. |

## Phases and exit gates

### Phase 0: VMM page arena primitive

- Add `map_at(offset, bytes, handle)` / `unmap_at(offset, bytes)` and
  granule lease/refcount support to `VmmArena` (or a thin `VmmPageArena`
  wrapper), including segment bookkeeping and fault-injection hooks.
- Unit/GPU smoke: reserve 3 granules, map pages at granule offsets, write a
  pattern, unmap one granule after an event, verify the rest; map the same
  physical handle at two offsets; OOM rollback; double-unmap refusal.
- Exit: no engine changes; tests prove map/unmap/lease/refcount and
  granularity rounding.

### Phase 1: descriptor + kernel variant parity

- Add `kv_slot_desc_vmm_paged.h`, the host mirror, `_vmm_paged` symbols and
  launch selection.
- Populate the new constants with `P` and granule bytes; make the arena
  `_paged` path use `granule_bytes=0` unchanged.
- Oracle: run every VMM-paged kernel at small shapes against the arena-paged
  twin (same block table, same bytes) and require byte-identical outputs on
  zero/nonzero bases, multiple slots, tails, poisoned guards.
- Exit: all symbols exist, parity passes, existing arena code objects are
  unchanged (digest check).

### Phase 2: single-slot VMM-paged pool

- `PageMapper` hook + `VmmPageMapper`; reserve VA, map pages on
  `set_seq_len`, unmap on release.
- Single-slot prefill/decode, page reuse, release/reuse cycle, auto and
  explicit page counts, OOM refusal.
- Byte parity vs arena-paged on a fixed prompt set; KLD/greedy checks where
  applicable.
- Exit: one slot runs a 40k generation; release returns physical granules;
  a second request reuses them; no mapping in a forward.

### Phase 3: multi-slot, COW, prefix cache, swap

- 4 slots, 4 × 40k on the 27B fixture; auto pool leaves VMM-like free VRAM at
  load; all four lockstep.
- COW: shared prefix pages, write after fork, byte parity, refcounts and
  granules released.
- Prefix cache: radix hit reuses pages; metrics `prefix_hits`/`reused_tokens`;
  eviction unmaps.
- Swap/restore: evict a session, restore, continue; granules released/re-mapped.
- Exhaustion: undersized pool → retryable 503; engine healthy; a later request
  succeeds after pages are released.
- MTP/DFlash/vision smoke if the paged route supports them (they are
  orthogonal to the mapper, but the step provisioning and copy paths differ).
- Exit: all of the above on gfx1100.

### Phase 4: arch/model matrix and default decision

- gfx1100, gfx1151, gfx1201; 4B/9B/27B; AR, MTP, DFlash; vision; prefix cache;
  swap.
- Record VRAM at load, page/granule counts, 4×40k concurrency, TTFT/decode,
  COW cost, and refusal behavior.
- Decide whether `--multi-slot-paged-backend vmm` becomes the default for
  full-residency legacy-backend multi-slot loads. Keep the arena-paged route
  and VMM route as explicit opt-ins; document every refused combination.

## Acceptance criteria

1. **VA-only load:** a 4-slot 27B q8 load with auto sizing reserves VMM VA and
   maps no KV granules before requests; free VRAM at load is at least the VMM
   route's baseline (no 6 GB pool preallocation).
2. **4 × 40k:** four concurrent `max_tokens=40000` requests are admitted, all
   four generate in lockstep, and mapped physical KV stays within the budget.
3. **Release/reuse:** after the four sessions close/evict, their granules are
   unmapped and a second workload can reuse the physical budget without a
   process restart.
4. **COW/sharing:** a shared prefix is read by two slots; a write after fork
   copies only the touched pages; both slots remain byte-correct; refcounts
   and granule occupancy return to zero after release.
5. **Exhaustion:** a pool smaller than the request's page demand returns a
   retryable per-request 503 (`page demand exceeds pool`/map budget), never an
   engine failure; `/health` stays ok.
6. **Byte parity:** VMM-paged outputs match arena-paged for the same batch and
   state at widths 1..64; poisoned/unmapped positions trap.
7. **No mapping inside a step:** an assertion/trace proves all maps happen
   before the forward; a graph-captured step never maps.
8. **Arena route unchanged:** existing paged code objects and tests are
   byte-identical; the new variant is the only new kernel family.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Granule map/unmap cost per page (32 arenas) | Map per granule group, batch contiguous granules, measure in phase 2; a 2 MiB granule covers 15 pages. |
| COW copies 4.45 MiB/page | Measure agentic/prefix-tail workloads; pack COW pages into partially used granules; if unacceptable, revisit a two-level offset table. |
| Unmap while a stream reads | Completion event/lease per page before unmap; engine-thread-only mapping. |
| `VmmArena` has no arbitrary map/unmap | Phase 0 adds it with segment bookkeeping and fault-injection tests. |
| VA pressure from one arena per K/V/layer | Reserve once per load; no per-request VA churn; VA is ~5.8 GiB at full 27B cap. |
| Kernel ABI drift / stale descriptors | New `_vmm_paged` variant only; host/device mirror tests; digest existing arena objects. |
| Windows VMM growth unsafe | Refuse VMM-paged on Windows (Linux-only phase 1). |
| Graph capture with changing page tables | Disable capture until a stable-table test; provision before capture; invalidate on COW. |
| Auto pages + partial offload | Refuse auto, require explicit page counts; document. |

## Open questions

1. Should the mapper expose a **physical budget** in bytes or in granules? Use
   bytes at the API boundary, granules internally.
2. Should COW allocate a fresh granule per page or pack into existing private
   granules? Start with packing; measure tails.
3. Do we need per-layer page layouts for unequal K/V strides (future rotated
   tiers), or keep VMM-paged q8/fp8-only?
4. What is the default `n_pages` VA reservation: full `n_slots × cap` or a
   configurable upper bound lower than the cap? Full VA is cheap; use it
   unless VA measurements say otherwise.
5. Does the prefix radix cache need a page-cost model for VMM-paged (granule
   tails) before default-on?

## Rollout

Add `--multi-slot-paged-backend arena|vmm` (config
`serve.multi_slot_paged_backend`) while developing. The current
`--multi-slot-paged` keeps the arena-paged behavior until the VMM backend
passes phases 2–3. Only then consider making `vmm` the default for
full-residency legacy-backend multi-slot loads, with `arena` and VMM owners as
explicit opt-ins.
