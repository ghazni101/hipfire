// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multi-slot KV descriptor — the single point of KV address translation for
// every batched attention kernel.
//
// A "slot" is one independent sequence in a batch. Each slot's KV is stored
// in a paged block table: `block_table[logical_page]` gives the physical
// page index, and the byte offset is:
//
//   phys_page * page_tokens * per_pos_bytes + page_off * per_pos_bytes
//
// When `block_table` is null the descriptor is in legacy contiguous mode and
// translation degenerates to `legacy_base + pos * stride`, byte-identical to
// the pre-paged kernel. This keeps every kernel on ONE code path: callers
// either pass a real paged descriptor or a legacy fallback built on the stack.
//
// Layout must stay byte-identical to the Rust mirror in
// crates/rdna-compute/src/kv_slots.rs. 24 bytes, 8-byte aligned.

#pragma once

#include <hip/hip_runtime.h>

struct KvSlotDesc {
    // GPU pointer to array of physical page indices, or nullptr for legacy
    // contiguous mode. In paged mode the array has `ceil(seq_len / page_tokens)`
    // valid entries; the kernel only reads entries for positions < seq_len.
    const int* __restrict__ block_table;

    // Byte offset of this slot's K/V slab in the arena, used ONLY in legacy
    // mode (block_table == nullptr). For Q8_0 the flash-prefill kernel uses
    // one shared slab offset, so K and V must sit at the same offset in their
    // respective arenas — legacy_base serves as both k_base and v_base.
    unsigned long long legacy_base;

    int seq_len;        // logical KV length; kernel reads [0, seq_len)
    int page_tokens;    // tokens per page (e.g. 128); 0 = legacy contiguous mode
};

// Byte offset of position `pos` within the arena.
// `per_pos_bytes` is the per-position stride in bytes, uniform across slots
// (n_kv_heads * (head_dim/32) * 34 for Q8_0).
__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    if (s.block_table != nullptr) {
        const int page_idx = pos / s.page_tokens;
        const int page_off = pos % s.page_tokens;
        const int phys_page = s.block_table[page_idx];
        return (unsigned long long)phys_page
             * (unsigned long long)s.page_tokens
             * (unsigned long long)per_pos_bytes
           + (unsigned long long)page_off
             * (unsigned long long)per_pos_bytes;
    }
    return s.legacy_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    // In paged mode K and V share the same block table (same page layout).
    // In legacy mode legacy_base serves as both k_base and v_base (Q8_0 ABI).
    if (s.block_table != nullptr) {
        const int page_idx = pos / s.page_tokens;
        const int page_off = pos % s.page_tokens;
        const int phys_page = s.block_table[page_idx];
        return (unsigned long long)phys_page
             * (unsigned long long)s.page_tokens
             * (unsigned long long)per_pos_bytes
           + (unsigned long long)page_off
             * (unsigned long long)per_pos_bytes;
    }
    return s.legacy_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Single-slot fallback used when the descriptor pointer is null. Keeps the
// ported kernels on ONE code path: callers build this on the stack from the
// legacy scalar args, so there is no `if (descs) ... else ...` around every
// KV read. Behaviour is then bitwise identical to the pre-paged kernel.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.block_table = nullptr;
    s.legacy_base = 0ULL;
    s.seq_len = seq_len;
    s.page_tokens = 0;
    return s;
}

// Legacy fallback for kernels that also honour the pre-descriptor
// independent-sequence contract: a NEGATIVE `max_seq` whose magnitude is one
// lane's token capacity, with batch row `row` reading only its own
// `[row * cap, (row + 1) * cap)` slice of a lane-major arena. Folding that
// base into the synthesised descriptor keeps those kernels on the single
// kv_offset_for_*() address path instead of branching at every KV read.
// A positive `max_seq` reproduces `kv_slot_legacy` exactly (base 0, shared
// cache), so sequential prefill stays byte-for-byte unchanged.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy_lane(
    int seq_len, int max_seq, int row, int per_pos_bytes)
{
    const bool independent = max_seq < 0;
    const int lane_capacity = independent ? -max_seq : max_seq;
    const unsigned long long lane_bytes =
        (unsigned long long)lane_capacity * (unsigned long long)per_pos_bytes;
    const unsigned long long base =
        independent ? (unsigned long long)row * lane_bytes : 0ULL;

    KvSlotDesc s;
    s.block_table = nullptr;
    s.legacy_base = base;
    s.seq_len = seq_len;
    s.page_tokens = 0;
    return s;
}
