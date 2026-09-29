// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Flat F16 KV-cache write.
//
// Layout: element `(pos, kv_head, d)` lives at
//   `pos * kv_dim + kv_head * head_dim + d`, one IEEE fp16 (2 bytes) each,
// where `kv_dim = n_kv_heads * head_dim`. No blocks, no per-block scale, no
// padding. Compare `kv_cache_write_q8_0.hip`, which packs 34-byte blocks of
// 32 elements and needs a wave reduction to find each block's amax — none of
// that exists here, so this kernel is a pure elementwise convert-and-store
// and the block/thread mapping is free to be a plain flat grid.
//
// The same kernel serves K and V: the caller launches it twice with different
// `dst`/`src`, exactly as the Q8_0 path does.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kv_slot_desc.h"

// Decode / single-position write.
// Grid: [ceil(kv_dim / 64), 1, 1]. Block: [64, 1, 1].
extern "C" __global__ void kv_cache_write_f16(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,    // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int kv_dim = n_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;

    const int pos = pos_buf[0];
    unsigned short* out =
        (unsigned short*)(dst + (size_t)pos * (size_t)kv_dim * 2);
    // IEEE fp16 round-to-nearest-even — __float2half_rn handles NaN/Inf/
    // overflow per the standard, so no custom rounding ladder is needed.
    out[i] = __half_as_ushort(__float2half_rn(src[i]));
}

// Batched prefill write.
// Grid: [ceil(kv_dim / 64), batch_size, 1]. Block: [64, 1, 1].
//
// `kv_offset_for_k` is used for both the K and the V launch, mirroring
// `kv_cache_write_q8_0_batched`. That is correct because `dst` already
// selects the K or V arena; the descriptor only adds a slot base within it.
// `slot_descs == nullptr` selects the legacy single-slab layout (base 0).
extern "C" __global__ void kv_cache_write_f16_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,              // [batch_size × kv_dim]
    const int* __restrict__ positions,          // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
    const int kv_dim = n_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (i >= kv_dim) return;

    const int slot = (row_slot != nullptr) ? row_slot[bid] : 0;
    const KvSlotDesc desc =
        (slot_descs != nullptr) ? slot_descs[slot] : kv_slot_legacy(0, 0);
    const int per_pos_bytes = kv_dim * 2;

    const int pos = positions[bid];
    unsigned short* out =
        (unsigned short*)(dst + kv_offset_for_k(desc, pos, per_pos_bytes));
    out[i] = __half_as_ushort(__float2half_rn(src[(size_t)bid * (size_t)kv_dim + i]));
}
