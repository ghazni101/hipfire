// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MOE_DOWN_LAST_COMBINE
#define HIPFIRE_MOE_DOWN_LAST_COMBINE 0
#endif

#ifndef HIPFIRE_MOE_DOWN_KERNEL
#define HIPFIRE_MOE_DOWN_KERNEL gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded
#endif

// V2 (MQ4G256V2, qt=44) atomic-free indexed MoE down GEMV.
//
// Generalized from gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip
// (the V1 / qt=13 kernel) with two changes:
//   1. The 8-byte group header is decoded as two fp16 (scale, zero) pairs
//      (V2 per-128 layout) instead of one f32 (scale, zero) pair (V1 per-256).
//   2. General group loop — the V1 kernel was hardcoded for groups==2 (K=512).
//      K2-Horizon needs K=768 (3 groups) for MoE down and K=2560 (10 groups)
//      for MoVA v_experts.
//
// Fix: per-row header decode. The original V2 kernel decoded the header only
// for row0 and reused sc/zp for all 4 rows — a correctness bug. This version
// decodes independently per row.
//
// V2 header layout (136 B/group stride, nibble payload at offset 8):
//   [0..2) fp16 s0  (scale for weights 0-127,  threads 0-15)
//   [2..4) fp16 z0  (zero  for weights 0-127,  threads 0-15)
//   [4..6) fp16 s1  (scale for weights 128-255, threads 16-31)
//   [6..8) fp16 z1  (zero  for weights 128-255, threads 16-31)

__device__ __forceinline__ void v2_decode_header(
    const char* gp, int tid, float& sc, float& zp
) {
    const unsigned int hA = *(const unsigned int*)(gp);
    const unsigned int hB = *(const unsigned int*)(gp + 4);
    const unsigned int hs = (tid < 16) ? hA : hB;
    sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
    zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
}

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MOE_DOWN_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float*       __restrict__ expert_outputs,
#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    const float* __restrict__ topk_weights,
    float* __restrict__ x_residual,
#endif
    int M, int K, int K_TOP
) {
    const int row0 = blockIdx.x * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
    const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Simple group loop with per-row V2 header decode.
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_ptr0 + g * 136;
        const char* gp1 = row_ptr1 + g * 136;
        const char* gp2 = row_ptr2 + g * 136;
        const char* gp3 = row_ptr3 + g * 136;

        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        v2_decode_header(gp0, tid, sc0, zp0);
        v2_decode_header(gp1, tid, sc1, zp1);
        v2_decode_header(gp2, tid, sc2, zp2);
        v2_decode_header(gp3, tid, sc3, zp3);

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;
        float x0 = x[base], x1 = x[base+1], x2 = x[base+2], x3 = x[base+3];
        float x4 = x[base+4], x5 = x[base+5], x6 = x[base+6], x7 = x[base+7];

        #define DOG(pk, sc, zp, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7

        DOG(pk0, sc0, zp0, acc0);
        DOG(pk1, sc1, zp1, acc1);
        DOG(pk2, sc2, zp2, acc2);
        DOG(pk3, sc3, zp3, acc3);
        #undef DOG
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset);
        acc1 += __shfl_down(acc1, offset);
        acc2 += __shfl_down(acc2, offset);
        acc3 += __shfl_down(acc3, offset);
    }

    if (tid == 0) {
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc0;
        if (have_row1) expert_outputs[out_base + row1] = acc1;
        if (have_row2) expert_outputs[out_base + row2] = acc2;
        if (have_row3) expert_outputs[out_base + row3] = acc3;
    }

#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    int prior = -1;
    const int row_tiles = (M + 3) / 4;
    int* counters = reinterpret_cast<int*>(
        expert_outputs + (long long)gridDim.z * K_TOP * M);
    const int counter_index = bid * row_tiles + blockIdx.x;
    if (tid == 0) {
        __threadfence();
        prior = atomicAdd(counters + counter_index, 1);
    }
    prior = __shfl(prior, 0);
    if (prior == K_TOP - 1) {
        if (tid < 4) {
            const int row = row0 + tid;
            if (row < M) {
                const long long token_base = (long long)bid * K_TOP * M;
                const int weights_base = bid * K_TOP;
                float combined = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < K_TOP) {
                        combined += topk_weights[weights_base + k]
                                  * expert_outputs[token_base + (long long)k * M + row];
                    }
                }
                x_residual[(long long)bid * M + row] += combined;
            }
        }
        __syncwarp();
        if (tid == 0)
            atomicExch(counters + counter_index, 0);
    }
#endif
}
