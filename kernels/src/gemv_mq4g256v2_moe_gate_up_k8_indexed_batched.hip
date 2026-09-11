// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// V2 (MQ4G256V2, qt=44) batched indexed MoE gate_up GEMV.
//
// Identical structure to gemv_hfq4g256_moe_gate_up_k8_indexed_batched.hip
// (the V1 / qt=13 kernel) with one change: the 8-byte group header is
// decoded as two fp16 (scale, zero) pairs — one per 128 weights — instead
// of one f32 (scale, zero) pair per 256 weights.
//
// V2 header layout (136 B/group stride, nibble payload at offset 8):
//   [0..2) fp16 s0  (scale for weights 0-127,  threads 0-15)
//   [2..4) fp16 z0  (zero  for weights 0-127,  threads 0-15)
//   [4..6) fp16 s1  (scale for weights 128-255, threads 16-31)
//   [6..8) fp16 z1  (zero  for weights 128-255, threads 16-31)

// Decode V2 group header. Returns (scale, zero) for this thread's 128-weight half.
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
extern "C" __global__ void gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

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

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc, zp;
        v2_decode_header(gp, tid, sc, zp);
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc, zp;
        v2_decode_header(gp, tid, sc, zp);
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc, zp;
        v2_decode_header(gp, tid, sc, zp);
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
