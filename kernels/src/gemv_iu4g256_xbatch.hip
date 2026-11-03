// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// x-batched decode GEMV for IU4G256 (qt 57): one weight row is decoded ONCE
// and dotted against B separate activation vectors. It is the IU4 twin of
// `gemv_mq4g256v2_xbatch.hip`, with the planar qt 57 address math of
// `gemv_iu4g256`.
//
// Why: the multi-slot decode path runs one GEMV launch per slot, so every
// projection's weight row is read once per slot. Decode is weight-bandwidth
// bound, so B slots cost B full weight passes per step. Every slot wants the
// same weights against a different x; the reads collapse to one pass here.
//
// Row layout (qt 57): `[K/2 code bytes][K/256 * 8 B headers]`, header
// `f16 sc0, f16 zp0, f16 sc1, f16 zp1`; low nibble = even K. `w = sc*code + zp`
// with sc/zp selected by `tid < 16` (first/second 128 of the group). x is
// already FWHT-256 rotated.
//
// Accuracy: each acc[b] accumulates the same terms in the same source order
// as `gemv_iu4g256` / `gemv_iu4g256_residual`; only the x vector index changes,
// and the weight decode is hoisted out of the b loop (a pure CSE). The two
// files' different register pressure can still pick different FMA
// contractions, so the result matches the per-token GEMV to f32 rounding
// (measured <= 2e-7 relative on the model's projection shapes) rather than
// bit for bit.
//
// x is [B x K] row-major, y is [B x M] row-major, so y[b*M + row] lands where
// a caller holding an [n_slots x M] block wants it.
//
// HIPFIRE_IU4G256_RESIDUAL_EPILOGUE selects `y[b] += acc`; the symbol is named
// by HIPFIRE_IU4G256_XBATCH_KERNEL.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef HIPFIRE_IU4G256_XBATCH_KERNEL
#define HIPFIRE_IU4G256_XBATCH_KERNEL gemv_iu4g256_xbatch
#endif

#ifndef HIPFIRE_IU4G256_XBATCH_MAX
#define HIPFIRE_IU4G256_XBATCH_MAX 4
#endif

// One group header: `f16 sc, f16 zp` (the second 4-byte pair covers the
// second 128 of the group and is selected by `hsel` at the load site).
__device__ __forceinline__ float2 iu4g32_hdr(const unsigned int hs) {
    return make_float2(
        __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))),
        __half2float(__ushort_as_half((unsigned short)(hs >> 16))));
}

// Occupancy 8, not the scalar kernel's 16: B accumulator sets multiply the
// live float state by B. Asking for 16 waves/SIMD at B=4 makes the compiler
// spill, and a spilled accumulator changes the FMA sequence.
__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_IU4G256_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K >> 8;
    const long long row_bytes = (long long)groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;
    const char* codes = row_ptr;
    const char* hdr = row_ptr + (long long)groups_per_row * 128;
    const int boff = tid * 4;
    const int hsel = (tid < 16) ? 0 : 4;

    float acc0[HIPFIRE_IU4G256_XBATCH_MAX];
    float acc1[HIPFIRE_IU4G256_XBATCH_MAX];
    float acc2[HIPFIRE_IU4G256_XBATCH_MAX];
    float acc3[HIPFIRE_IU4G256_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    #define IU4XDOG(pk, sc, zp, bs, a, xb) \
        (a) += (sc * (float)((pk) & 0xFu)           + zp) * (xb)[(bs)]      \
             + (sc * (float)(((pk) >> 4) & 0xFu)    + zp) * (xb)[(bs) + 1]  \
             + (sc * (float)(((pk) >> 8) & 0xFu)    + zp) * (xb)[(bs) + 2]  \
             + (sc * (float)(((pk) >> 12) & 0xFu)   + zp) * (xb)[(bs) + 3]  \
             + (sc * (float)(((pk) >> 16) & 0xFu)   + zp) * (xb)[(bs) + 4]  \
             + (sc * (float)(((pk) >> 20) & 0xFu)   + zp) * (xb)[(bs) + 5]  \
             + (sc * (float)(((pk) >> 24) & 0xFu)   + zp) * (xb)[(bs) + 6]  \
             + (sc * (float)(((pk) >> 28) & 0xFu)   + zp) * (xb)[(bs) + 7]

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const float2 h0 = iu4g32_hdr(*(const unsigned int*)(hdr + (g + 0) * 8 + hsel));
        const float2 h1 = iu4g32_hdr(*(const unsigned int*)(hdr + (g + 1) * 8 + hsel));
        const float2 h2 = iu4g32_hdr(*(const unsigned int*)(hdr + (g + 2) * 8 + hsel));
        const float2 h3 = iu4g32_hdr(*(const unsigned int*)(hdr + (g + 3) * 8 + hsel));
        const float sc0 = h0.x, zp0 = h0.y;
        const float sc1 = h1.x, zp1 = h1.y;
        const float sc2 = h2.x, zp2 = h2.y;
        const float sc3 = h3.x, zp3 = h3.y;

        const char* cp = codes + g * 128 + boff;
        const unsigned int pk0 = *(const unsigned int*)(cp);
        const unsigned int pk1 = *(const unsigned int*)(cp + 128);
        const unsigned int pk2 = *(const unsigned int*)(cp + 256);
        const unsigned int pk3 = *(const unsigned int*)(cp + 384);

        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            IU4XDOG(pk0, sc0, zp0, base,       acc0[b], xb);
            IU4XDOG(pk1, sc1, zp1, base + 256, acc1[b], xb);
            IU4XDOG(pk2, sc2, zp2, base + 512, acc2[b], xb);
            IU4XDOG(pk3, sc3, zp3, base + 768, acc3[b], xb);
        }
    }

    // Tail groups accumulate into acc[g % 4], the scalar kernel's 4-way
    // interleaving the combine below assumes. Three separate blocks (not a
    // runtime `t` switch) keep the per-group source shape identical to the
    // scalar kernel's tail.
    if (tail >= 1) {
        const int g = quads << 2;
        const float2 ht = iu4g32_hdr(*(const unsigned int*)(hdr + g * 8 + hsel));
        const float sc = ht.x, zp = ht.y;
        const unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            IU4XDOG(pk, sc, zp, base, acc0[b], xb);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const float2 ht = iu4g32_hdr(*(const unsigned int*)(hdr + g * 8 + hsel));
        const float sc = ht.x, zp = ht.y;
        const unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            IU4XDOG(pk, sc, zp, base, acc1[b], xb);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const float2 ht = iu4g32_hdr(*(const unsigned int*)(hdr + g * 8 + hsel));
        const float sc = ht.x, zp = ht.y;
        const unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            IU4XDOG(pk, sc, zp, base, acc2[b], xb);
        }
    }
    #undef IU4XDOG

    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU4G256_XBATCH_MAX; b++) {
        if (b >= B) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
#if defined(HIPFIRE_IU4G256_RESIDUAL_EPILOGUE)
        if (tid == 0) y[(long long)b * M + row] += acc;
#else
        if (tid == 0) y[(long long)b * M + row] = acc;
#endif
    }
}
