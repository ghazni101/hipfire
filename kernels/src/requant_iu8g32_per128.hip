// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// In-place qt59 (IU8G32) -> int8-per-128 requantization.
//
// Row layout is unchanged, so every existing qt59 reader stays correct:
//   [K signed int8 codes][K/32 f16 scales]
// The 4 per-32 scales of each 128-weight block are replaced by copies of one
// per-128 scale, and the codes are requantized linearly to that scale.
// Readers that fold per 32 see four equal scales (same value as per-128);
// a per-128 consumer reads the first slot and can accumulate the full K128.
//
// One warp per (row, K128 block). Lane l owns K elements 4l..4l+3 of the
// block, all inside one 32-group (g = l >> 3). The block max is a warp
// reduction; each lane rewrites only its own 4 bytes, so in-place is safe.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void requant_iu8g32_per128(
    char* __restrict__ A,
    int M,
    int K
) {
    const int lane = (int)(threadIdx.x & 31);
    const int idx = (int)((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5));
    const int nblk = K / 128;
    const int row = idx / nblk;
    const int kb = idx - row * nblk;
    if (row >= M) return;

    const long long stride = (long long)K + (long long)(K >> 4);
    char* base = A + (long long)row * stride;
    const int g = lane >> 3;
    const int k0 = kb * 128 + lane * 4;

    unsigned short sb;
    __builtin_memcpy(&sb, base + K + (long long)(kb * 4 + g) * 2, 2);
    const _Float16 sc = __ushort_as_half(sb);

    int c[4];
    float w[4];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i] = (int)(signed char)base[k0 + i];
        w[i] = (float)c[i] * (float)sc;
        amax = fmaxf(amax, fabsf(w[i]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));

    const float d = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int q = 0;
        if (d > 0.0f) {
            const float qf = rintf(__fdiv_rn(w[i], d));
            q = (int)fminf(fmaxf(qf, -127.0f), 127.0f);
        }
        base[k0 + i] = (char)q;
    }
    if (lane == 0) {
        const _Float16 hd = (_Float16)d;
        const unsigned short bits = __half_as_ushort(hd);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_memcpy(base + K + (long long)(kb * 4 + i) * 2, &bits, 2);
        }
    }
}
