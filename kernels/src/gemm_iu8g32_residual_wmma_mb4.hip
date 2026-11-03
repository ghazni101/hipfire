// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// IU8G32 (qt59) WMMA residual GEMM with 4x batch-tile fanout per wave.
//
// Sibling of gemm_q8_0_residual_wmma.hip (same f16 WMMA math and output
// mapping), but each 16-row wave walks four 16-token batch tiles before it
// reloads and re-dequantizes its weight tile. The base kernel repeats that
// weight work once per 16-token tile; at N=2048 that is 128 redundant
// dequantizations and global re-reads of every weight row per M tile.
//
// Grid: [ceil(M/16), ceil(N/64)]. Block: [32] (wave32). LDS: 0 B.
// Each Y cell is still written by exactly one lane, so the non-atomic
// `+=` residual invariant of the base kernel is preserved.
//
// Row layout (qt59): [K int8 codes][K/32 f16 scales], 34 B per 32.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_iu8g32_residual_wmma_mb4(
    const char* __restrict__ A_q8,
    const _Float16* __restrict__ X,      // [N, K] FP16
    float* __restrict__ Y,               // [N, M] FP32 - caller seeds residual
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 64;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= N) return;

    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)K + (long long)(K >> 6);
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    // Four 16-token batch tiles; lanes past N read row 0 (results dropped).
    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + (tid & 15);
        safe_batch_s[s] = (b_idx < N) ? (long long)b_idx : 0LL;
    }

    float8_t acc[4] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* cp = row_base + bi * 32;
        unsigned short s_bits;
        __builtin_memcpy(&s_bits, row_base + K + ((bi >> 2) << 1), 2);
        _Float16 sc;
        __builtin_memcpy(&sc, &s_bits, 2);

        const _Float16* x_base = X + bi * 32;
        const signed char* w0 = (const signed char*)(cp);
        const signed char* w1 = (const signed char*)(cp + 16);

        half16_t a0;
        #pragma unroll
        for (int i = 0; i < 16; i++) a0[i] = sc * (_Float16)(float)(int)w0[i];
        half16_t a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) a1[i] = sc * (_Float16)(float)(int)w1[i];

        #pragma unroll
        for (int s = 0; s < 4; s++) {
            const _Float16* xs = x_base + safe_batch_s[s] * (long long)K;
            half16_t b0 = *(const half16_t*)(xs);
            half16_t b1 = *(const half16_t*)(xs + 16);
            acc[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc[s]);
            acc[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc[s]);
        }
    }

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int out_col = batch_start + 16 * s + (tid & 15);
        if (out_col >= N) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] += acc[s][j];
            }
        }
    }
}
