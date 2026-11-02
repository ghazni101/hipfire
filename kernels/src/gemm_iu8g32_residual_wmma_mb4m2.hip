// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// IU8G32 (qt59) WMMA residual GEMM: 32 rows x 64 tokens per wave.
//
// Extends gemm_iu8g32_residual_wmma_mb4 with a second 16-row M tile per
// wave. Both tiles consume the same four 16-token X fragments, so a fixed
// M tile no longer re-reads X once per 16 output rows. Weight dequantization
// and X traffic are both amortized over 32 rows and 64 tokens.
//
// Grid: [ceil(M/32), ceil(N/64)]. Block: [32] (wave32). LDS: 0 B.
// Every Y cell is written by exactly one lane in one block, so the `+=`
// residual invariant of the base kernel is preserved. Per-output-row
// arithmetic is identical to gemm_iu8g32_residual_wmma (same K order, same
// f16 WMMA operands), hence bit-identical results.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_iu8g32_residual_wmma_mb4m2(
    const char* __restrict__ A_q8,
    const _Float16* __restrict__ X,      // [N, K] FP16
    float* __restrict__ Y,               // [N, M] FP32 - caller seeds residual
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 32;
    const int batch_start = blockIdx.y * 64;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= N) return;

    const int lane_row = tid & 15;
    const int my_row0 = row_start + lane_row;
    const int my_row1 = my_row0 + 16;
    const int safe_row0 = (my_row0 < M) ? my_row0 : (M - 1);
    const int safe_row1 = (my_row1 < M) ? my_row1 : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row0_base = A_q8 + (long long)safe_row0 * row_stride;
    const char* row1_base = A_q8 + (long long)safe_row1 * row_stride;

    long long safe_batch_s[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int b_idx = batch_start + 16 * s + lane_row;
        safe_batch_s[s] = (b_idx < N) ? (long long)b_idx : 0LL;
    }

    float8_t acc0[4] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };
    float8_t acc1[4] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };

    const bool sc_vec = (K & 255) == 0;
    uint4 sc0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 sc1 = make_uint4(0u, 0u, 0u, 0u);

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const int j = bi & 7;
        unsigned short s0_bits, s1_bits;
        if (sc_vec) {
            if (j == 0) {
                sc0 = *(const uint4*)(row0_base + K + bi * 2);
                sc1 = *(const uint4*)(row1_base + K + bi * 2);
            }
            const unsigned int w0 = (j < 2) ? sc0.x : (j < 4) ? sc0.y : (j < 6) ? sc0.z : sc0.w;
            const unsigned int w1 = (j < 2) ? sc1.x : (j < 4) ? sc1.y : (j < 6) ? sc1.z : sc1.w;
            s0_bits = (unsigned short)((j & 1) ? (w0 >> 16) : (w0 & 0xffffu));
            s1_bits = (unsigned short)((j & 1) ? (w1 >> 16) : (w1 & 0xffffu));
        } else {
            __builtin_memcpy(&s0_bits, row0_base + K + bi * 2, 2);
            __builtin_memcpy(&s1_bits, row1_base + K + bi * 2, 2);
        }
        _Float16 s0, s1;
        __builtin_memcpy(&s0, &s0_bits, 2);
        __builtin_memcpy(&s1, &s1_bits, 2);

        const char* cp0 = row0_base + bi * 32;
        const char* cp1 = row1_base + bi * 32;

        half16_t a0, a1, b0, b1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = s0 * (_Float16)(float)(int)((const signed char*)cp0)[i];
            b0[i] = s1 * (_Float16)(float)(int)((const signed char*)cp1)[i];
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a1[i] = s0 * (_Float16)(float)(int)((const signed char*)cp0)[16 + i];
            b1[i] = s1 * (_Float16)(float)(int)((const signed char*)cp1)[16 + i];
        }

        const _Float16* x_base = X + bi * 32;
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            const _Float16* xs = x_base + safe_batch_s[s] * (long long)K;
            half16_t x0 = *(const half16_t*)(xs);
            half16_t x1 = *(const half16_t*)(xs + 16);
            acc0[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, x0, acc0[s]);
            acc0[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, x1, acc0[s]);
            acc1[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b0, x0, acc1[s]);
            acc1[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b1, x1, acc1[s]);
        }
    }

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int out_col = batch_start + 16 * s + lane_row;
        if (out_col >= N) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int orow = 2 * j + (tid >> 4);
            const int r0 = row_start + orow;
            const int r1 = row_start + 16 + orow;
            const long long base = (long long)out_col * M;
            if (r0 < M) Y[base + r0] += acc0[s][j];
            if (r1 < M) Y[base + r1] += acc1[s][j];
        }
    }
}
