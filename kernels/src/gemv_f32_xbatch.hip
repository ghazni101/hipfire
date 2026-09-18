// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// F16-weight variants of the two LoRA GEMVs below (P2b: store the rank-128
// LoRA A/B factors as __half to halve the ~1.36 GB/fwd weight read — the
// dominant draft-forward cost). x and y stay F32 (activations/accumulators);
// only the weight matrix is quantized, so the delta accumulates in F32 and
// the small rank-128 delta stays well inside the tie-tolerant identity gate.
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_GEMV_F32_XBATCH_MAX
#define HIPFIRE_GEMV_F32_XBATCH_MAX 8
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_f16_xbatch(
    const __half* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B, int accumulate
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const __half* arow = A + (size_t)row * K;

    float acc[HIPFIRE_GEMV_F32_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) acc[b] = 0.0f;

    for (int i = tid; i < K; i += 32) {
        const float w = __half2float(arow[i]);
        #pragma unroll
        for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
            if (b >= B) break;
            acc[b] += w * x[(size_t)b * K + i];
        }
    }

    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
        if (b >= B) break;
        float v = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) {
            float* yb = y + (size_t)b * M + row;
            *yb = accumulate ? *yb + v : v;
        }
    }
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_f16_xbatch_splitk(
    const __half* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B, int splits
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int slice = (K + splits - 1) / splits;
    const int k0 = s * slice;
    const int k1 = k0 + slice < K ? k0 + slice : K;
    const __half* arow = A + (size_t)row * K;

    float acc[HIPFIRE_GEMV_F32_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) acc[b] = 0.0f;

    for (int i = k0 + tid; i < k1; i += 32) {
        const float w = __half2float(arow[i]);
        #pragma unroll
        for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
            if (b >= B) break;
            acc[b] += w * x[(size_t)b * K + i];
        }
    }

    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
        if (b >= B) break;
        float v = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) atomicAdd(y + (size_t)b * M + row, v);
    }
}

// F32 x-batched GEMV: one weight row dotted against B activation rows.
//
// Built for the Uno conditional-LoRA deltas (rank-128 factors over 3..7
// noise rows), where the generic one-thread-per-output `gemm_f32_batched`
// launches 504 tiny GEMMs per draft pass with poor shapes: the A side is
// m=rank (128 blocks, k=4096..12288) and the B side re-reads per-row.
// This kernel keeps the proven GEMV shape — grid = M output rows, one
// 32-thread workgroup per row streaming K — and simply carries B
// accumulators, with an optional read-modify-write epilogue so the B-side
// delta lands directly on the base projection (y += B·(A·x), no separate
// add launch, no delta scratch).
//
// A is [M × K] row-major F32; x is [B × K] row-major; y is [B × M]
// row-major. `accumulate != 0` → y[b*M+row] += acc, else plain store.
// B must be ≤ 8 (accumulator arrays are sized to XBATCH_MAX).

#ifndef HIPFIRE_GEMV_F32_XBATCH_MAX
#define HIPFIRE_GEMV_F32_XBATCH_MAX 8
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_f32_xbatch(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B, int accumulate
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const float* arow = A + (size_t)row * K;

    float acc[HIPFIRE_GEMV_F32_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) acc[b] = 0.0f;

    for (int i = tid; i < K; i += 32) {
        const float w = arow[i];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
            if (b >= B) break;
            acc[b] += w * x[(size_t)b * K + i];
        }
    }

    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
        if (b >= B) break;
        float v = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) {
            float* yb = y + (size_t)b * M + row;
            *yb = accumulate ? *yb + v : v;
        }
    }
}

// Split-K companion for small-M/large-K shapes (the LoRA A side: M=rank=128,
// K=4096..12288 — the plain kernel would launch only M blocks and starve the
// GPU). Grid is (M, splits); block (row, s) reduces its K slice for all B
// rows and atomicAdd's the partials. `y` MUST be zeroed first.
__launch_bounds__(32, 8)
extern "C" __global__ void gemv_f32_xbatch_splitk(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B, int splits
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int slice = (K + splits - 1) / splits;
    const int k0 = s * slice;
    const int k1 = k0 + slice < K ? k0 + slice : K;
    const float* arow = A + (size_t)row * K;

    float acc[HIPFIRE_GEMV_F32_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) acc[b] = 0.0f;

    for (int i = k0 + tid; i < k1; i += 32) {
        const float w = arow[i];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
            if (b >= B) break;
            acc[b] += w * x[(size_t)b * K + i];
        }
    }

    #pragma unroll
    for (int b = 0; b < HIPFIRE_GEMV_F32_XBATCH_MAX; b++) {
        if (b >= B) break;
        float v = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down(v, offset);
        if (tid == 0) atomicAdd(y + (size_t)b * M + row, v);
    }
}
