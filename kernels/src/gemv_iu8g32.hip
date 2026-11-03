// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU8G32 (qt 59), the lifted-role tier of an iu4/iu8 file.
//
// The grid is Q8F16's (signed int8 codes, no FWHT), but the compact row
// carries ONE f16 scale per 128 at the tail so the code plane is one
// contiguous operand:
//
//   [K signed codes][K / 128 f16 scales]
//
// The math is Q8F16's, element for element: one wave per row, thread `t`
// folds element `t` of every 32-block and the warp reduces by shuffle, so
// this kernel is bit-identical to `gemv_q8_0` on the same grid (the only
// difference is where the scale is read from). `x` is the natural-basis
// activation; qt 59 is not FWHT-rotated.
//
// `gemv_iu8g32_residual` is the fused `y += W·x` twin used by the DeltaNet
// out_proj (ssm_out) decode step.
//
// Contract: `hipfire_runtime::iu_quant::dequant_q8` (the dot of the
// dequantized row with `x`). Wave32, block 32.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float iu8g32_row_dot(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    int K) {
    const int blocks = K >> 5;
    const unsigned char* codes = A;
    const unsigned char* scales = A + K;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; ++bi) {
        const float d = (float)*((const _Float16*)(scales + (((bi >> 2)) << 1)));
        const signed char q = (signed char)codes[(bi << 5) + threadIdx.x];
        sum += d * (float)q * x[(bi << 5) + threadIdx.x];
    }
    return sum;
}

extern "C" __global__ void gemv_iu8g32(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    const int t = (int)threadIdx.x;
    if (row >= M) return;

    const int blocks = K >> 5;
    const long long row_bytes = (long long)K + (long long)((blocks >> 2) << 1);
    float sum = iu8g32_row_dot(A + (long long)row * row_bytes, x, K);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if (t == 0) y[row] = sum;
}

extern "C" __global__ void gemv_iu8g32_residual(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    const int t = (int)threadIdx.x;
    if (row >= M) return;

    const int blocks = K >> 5;
    const long long row_bytes = (long long)K + (long long)((blocks >> 2) << 1);
    float sum = iu8g32_row_dot(A + (long long)row * row_bytes, x, K);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if (t == 0) y[row] = y[row] + sum;
}

// ---------------------------------------------------------------------------
// v2 (review/iu4): R rows per wave, 16 codes per lane per step (one dwordx4
// load), x read once per lane-chunk and reused across the R rows. Lane `l`
// takes 16-element chunk c = l, l+32, ... (chunk c lies in 32-block c>>1).
// Reduction order differs from v1/gemv_q8_0 (not bit-identical to them);
// contract is still dequant_q8 . x. Requires 16B-aligned rows
// ((K + K/64) % 16 == 0) and 16B-aligned x; the launcher falls back to v1.
__device__ __forceinline__ float iu8_dot16(uint4 w, float4 a, float4 b, float4 c, float4 e) {
    float s = 0.0f;
    const unsigned int ws[4] = {w.x, w.y, w.z, w.w};
    const float4 xs[4] = {a, b, c, e};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = (int)ws[j];
        s += (float)((v << 24) >> 24) * xs[j].x;
        s += (float)((v << 16) >> 24) * xs[j].y;
        s += (float)((v << 8) >> 24) * xs[j].z;
        s += (float)(v >> 24) * xs[j].w;
    }
    return s;
}

template <int R, bool ADD>
__device__ __forceinline__ void iu8g32_v2_body(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int lane = (int)(threadIdx.x & 31);
    const int wave = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const int row0 = wave * R;
    if (row0 >= M) return;
    const long long rb = (long long)K + (long long)(K >> 6);
    const int chunks = K >> 4;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    const unsigned char* base[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = (row0 + r < M) ? (row0 + r) : row0;
        base[r] = A + (long long)row * rb;
    }
    for (int c = lane; c < chunks; c += 32) {
        const float4* xp = (const float4*)(x + (c << 4));
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = *(const uint4*)(base[r] + (c << 4));
            const float d = (float)*((const _Float16*)(base[r] + K + ((c >> 3) << 1)));
            acc[r] += d * iu8_dot16(w, x0, x1, x2, x3);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off);
        acc[r] = s;
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < M) y[row] = ADD ? (y[row] + acc[r]) : acc[r];
        }
    }
}

#define IU8G32_V2(R)                                                                 \
    extern "C" __global__ __launch_bounds__(128) void gemv_iu8g32_v2_r##R(           \
        const unsigned char* __restrict__ A, const float* __restrict__ x,            \
        float* __restrict__ y, int M, int K) {                                       \
        iu8g32_v2_body<R, false>(A, x, y, M, K);                                     \
    }                                                                                \
    extern "C" __global__ __launch_bounds__(128) void gemv_iu8g32_v2_residual_r##R(  \
        const unsigned char* __restrict__ A, const float* __restrict__ x,            \
        float* __restrict__ y, int M, int K) {                                       \
        iu8g32_v2_body<R, true>(A, x, y, M, K);                                      \
    }
IU8G32_V2(1)
IU8G32_V2(2)
IU8G32_V2(4)
