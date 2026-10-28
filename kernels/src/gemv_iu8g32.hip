// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU8G32 (qt 59), the lifted-role tier of an iu4/iu8 file.
//
// The grid is Q8F16's (signed int8 codes, one f16 scale per 32, no FWHT), but
// the row stores its scales at the tail so the code plane is one contiguous
// operand:
//
//   [K signed codes][f16 scale] * (K / 32)
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
        const float d = (float)*((const _Float16*)(scales + (bi << 1)));
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
    const long long row_bytes = (long long)K + (long long)(blocks << 1);
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
    const long long row_bytes = (long long)K + (long long)(blocks << 1);
    float sum = iu8g32_row_dot(A + (long long)row * row_bytes, x, K);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if (t == 0) y[row] = y[row] + sum;
}
