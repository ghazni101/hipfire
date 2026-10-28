// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU8G256 (qt 58): the MQ4V2 grid (FWHT-256, unsigned codes
// 0..=15, one f16 `(scale, zero)` pair per 128 weights) stored one byte per
// weight instead of packed nibbles.
//
//   [K code bytes][f16 scale, f16 zero] * (K / 128)
//
// The code plane is already expanded, so there is no nibble unpack. The math
// is the stored affine grid exactly: `w = f32(scale) * f32(code) + f32(zero)`
// and `acc += w * x` with `x` the FWHT-256 activation in f32 — the byte
// operand exists to remove the VALU expand, not to change the numerics. The
// CPU contract is `hipfire_runtime::iu_quant::dequant(Layout::Iu8)` · x
// (`gemv_iu8_dot`).
//
// `gemv_iu8g256_residual` is the fused `y += W·x` twin (`y` is both in and
// out), used by a body-dtype `ssm_out` decode step.
//
// `x` must already be FWHT-256 rotated (`dtype_rotation_plan(IU8G256)` is
// `FwhtG256`). Prefill does not use this kernel. Wave32, block 32; no
// ISA-specific intrinsic, so it runs on every arch.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float iu8g256_row_dot(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    int K) {
    const int blocks = K >> 5;
    const unsigned char* codes = A;
    const unsigned char* scales = codes + K;
    const int t = (int)threadIdx.x;

    float acc = 0.0f;
    for (int bi = 0; bi < blocks; ++bi) {
        // One (scale, zero) pair per 128 weights = four 32-blocks.
        const unsigned char* pair = scales + ((bi >> 2) << 2);
        const float sc = (float)*((const _Float16*)pair);
        const float zp = (float)*((const _Float16*)(pair + 2));
        const float q = (float)codes[(bi << 5) + t];
        acc += (sc * q + zp) * x[(bi << 5) + t];
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    return acc;
}

extern "C" __global__ void gemv_iu8g256(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)K + (long long)(K >> 7) * 4;
    const float acc = iu8g256_row_dot(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = acc;
}

extern "C" __global__ void gemv_iu8g256_residual(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)K + (long long)(K >> 7) * 4;
    const float acc = iu8g256_row_dot(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = y[row] + acc;
}
