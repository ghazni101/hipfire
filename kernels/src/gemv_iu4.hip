// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU4G256 (qt 57). One wave per output row.
//
// The weight dword is the planar IU4 code plane: low nibble = even K, which
// is nibble 0 of `v_dot8_i32_iu4`. It is loaded and issued unsigned. There is
// no nibble expand. The activation is signed int4, one scale per 32 elements
// (`d = max(|x|) / 7`), packed in the same nibble order. The fold per eight
// weights is the q8x identity, on this operand:
//
//   acc += (d * scale) * dot(q, xq) + zero * (d * sumq)
//
// `x` is already FWHT-256 rotated. The integer dot is exact. The activation
// quant is the approximation. The CPU contract is
// `hipfire_runtime::iu_quant::gemv_iu4_dot8`. Prefill does not use this kernel.
// qt 58 and qt 59 are a different operand and are not read here.
//
// `gemv_iu4g256_residual` is the fused `y += W·x` twin (`y` is both in and
// out), used by a body-dtype `ssm_out` decode step. With `y` pre-zeroed it is
// the SET kernel plus one f32 add at the store.
//
// RDNA3 and RDNA4 only (`v_dot8_i32_iu4`). Wave32, block 32.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ int iu4_pack_i4(const float v[8], float id) {
    int packed = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int q = (int)rintf(v[j] * id);
        q = max(-8, min(7, q));
        packed |= (q & 0xF) << (4 * j);
    }
    return packed;
}

// a_sign, a, b_sign, b, acc, clamp. Weight unsigned, activation signed.
__device__ __forceinline__ int iu4_dot8(int weight, int act) {
    return __builtin_amdgcn_sudot8(false, weight, true, act, 0, false);
}

__device__ __forceinline__ int iu4_sum_i4(int act) {
    return __builtin_amdgcn_sudot8(false, 0x11111111, true, act, 0, false);
}

__device__ __forceinline__ float iu4_row_dot(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    int K) {
    const int groups = K >> 8;
    const unsigned char* codes = A;
    const unsigned char* scales = codes + (K >> 1);
    const int t = (int)threadIdx.x;

    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const float* xp = x + ((long long)g << 8) + (t << 3);
        const float4 a = *(const float4*)xp;
        const float4 b = *(const float4*)(xp + 4);
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        const float d = amax * (1.0f / 7.0f);
        const float id = amax > 0.0f ? 7.0f / amax : 0.0f;
        const int act = iu4_pack_i4(v, id);
        const int weight = *(const int*)(codes + (g << 7) + (t << 2));
        const int dot = iu4_dot8(weight, act);
        const int sumq = iu4_sum_i4(act);
        const unsigned h = *(const unsigned*)(scales + (g << 3) + ((t >> 4) << 2));
        const float sc = __half2float(__ushort_as_half((unsigned short)h));
        const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        acc += (d * sc) * (float)dot + zp * (d * (float)sumq);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
    return acc;
}

extern "C" __global__ void gemv_iu4g256(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)(K >> 1) + (K >> 5);
    const float acc = iu4_row_dot(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = acc;
}

extern "C" __global__ void gemv_iu4g256_residual(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)(K >> 1) + (K >> 5);
    const float acc = iu4_row_dot(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = y[row] + acc;
}
