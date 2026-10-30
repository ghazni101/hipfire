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

// ---------------------------------------------------------------------------
// v2 (review/iu4): R rows per wave, 16 codes per lane per step (one dwordx4
// load), x read once per lane-chunk and reused across the R rows. Lane `l`
// takes 16-element chunk c = l, l+32, ... (chunk c lies in 32-block c>>1).
// Reduction order differs from v1/gemv_q8_0 (not bit-identical to them);
// contract is still dequant_q8 . x. Requires 16B-aligned rows
// ((K + K/16) % 16 == 0) and 16B-aligned x; the launcher falls back to v1.
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
    const long long rb = (long long)K + (long long)(K >> 4);
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
            const float d = (float)*((const _Float16*)(base[r] + K + ((c >> 1) << 1)));
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

// ── W8A8 variant (opt-in, HIPFIRE_IU8_GEMV=dot4) ─────────────────────────
// v2's lane map; x is quantized once per chunk to signed int8 per 32
// (d_x = amax/127, the pair of lanes c, c^1 owns one 32-group) and reused
// over the R rows. Per 16-element chunk: 4x V_DOT4_I32_IU8 with both sign bits set (RDNA3 ISA
// §16.10 VOP3P opcode 22, signed x signed), then y += (d_w * d_x) * dot.
#if defined(__GFX11__) || defined(__GFX12__)
__device__ __forceinline__ int iu8_pack4(float a, float b, float c, float d, float id) {
    const int qa = max(-127, min(127, (int)rintf(a * id)));
    const int qb = max(-127, min(127, (int)rintf(b * id)));
    const int qc = max(-127, min(127, (int)rintf(c * id)));
    const int qd = max(-127, min(127, (int)rintf(d * id)));
    return (qa & 0xFF) | ((qb & 0xFF) << 8) | ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
}

template <int R, bool ADD>
__device__ __forceinline__ void iu8g32_a8_body(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int lane = (int)(threadIdx.x & 31);
    const int wave = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const int row0 = wave * R;
    if (row0 >= M) return;
    const long long rb = (long long)K + (long long)(K >> 4);
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
        float amax = fmaxf(fmaxf(fmaxf(fabsf(x0.x), fabsf(x0.y)), fmaxf(fabsf(x0.z), fabsf(x0.w))),
                           fmaxf(fmaxf(fabsf(x1.x), fabsf(x1.y)), fmaxf(fabsf(x1.z), fabsf(x1.w))));
        amax = fmaxf(amax, fmaxf(fmaxf(fmaxf(fabsf(x2.x), fabsf(x2.y)), fmaxf(fabsf(x2.z), fabsf(x2.w))),
                                 fmaxf(fmaxf(fabsf(x3.x), fabsf(x3.y)), fmaxf(fabsf(x3.z), fabsf(x3.w)))));
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        const float dx = amax * (1.0f / 127.0f);
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        const int a0 = iu8_pack4(x0.x, x0.y, x0.z, x0.w, id);
        const int a1 = iu8_pack4(x1.x, x1.y, x1.z, x1.w, id);
        const int a2 = iu8_pack4(x2.x, x2.y, x2.z, x2.w, id);
        const int a3 = iu8_pack4(x3.x, x3.y, x3.z, x3.w, id);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = *(const uint4*)(base[r] + (c << 4));
            const float d = (float)*((const _Float16*)(base[r] + K + ((c >> 1) << 1)));
            int dot = __builtin_amdgcn_sudot4(true, (int)w.x, true, a0, 0, false);
            dot = __builtin_amdgcn_sudot4(true, (int)w.y, true, a1, dot, false);
            dot = __builtin_amdgcn_sudot4(true, (int)w.z, true, a2, dot, false);
            dot = __builtin_amdgcn_sudot4(true, (int)w.w, true, a3, dot, false);
            acc[r] += (d * dx) * (float)dot;
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

#define IU8G32_A8(R)                                                                 \
    extern "C" __global__ __launch_bounds__(128) void gemv_iu8g32_a8_r##R(           \
        const unsigned char* __restrict__ A, const float* __restrict__ x,            \
        float* __restrict__ y, int M, int K) {                                       \
        iu8g32_a8_body<R, false>(A, x, y, M, K);                                     \
    }                                                                                \
    extern "C" __global__ __launch_bounds__(128) void gemv_iu8g32_a8_residual_r##R(  \
        const unsigned char* __restrict__ A, const float* __restrict__ x,            \
        float* __restrict__ y, int M, int K) {                                       \
        iu8g32_a8_body<R, true>(A, x, y, M, K);                                      \
    }
IU8G32_A8(1)
IU8G32_A8(2)
IU8G32_A8(4)
#endif
