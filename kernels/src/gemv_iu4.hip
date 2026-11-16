// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU4G128 (qt 60).
//
// Row layout: [K/2 code bytes][f16 sc, f16 zp per 128] x K/128, zero-padded
// to 16 B (IU4_ROW_BYTES). Low nibble = even K. Dequant w = sc*code + zp.
// x is already FWHT-256 rotated.
//
// Default kernels (`gemv_iu4g128`, `gemv_iu4g128_residual`,
// `fused_iu4g128_rows`): f32 activation, wide loads. Block 128 = four
// 32-lane waves, one output row per wave. Lane l owns the 32-weight chunks
// c = l, l + 32, ...: one 16-byte code load at row + c*16, the (sc, zp) dword
// of its 128-group at row + K/2 + (c/4)*4, and x[c*32 .. c*32+31] as eight
// float4. Per chunk t = sum(code * x), xs = sum(x), acc += sc*t + zp*xs; then
// a shfl_xor tree. The row base must be 16-byte aligned (the launcher checks).
// Plain, residual and fused rows share iu4_row_dot_wide, so the fused output
// equals the plain kernel's bits.
//
// The `_dot4` kernels are W4A8 (int8 activation per 32, `v_dot4_i32_iu8`,
// HIPFIRE_IU4_DECODE=dot4, opt-in).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// qt60 rows are padded to a 16-byte multiple: K/2 code bytes + K/32 scale
// bytes, rounded up (hipfire_runtime::iu_quant::row_bytes). Wide loads need it.
#ifndef IU4_ROW_BYTES
#define IU4_ROW_BYTES(gpr) ((((long long)(gpr) * 136LL) + 15LL) & ~15LL)
#define IU4_ROW_BYTES_U(gpr) ((((unsigned int)(gpr) * 136u) + 15u) & ~15u)
#endif

__device__ __forceinline__ float iu4_row_dot_wide(
    const unsigned char* __restrict__ row,
    const float* __restrict__ x,
    int K) {
    const int lane = (int)(threadIdx.x & 31);
    const int chunks = K >> 5;
    const unsigned char* hdr = row + (K >> 1);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const float4* xp = (const float4*)(x + ((long long)c << 5));
        float xv[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 v = xp[i];
            xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
        }
        float xs = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) xs += xv[i];
        const uint4 pk = *(const uint4*)(row + ((long long)c << 4));
        const unsigned int h = *(const unsigned int*)(hdr + ((c >> 2) << 2));
        const unsigned int w[4] = {pk.x, pk.y, pk.z, pk.w};
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int n = 0; n < 8; ++n) t += (float)((w[j] >> (4 * n)) & 0xFu) * xv[8 * j + n];
        }
        const float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        acc += sc * t + zp * xs;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    return acc;
}

// Grid = ceil(M / 4), block 128: wave w of block b computes row 4b + w.
__launch_bounds__(128)
extern "C" __global__ void gemv_iu4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K) {
    const int row = (int)(blockIdx.x * 4 + (threadIdx.x >> 5));
    if (row >= M) return;
    const float acc = iu4_row_dot_wide(
        (const unsigned char*)A + (long long)row * IU4_ROW_BYTES(K >> 8), x, K);
    if ((threadIdx.x & 31) == 0) y[row] = acc;
}

// `y += W x`. Same grid and dot as gemv_iu4g128.
__launch_bounds__(128)
extern "C" __global__ void gemv_iu4g128_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K) {
    const int row = (int)(blockIdx.x * 4 + (threadIdx.x >> 5));
    if (row >= M) return;
    const float acc = iu4_row_dot_wide(
        (const unsigned char*)A + (long long)row * IU4_ROW_BYTES(K >> 8), x, K);
    if ((threadIdx.x & 31) == 0) y[row] = y[row] + acc;
}

// Up to four matrices sharing x. Global row 4b + w maps to the first matrix
// whose cumulative row count exceeds it; grid = ceil((M0+M1+M2+M3) / 4).
__launch_bounds__(128)
extern "C" __global__ void fused_iu4g128_rows(
    const char* __restrict__ A0, const char* __restrict__ A1,
    const char* __restrict__ A2, const char* __restrict__ A3,
    const float* __restrict__ x,
    float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K) {
    int row = (int)(blockIdx.x * 4 + (threadIdx.x >> 5));
    const char* A;
    float* y;
    if (row < M0) { A = A0; y = y0; }
    else if ((row -= M0) < M1) { A = A1; y = y1; }
    else if ((row -= M1) < M2) { A = A2; y = y2; }
    else if ((row -= M2) < M3) { A = A3; y = y3; }
    else return;
    const float acc = iu4_row_dot_wide(
        (const unsigned char*)A + (long long)row * IU4_ROW_BYTES(K >> 8), x, K);
    if ((threadIdx.x & 31) == 0) y[row] = acc;
}

// ── integer-activation opt-in (gfx11/gfx12) ─────────────────────────────
#if defined(__GFX11__) || defined(__GFX12__)

// ── W4A8 path (opt-in, HIPFIRE_IU4_DECODE=dot4) ─────────────────────────
// Lane t owns weights g*256 + t*8 .. +7, the
// activation scale spans 32 = 4 lanes), but the activation is signed int8
// (d = amax/127) and the dot is two V_DOT4_I32_IU8 (RDNA3 ISA §16.10 VOP3P
// opcode 22, unsigned weight bytes x signed activation bytes). Nibbles are
// split even/odd (pk & 0x0F0F0F0F, (pk >> 4) & 0x0F0F0F0F) and the
// activation bytes are packed in the same even/odd order.
__device__ __forceinline__ int iu4_dot4(int w, int a, int c) {
    return __builtin_amdgcn_sudot4(false, w, true, a, c, false);
}

__device__ __forceinline__ float iu4_row_dot_a8(
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
        const float d = amax * (1.0f / 127.0f);
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int q[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) q[j] = max(-127, min(127, (int)rintf(v[j] * id)));
        const int a_lo = (q[0] & 0xFF) | ((q[2] & 0xFF) << 8) | ((q[4] & 0xFF) << 16) | ((q[6] & 0xFF) << 24);
        const int a_hi = (q[1] & 0xFF) | ((q[3] & 0xFF) << 8) | ((q[5] & 0xFF) << 16) | ((q[7] & 0xFF) << 24);
        const unsigned pk = *(const unsigned*)(codes + (g << 7) + (t << 2));
        const int w_lo = (int)(pk & 0x0F0F0F0Fu);
        const int w_hi = (int)((pk >> 4) & 0x0F0F0F0Fu);
        const int dot = iu4_dot4(w_hi, a_hi, iu4_dot4(w_lo, a_lo, 0));
        const int sumq = iu4_dot4(0x01010101, a_hi, iu4_dot4(0x01010101, a_lo, 0));
        const unsigned h = *(const unsigned*)(scales + (g << 3) + ((t >> 4) << 2));
        const float sc = __half2float(__ushort_as_half((unsigned short)h));
        const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        acc += (d * sc) * (float)dot + zp * (d * (float)sumq);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
    return acc;
}

extern "C" __global__ void gemv_iu4g128_dot4(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = IU4_ROW_BYTES(K >> 8);
    const float acc = iu4_row_dot_a8(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = acc;
}

extern "C" __global__ void gemv_iu4g128_dot4_residual(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = IU4_ROW_BYTES(K >> 8);
    const float acc = iu4_row_dot_a8(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = y[row] + acc;
}

// ── W4A8 v2 (HIPFIRE_IU4_DECODE=dot4): x quantized once per GEMV ─────────
// `iu4_quant_x_a8` runs the exact per-lane quantization of iu4_row_dot_a8
// once (grid K/256, block 32) and stores, per (group g, lane t), int4
// {a_lo, a_hi, sumq, d_bits} at xq[g*32 + t] (16 B per 8 activations, 2K B
// for a length-K x). The `_dot4q` kernels then read that instead of
// requantizing x in every row's wave; the per-group expression is the same
// as iu4_row_dot_a8, so the result matches the v1 dot4 kernel.
extern "C" __global__ void iu4_quant_x_a8(
    const float* __restrict__ x, int4* __restrict__ xq, int K) {
    const int g = (int)blockIdx.x;
    const int t = (int)threadIdx.x;
    if (g >= (K >> 8)) return;
    const float* xp = x + ((long long)g << 8) + (t << 3);
    const float4 a = *(const float4*)xp;
    const float4 b = *(const float4*)(xp + 4);
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    const float d = amax * (1.0f / 127.0f);
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = max(-127, min(127, (int)rintf(v[j] * id)));
    const int a_lo = (q[0] & 0xFF) | ((q[2] & 0xFF) << 8) | ((q[4] & 0xFF) << 16) | ((q[6] & 0xFF) << 24);
    const int a_hi = (q[1] & 0xFF) | ((q[3] & 0xFF) << 8) | ((q[5] & 0xFF) << 16) | ((q[7] & 0xFF) << 24);
    const int sumq = iu4_dot4(0x01010101, a_hi, iu4_dot4(0x01010101, a_lo, 0));
    xq[(g << 5) + t] = make_int4(a_lo, a_hi, sumq, __float_as_int(d));
}

__device__ __forceinline__ float iu4_row_dot_a8q(
    const unsigned char* __restrict__ A,
    const int4* __restrict__ xq,
    int K) {
    const int groups = K >> 8;
    const unsigned char* codes = A;
    const unsigned char* scales = codes + (K >> 1);
    const int t = (int)threadIdx.x;
    const int hoff = (t >> 4) << 2;
    float acc = 0.0f;
#pragma unroll 4
    for (int g = 0; g < groups; ++g) {
        const int4 a = xq[(g << 5) + t];
        const unsigned pk = *(const unsigned*)(codes + (g << 7) + (t << 2));
        const unsigned h = *(const unsigned*)(scales + (g << 3) + hoff);
        const int w_lo = (int)(pk & 0x0F0F0F0Fu);
        const int w_hi = (int)((pk >> 4) & 0x0F0F0F0Fu);
        const int dot = iu4_dot4(w_hi, a.y, iu4_dot4(w_lo, a.x, 0));
        const float d = __int_as_float(a.w);
        const float sc = __half2float(__ushort_as_half((unsigned short)h));
        const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        acc += (d * sc) * (float)dot + zp * (d * (float)a.z);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
    return acc;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iu4g128_dot4q(
    const unsigned char* __restrict__ A, const int4* __restrict__ xq,
    float* __restrict__ y, int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = IU4_ROW_BYTES(K >> 8);
    const float acc = iu4_row_dot_a8q(A + (long long)row * row_bytes, xq, K);
    if (threadIdx.x == 0) y[row] = acc;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iu4g128_dot4q_residual(
    const unsigned char* __restrict__ A, const int4* __restrict__ xq,
    float* __restrict__ y, int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = IU4_ROW_BYTES(K >> 8);
    const float acc = iu4_row_dot_a8q(A + (long long)row * row_bytes, xq, K);
    if (threadIdx.x == 0) y[row] = y[row] + acc;
}

// Up to four matrices sharing x (gate/up, q/k/v, qkv/z/beta/alpha) in one
// launch; each output equals gemv_iu4g128_dot4q on that matrix.
__launch_bounds__(32, 16)
extern "C" __global__ void fused_iu4g128_dot4q_rows(
    const unsigned char* __restrict__ A0, const unsigned char* __restrict__ A1,
    const unsigned char* __restrict__ A2, const unsigned char* __restrict__ A3,
    const int4* __restrict__ xq,
    float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K) {
    int row = (int)blockIdx.x;
    const unsigned char* A;
    float* y;
    if (row < M0) { A = A0; y = y0; }
    else if ((row -= M0) < M1) { A = A1; y = y1; }
    else if ((row -= M1) < M2) { A = A2; y = y2; }
    else if ((row -= M2) < M3) { A = A3; y = y3; }
    else return;
    const long long row_bytes = IU4_ROW_BYTES(K >> 8);
    const float acc = iu4_row_dot_a8q(A + (long long)row * row_bytes, xq, K);
    if (threadIdx.x == 0) y[row] = acc;
}
#endif
