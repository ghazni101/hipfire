// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Decode GEMV for IU4G256 (qt 57). One wave per output row, wave32, block 32.
//
// Row layout: [K/2 code bytes][8 B per 256-group: f16 sc0, f16 zp0, f16 sc1,
// f16 zp1] x K/256. Low nibble = even K. Dequant w = sc*code + zp. x is
// already FWHT-256 rotated.
//
// Default kernels (`gemv_iu4g256`, `gemv_iu4g256_residual`,
// `fused_iu4g256_rows`): f32 activation, the MQ4G256V2 decode algorithm with
// IU4 address math. Each lane owns code dword `tid` of a group (weights
// g*256 + tid*8 .. +7), the half select is tid < 16, groups interleave over
// four accumulators (tail group g into acc[g % 4]), combine is
// (acc0 + acc1) + (acc2 + acc3), then a shfl_down tree. The expression forms
// are those of gemv_mq4g256v2.hip, so on a qt 57 tensor that is the
// `repack_mq4v2` relayout of a qt 44 tensor the result is bit-identical to
// the qt 44 kernel (tests/iu_quant_gemv_gpu.rs checks this).
//
// `fused_iu4g256_rows` runs up to four matrices that share x in one launch
// (gate/up, q/k/v, qkv/z/beta/alpha). Block b belongs to the first matrix
// whose cumulative row count exceeds b; the per-row arithmetic is the plain
// kernel's, so each output equals the plain kernel on that matrix.
//
// The `_dot4` kernels are W4A8 (int8 activation per 32, `v_dot4_i32_iu8`,
// HIPFIRE_IU4_DECODE=dot4).
// The `_dot8` kernels are the original W4A4 path (signed int4 activation per
// 32, `v_dot8_i32_iu4`), kept for A/B behind HIPFIRE_IU4_DECODE=dot8. CPU
// contract: `hipfire_runtime::iu_quant::gemv_iu4_dot8`. RDNA3/RDNA4 only.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define IU4_DOG(pk, sc, zp, b, a) \
    (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
         + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
         + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
         + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
         + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
         + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
         + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
         + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

#define IU4_HDR(hdr, g, sc, zp) \
    float sc, zp; \
    { \
        const unsigned int hs = *(const unsigned int*)((hdr) + (g) * 8 + hsel); \
        sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
    }

// Lane-reduced dot of one IU4 row with x. Every lane returns the row sum.
__device__ __forceinline__ float iu4_row_dot_f32(
    const char* __restrict__ row_ptr,
    const float* __restrict__ x,
    int groups_per_row) {
    const int tid = threadIdx.x;
    const int boff = tid * 4;
    const int hsel = (tid < 16) ? 0 : 4;
    const char* codes = row_ptr;
    const char* hdr = row_ptr + (long long)groups_per_row * 128;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        IU4_HDR(hdr, g + 0, sc0, zp0)
        IU4_HDR(hdr, g + 1, sc1, zp1)
        IU4_HDR(hdr, g + 2, sc2, zp2)
        IU4_HDR(hdr, g + 3, sc3, zp3)
        const char* cp = codes + g * 128 + boff;
        unsigned int pk0 = *(const unsigned int*)(cp);
        unsigned int pk1 = *(const unsigned int*)(cp + 128);
        unsigned int pk2 = *(const unsigned int*)(cp + 256);
        unsigned int pk3 = *(const unsigned int*)(cp + 384);
        int base = g * 256 + tid * 8;
        IU4_DOG(pk0, sc0, zp0, base, acc0);
        IU4_DOG(pk1, sc1, zp1, base + 256, acc1);
        IU4_DOG(pk2, sc2, zp2, base + 512, acc2);
        IU4_DOG(pk3, sc3, zp3, base + 768, acc3);
    }
    if (tail >= 1) {
        const int g = quads << 2;
        IU4_HDR(hdr, g, sc, zp)
        unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        int base = g * 256 + tid * 8;
        IU4_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        IU4_HDR(hdr, g, sc, zp)
        unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        int base = g * 256 + tid * 8;
        IU4_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        IU4_HDR(hdr, g, sc, zp)
        unsigned int pk = *(const unsigned int*)(codes + g * 128 + boff);
        int base = g * 256 + tid * 8;
        IU4_DOG(pk, sc, zp, base, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    return acc;
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iu4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int G = K >> 8;
    const float acc = iu4_row_dot_f32(A + (long long)row * G * 136, x, G);
    if (threadIdx.x == 0) y[row] = acc;
}

// `gemv_iu4g256_residual`: y += W.x, a port of gemv_mq4g256v2_residual.hip
// (two rows per block, same load order, expression forms and epilogue) with
// IU4 address math, so on a `repack_mq4v2` twin the result is bit-identical
// to the qt 44 residual kernel. The plain and residual MQ4V2 kernels are
// compiled separately and contract FMAs differently, so the residual cannot
// reuse iu4_row_dot_f32 and stay bit-exact. Grid = ceil(M / 2).
#if defined(__gfx1200__) || defined(__gfx1201__)
// RDNA4-only occupancy probe: force the compiler to target 32 resident
// one-wave blocks (64 VGPR/thread budget) on this low-occupancy kernel.
// The non-gfx12 device code remains byte-identical to the baseline.
__launch_bounds__(32, 32)
#else
__launch_bounds__(32, 16)
#endif
extern "C" __global__ void gemv_iu4g256_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Exact Qwen3.6 A3B decode contract. Making both quads static lets LLVM
    // schedule the two independent row chains across the full K extent and
    // removes the dead generic tail machinery on gfx1151.
    const int groups_per_row = 16;
#else
    const int groups_per_row = K / 256;
#endif
    const long long row_stride = (long long)groups_per_row * 136;
    const int hdr_off = groups_per_row * 128;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)


    #define LOAD_W(rp, gi, sc, zp, pk) do { \
        const char* hp_ = (rp) + hdr_off + (gi) * 8; \
        const unsigned int hA = *(const unsigned int*)(hp_); \
        const unsigned int hB = *(const unsigned int*)(hp_ + 4); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        (pk) = *(const unsigned int*)((rp) + (gi) * 128 + boff); \
    } while (0)

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Keep the fixed four-quad bound without expanding the body. Full unroll
    // doubles the instruction footprint; factor-two unroll spills private=16
    // and is not admissible for direct gfx1151 PM4 replay.
    #pragma unroll 1
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this quad (4 groups × 8) via float4 pairs.
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // Weight MEMORY phase: sequential along row0 then row1 (8 headers).
        float sc0a, zp0a, sc1a, zp1a, sc2a, zp2a, sc3a, zp3a;
        unsigned int pk0a, pk1a, pk2a, pk3a;
        float sc0b, zp0b, sc1b, zp1b, sc2b, zp2b, sc3b, zp3b;
        unsigned int pk0b, pk1b, pk2b, pk3b;

#if defined(__gfx1100__) || defined(HIPFIRE_GFX1151_RESIDUAL_ROW_SERIAL)
        LOAD_W(row_ptr0, g, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0, (g + 1), sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0, (g + 2), sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0, (g + 3), sc3a, zp3a, pk3a);

        // Finish row0 before making row1's headers live. The empty exact-gfx1100
        // dependency barrier prevents LLVM from hoisting the second row's loads
        // back across this point; it emits no ISA instruction.
        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);

        asm volatile("" : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3) :: "memory");

        LOAD_W(row_ptr1, g, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1, (g + 1), sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1, (g + 2), sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1, (g + 3), sc3b, zp3b, pk3b);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#else
        LOAD_W(row_ptr0, g, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0, (g + 1), sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0, (g + 2), sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0, (g + 3), sc3a, zp3a, pk3a);
        LOAD_W(row_ptr1, g, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1, (g + 1), sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1, (g + 2), sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1, (g + 3), sc3b, zp3b, pk3b);

        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#endif
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0, g, sc0, zp0, pk0);
        LOAD_W(row_ptr1, g, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0, g, sc0, zp0, pk0);
        LOAD_W(row_ptr1, g, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0, g, sc0, zp0, pk0);
        LOAD_W(row_ptr1, g, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W
    #undef DOG_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void fused_iu4g256_rows(
    const char* __restrict__ A0, const char* __restrict__ A1,
    const char* __restrict__ A2, const char* __restrict__ A3,
    const float* __restrict__ x,
    float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K) {
    int row = blockIdx.x;
    const char* A;
    float* y;
    if (row < M0) { A = A0; y = y0; }
    else if ((row -= M0) < M1) { A = A1; y = y1; }
    else if ((row -= M1) < M2) { A = A2; y = y2; }
    else if ((row -= M2) < M3) { A = A3; y = y3; }
    else return;
    const int G = K >> 8;
    const float acc = iu4_row_dot_f32(A + (long long)row * G * 136, x, G);
    if (threadIdx.x == 0) y[row] = acc;
}

#undef IU4_HDR
#undef IU4_DOG

// ── W4A4 reference path (opt-in) ────────────────────────────────────────
#if defined(__GFX11__) || defined(__GFX12__)

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

// ── W4A8 path (opt-in, HIPFIRE_IU4_DECODE=dot4) ─────────────────────────
// Same lane map as the W4A4 path (lane t owns weights g*256 + t*8 .. +7, the
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

extern "C" __global__ void gemv_iu4g256_dot4(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)(K >> 1) + (K >> 5);
    const float acc = iu4_row_dot_a8(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = acc;
}

extern "C" __global__ void gemv_iu4g256_dot4_residual(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long row_bytes = (long long)(K >> 1) + (K >> 5);
    const float acc = iu4_row_dot_a8(A + (long long)row * row_bytes, x, K);
    if (threadIdx.x == 0) y[row] = y[row] + acc;
}

extern "C" __global__ void gemv_iu4g256_dot8(
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

extern "C" __global__ void gemv_iu4g256_dot8_residual(
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
#endif
