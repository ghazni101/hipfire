// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Minimal LDS-tiled W8A8 GEMM for a planar int8 weight format:
//   row = [K signed int8 codes][K/32 f16 scales] (the qt59 row size), where
//   the four per-32 slots of each K128 block carry the same per-128 scale.
// Activations are the existing block_i8_128 plane (per-128 d, nibble-pair
// qs). WMMA-IU8 accumulates a whole K128 in int32, then one scale
// (sc_w * d_x) folds it into f32 - no per-32 weight dequantization.
//
// 128x128 output tile, 512 threads / 16 waves, one CTA per CU.
// LDS: X codes 16 KiB, W codes 16 KiB, X d 512 B, W sc 512 B = 33,792 B.
// Grid [ceil(M/128), ceil(N/128)]. SET or ADD epilogue.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef unsigned __attribute__((ext_vector_type(4))) u32x4_t;

#define W8_TILE 128
#define W8_NTHREADS (W8_TILE * 4)
#define W8_X_OFF 0u
#define W8_W_OFF (W8_TILE * 128u)
#define W8_DX_OFF (W8_W_OFF + W8_TILE * 128u)
#define W8_SW_OFF (W8_DX_OFF + W8_TILE * 4u)
#define W8_LDS_BYTES (W8_SW_OFF + W8_TILE * 4u)
#define W8_MAGIC 0x4B400000

// 16-byte chunk c of tile-relative entity e, swizzled for LDS bank spread.
static __device__ __forceinline__ unsigned w8_chunk(unsigned e, unsigned c) {
    return e * 128u + ((c ^ (e & 7u)) << 4);
}

template <bool ADD>
__device__ __forceinline__ void w8a8_body(
    const char* __restrict__ A,          // [M x (K + K/64)]
    const block_i8_128* __restrict__ Xq, // [K/128 x N]
    float* __restrict__ Y,               // [N x M]
    int M, int K, int N
) {
    extern __shared__ char LDS[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const unsigned l16 = (unsigned)lane & 15u;
    const unsigned hi = (unsigned)lane >> 4;

    const int tile_tok0 = (int)(blockIdx.y * W8_TILE);
    const int tile_row0 = (int)(blockIdx.x * W8_TILE);
    const int wave_tok = (wave >> 2) * 32;
    const int wave_row = (wave & 3) * 32;
    if (tile_tok0 >= N || tile_row0 >= M) return;

    const int kblocks = K / 128;
    const long long row_stride = (long long)K + (long long)(K >> 4);

    const unsigned st_e = (unsigned)(tid >> 2);
    const unsigned st_c0 = (unsigned)((tid & 3) * 2);
    const int st_tok = tile_tok0 + (int)st_e;
    const int st_row = tile_row0 + (int)st_e;
    const int safe_tok = st_tok < N ? st_tok : N - 1;
    const int safe_row = st_row < M ? st_row : M - 1;
    const unsigned st_xoff = 8u + st_c0 * 16u;
    const long long st_woff_row = (long long)safe_row * row_stride;

    float acc[2][2][8];
    #pragma unroll
    for (int nb = 0; nb < 2; nb++)
        #pragma unroll
        for (int rg = 0; rg < 2; rg++)
            #pragma unroll
            for (int l = 0; l < 8; l++) acc[nb][rg][l] = 0.0f;

    for (int kb = 0; kb < kblocks; kb++) {
        const char* xb = (const char*)(Xq + (long long)kb * N + safe_tok);
        const char* wb = A + st_woff_row + (long long)kb * 128;
        u32x4_t xv0 = *(const u32x4_t*)(xb + st_xoff);
        u32x4_t xv1 = *(const u32x4_t*)(xb + st_xoff + 16);
        u32x4_t wv0 = *(const u32x4_t*)(wb + st_c0 * 16u);
        u32x4_t wv1 = *(const u32x4_t*)(wb + (st_c0 + 1) * 16u);
        *(u32x4_t*)(LDS + W8_X_OFF + w8_chunk(st_e, st_c0)) = xv0;
        *(u32x4_t*)(LDS + W8_X_OFF + w8_chunk(st_e, st_c0 + 1)) = xv1;
        *(u32x4_t*)(LDS + W8_W_OFF + w8_chunk(st_e, st_c0)) = wv0;
        *(u32x4_t*)(LDS + W8_W_OFF + w8_chunk(st_e, st_c0 + 1)) = wv1;
        if (st_tok < N) ((float*)(LDS + W8_DX_OFF))[st_e] = ((const block_i8_128*)xb)->d;
        else ((float*)(LDS + W8_DX_OFF))[st_e] = 0.0f;
        const unsigned short sb = (st_row < M)
            ? *(const unsigned short*)(A + st_woff_row + K + (long long)kb * 8)
            : 0u;
        ((float*)(LDS + W8_SW_OFF))[st_e] = __half2float(__ushort_as_half(sb));
        __syncthreads();

        int32x8_t c[2][2];
        const int32x8_t seed = {W8_MAGIC, W8_MAGIC, W8_MAGIC, W8_MAGIC,
                                W8_MAGIC, W8_MAGIC, W8_MAGIC, W8_MAGIC};
        #pragma unroll
        for (int nb = 0; nb < 2; nb++)
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) c[nb][rg] = seed;

        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const unsigned ct = (unsigned)t;
            int32x4_t af[2], bf[2];
            #pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                const unsigned e = (unsigned)(wave_tok + nb * 16 + (int)l16);
                af[nb] = *(const int32x4_t*)(LDS + W8_X_OFF + w8_chunk(e, ct));
            }
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const unsigned e = (unsigned)(wave_row + rg * 16 + (int)l16);
                bf[rg] = *(const int32x4_t*)(LDS + W8_W_OFF + w8_chunk(e, ct));
            }
            #pragma unroll
            for (int nb = 0; nb < 2; nb++)
                #pragma unroll
                for (int rg = 0; rg < 2; rg++)
                    c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, af[nb], true, bf[rg], c[nb][rg], false);
        }

        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                #pragma unroll
                for (int l = 0; l < 8; l++) {
                    const float dx = ((const float*)(LDS + W8_DX_OFF))
                        [wave_tok + nb * 16 + 2 * l + (int)hi];
                    const float sw = ((const float*)(LDS + W8_SW_OFF))
                        [wave_row + rg * 16 + (int)l16];
                    const float fc = __int_as_float(c[nb][rg][l]) - 12582912.0f;
                    acc[nb][rg][l] = __builtin_fmaf(__fmul_rn(sw, dx), fc, acc[nb][rg][l]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                const int tok = tile_tok0 + wave_tok + nb * 16 + 2 * l + (int)hi;
                const int row = tile_row0 + wave_row + rg * 16 + (int)l16;
                if (tok >= N || row >= M) continue;
                float* yp = Y + (long long)tok * M + row;
                if constexpr (ADD) *yp += acc[nb][rg][l];
                else *yp = acc[nb][rg][l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(W8_NTHREADS)
void gemm_iu8g32_w8a8_min_set(
    const char* __restrict__ A, const block_i8_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int
) { w8a8_body<false>(A, Xq, Y, M, K, N); }

extern "C" __global__ __launch_bounds__(W8_NTHREADS)
void gemm_iu8g32_w8a8_min_add(
    const char* __restrict__ A, const block_i8_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int
) { w8a8_body<true>(A, Xq, Y, M, K, N); }
