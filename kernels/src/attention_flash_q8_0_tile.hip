// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention tile kernel — sequential dot product + 2-pass softmax.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (one WAVE32).
// LDS: tile_size floats for scores + head_dim floats for Q.
//   head_dim=128 → (128+128)*4 = 1024 B.
//   head_dim=256 → (128+256)*4 = 1536 B.
//
// Wave-cooperative dot product: each thread owns 4 of head_dim K elements
// (per half), wave-reduces via __shfl_xor. For head_dim > 128 the kernel
// iterates over `n_halves = (head_dim + 127) / 128` halves (one of 128
// elements each) inside the per-token loop and accumulates the partial
// before the wave reduction — preserves the exact same mathematical sum
// the baseline kernel produces and avoids any softmax-amplified score
// error from re-ordering FMAs.
//
// V accumulation also loops over halves so all `head_dim` output dims are
// written to partials (the previous version only wrote the first 128 dims,
// leaving the upper half of the partials row uninitialized — fine when the
// buffer was freshly malloc'd-zero but catastrophic when stale prefill data
// later collided with the reduce kernel's full-head_dim read).
//
// Works on RDNA1 (gfx1010), RDNA2 (gfx1030), RDNA3 (gfx1100), RDNA4 (gfx1200+).
#include <hip/hip_runtime.h>

#if defined(HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE)
// Preserve the portable kernel's descending XOR tree (16,8,4,2,1) while
// avoiding generic ds_bpermute for the three intra-row exchanges. The first
// two stages use exact XOR ds_swizzles; DPP8/quad-perm handle the remaining
// stages entirely in VALU. This keeps the floating-point association intact.
__device__ __forceinline__ int hipfire_attention_xor16(int bits) {
    return __builtin_amdgcn_ds_swizzle(bits, 0x401f);
}

__device__ __forceinline__ int hipfire_attention_xor8(int bits) {
    return __builtin_amdgcn_ds_swizzle(bits, 0x201f);
}

__device__ __forceinline__ int hipfire_attention_xor4_dpp8(int bits) {
    // dpp8:[4,5,6,7,0,1,2,3]
    return __builtin_amdgcn_mov_dpp8(bits, 0x688fac);
}

__device__ __forceinline__ int hipfire_attention_xor2_dpp(int bits) {
    // quad_perm:[2,3,0,1]
    return __builtin_amdgcn_mov_dpp(bits, 0x4e, 0xf, 0xf, false);
}

__device__ __forceinline__ int hipfire_attention_xor1_dpp(int bits) {
    // quad_perm:[1,0,3,2]
    return __builtin_amdgcn_mov_dpp(bits, 0xb1, 0xf, 0xf, false);
}

__device__ __forceinline__ float hipfire_attention_reduce_sum_xor(float value) {
    value += __builtin_bit_cast(float,
        hipfire_attention_xor16(__builtin_bit_cast(int, value)));
    value += __builtin_bit_cast(float,
        hipfire_attention_xor8(__builtin_bit_cast(int, value)));
    value += __builtin_bit_cast(float,
        hipfire_attention_xor4_dpp8(__builtin_bit_cast(int, value)));
    value += __builtin_bit_cast(float,
        hipfire_attention_xor2_dpp(__builtin_bit_cast(int, value)));
    value += __builtin_bit_cast(float,
        hipfire_attention_xor1_dpp(__builtin_bit_cast(int, value)));
    return value;
}

__device__ __forceinline__ float hipfire_attention_reduce_max_xor(float value) {
    value = fmaxf(value, __builtin_bit_cast(float,
        hipfire_attention_xor16(__builtin_bit_cast(int, value))));
    value = fmaxf(value, __builtin_bit_cast(float,
        hipfire_attention_xor8(__builtin_bit_cast(int, value))));
    value = fmaxf(value, __builtin_bit_cast(float,
        hipfire_attention_xor4_dpp8(__builtin_bit_cast(int, value))));
    value = fmaxf(value, __builtin_bit_cast(float,
        hipfire_attention_xor2_dpp(__builtin_bit_cast(int, value))));
    value = fmaxf(value, __builtin_bit_cast(float,
        hipfire_attention_xor1_dpp(__builtin_bit_cast(int, value))));
    return value;
}
#endif

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile(
    const float* __restrict__ q,         // [n_heads, head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,        // [n_heads, max_tiles, 2 + head_dim]
    const int* __restrict__ pos_buf,     // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int window,                          // sliding-window span; <= 0 = full causal
    int effective_seq_len                // retained ABI; dynamic length comes from pos_buf
) {
    (void)effective_seq_len;
    const int logical_seq_len = pos_buf[0] + 1;
    const int seq_len = (window > 0) ? min(logical_seq_len, max_seq) : logical_seq_len;
    // Sliding-window lower bound: keys at t < win_lo are outside the window
    // [seq_len-window, seq_len) and masked. window <= 0 = full causal.
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int row_stride = total_blocks_per_pos * 34;

    // Number of 128-element halves spanning head_dim. One pass per half
    // in Phase A (Q·K) and Phase D (V) so a 32-thread wave covers the
    // full head_dim regardless of size.
    const int n_halves = (head_dim + 127) / 128;

    // LDS layout: [tile_size scores][head_dim Q values]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* q_lds  = sdata + tile_size;

    // Load Q into LDS. 32 threads × 4 values × n_halves = head_dim values.
    const float* q_head = q + h * head_dim;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        q_lds[d_base + 0] = q_head[d_base + 0];
        q_lds[d_base + 1] = q_head[d_base + 1];
        q_lds[d_base + 2] = q_head[d_base + 2];
        q_lds[d_base + 3] = q_head[d_base + 3];
    }
    __syncthreads();

    // ═══ Phase A: compute tile_len dot products (wave-cooperative) ═══
    // Each thread handles 4 of 128 elements per half, accumulates across
    // halves, then wave-reduces via __shfl_xor. Thread assignment within a
    // half: bi_k = tid/8, offset = (tid%8)*4. Block index across halves:
    // half*4 + bi_k.
    const int k_off = (tid % 8) * 4;

    int valid_start = win_lo - tile_start;
    if (valid_start < 0) valid_start = 0;
    if (valid_start > tile_len) valid_start = tile_len;
    for (int t_local = 0; t_local < valid_start; t_local++) {
        if (tid == 0) scores[t_local] = -INFINITY;
    }

    int t_local = valid_start;
    for (; t_local + 3 < tile_len; t_local += 4) {
        const int t0 = tile_start + t_local + 0;
        const int t1 = tile_start + t_local + 1;
        const int t2 = tile_start + t_local + 2;
        const int t3 = tile_start + t_local + 3;
        const int s0 = (window > 0) ? (t0 % max_seq) : t0;
        const int s1 = (window > 0) ? (t1 % max_seq) : t1;
        const int s2 = (window > 0) ? (t2 % max_seq) : t2;
        const int s3 = (window > 0) ? (t3 % max_seq) : t3;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int k_bi = half * 4 + tid / 8;
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;
            const unsigned char* kb0 = k_cache + (size_t)s0 * row_stride + k_blk_off;
            const unsigned char* kb1 = k_cache + (size_t)s1 * row_stride + k_blk_off;
            const unsigned char* kb2 = k_cache + (size_t)s2 * row_stride + k_blk_off;
            const unsigned char* kb3 = k_cache + (size_t)s3 * row_stride + k_blk_off;
            float ks0 = (float)*((const _Float16*)kb0);
            float ks1 = (float)*((const _Float16*)kb1);
            float ks2 = (float)*((const _Float16*)kb2);
            float ks3 = (float)*((const _Float16*)kb3);
            const signed char k00 = (signed char)kb0[2 + k_off + 0];
            const signed char k01 = (signed char)kb0[2 + k_off + 1];
            const signed char k02 = (signed char)kb0[2 + k_off + 2];
            const signed char k03 = (signed char)kb0[2 + k_off + 3];
            const signed char k10 = (signed char)kb1[2 + k_off + 0];
            const signed char k11 = (signed char)kb1[2 + k_off + 1];
            const signed char k12 = (signed char)kb1[2 + k_off + 2];
            const signed char k13 = (signed char)kb1[2 + k_off + 3];
            const signed char k20 = (signed char)kb2[2 + k_off + 0];
            const signed char k21 = (signed char)kb2[2 + k_off + 1];
            const signed char k22 = (signed char)kb2[2 + k_off + 2];
            const signed char k23 = (signed char)kb2[2 + k_off + 3];
            const signed char k30 = (signed char)kb3[2 + k_off + 0];
            const signed char k31 = (signed char)kb3[2 + k_off + 1];
            const signed char k32 = (signed char)kb3[2 + k_off + 2];
            const signed char k33 = (signed char)kb3[2 + k_off + 3];
            const float* qb_thread = q_lds + k_bi * 32 + k_off;
            partial0 += qb_thread[0] * (ks0 * (float)k00)
                      + qb_thread[1] * (ks0 * (float)k01)
                      + qb_thread[2] * (ks0 * (float)k02)
                      + qb_thread[3] * (ks0 * (float)k03);
            partial1 += qb_thread[0] * (ks1 * (float)k10)
                      + qb_thread[1] * (ks1 * (float)k11)
                      + qb_thread[2] * (ks1 * (float)k12)
                      + qb_thread[3] * (ks1 * (float)k13);
            partial2 += qb_thread[0] * (ks2 * (float)k20)
                      + qb_thread[1] * (ks2 * (float)k21)
                      + qb_thread[2] * (ks2 * (float)k22)
                      + qb_thread[3] * (ks2 * (float)k23);
            partial3 += qb_thread[0] * (ks3 * (float)k30)
                      + qb_thread[1] * (ks3 * (float)k31)
                      + qb_thread[2] * (ks3 * (float)k32)
                      + qb_thread[3] * (ks3 * (float)k33);
        }
#if defined(HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE)
        partial0 = hipfire_attention_reduce_sum_xor(partial0);
        partial1 = hipfire_attention_reduce_sum_xor(partial1);
        partial2 = hipfire_attention_reduce_sum_xor(partial2);
        partial3 = hipfire_attention_reduce_sum_xor(partial3);
#else
        for (int off = 16; off > 0; off >>= 1) {
            partial0 += __shfl_xor(partial0, off);
            partial1 += __shfl_xor(partial1, off);
            partial2 += __shfl_xor(partial2, off);
            partial3 += __shfl_xor(partial3, off);
        }
#endif
        if (tid == 0) {
            scores[t_local + 0] = partial0 * scale_attn;
            scores[t_local + 1] = partial1 * scale_attn;
            scores[t_local + 2] = partial2 * scale_attn;
            scores[t_local + 3] = partial3 * scale_attn;
        }
    }
    for (; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const int slot = (window > 0) ? (t % max_seq) : t;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int k_bi = half * 4 + tid / 8;
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;
            const unsigned char* kb = k_cache + (size_t)slot * row_stride + k_blk_off;
            float ks = (float)*((const _Float16*)kb);
            const float* qb_thread = q_lds + k_bi * 32 + k_off;
            partial += qb_thread[0] * (ks * (float)((signed char)kb[2 + k_off + 0]))
                     + qb_thread[1] * (ks * (float)((signed char)kb[2 + k_off + 1]))
                     + qb_thread[2] * (ks * (float)((signed char)kb[2 + k_off + 2]))
                     + qb_thread[3] * (ks * (float)((signed char)kb[2 + k_off + 3]));
        }
#if defined(HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE)
        partial = hipfire_attention_reduce_sum_xor(partial);
#else
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
#endif
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: find tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
#if defined(HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE)
    local_max = hipfire_attention_reduce_max_xor(local_max);
#else
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
#endif
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
#if defined(HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE)
    local_sum = hipfire_attention_reduce_sum_xor(local_sum);
#else
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
#endif
    float tile_sum = local_sum;
    __syncthreads();

    // ═══ Write tile partials ═══
    //
    // Stride by max_tiles (derived from max_seq) so the layout matches the
    // attention_flash_q8_0_reduce kernel, which uses max_tiles for its
    // per-head row stride. Using n_tiles (dynamic, ceil(seq_len/tile_size))
    // here would put head h's data at offset h*n_tiles*stride, while the
    // reduce reads from h*max_tiles*stride — mismatch for h>0 whenever
    // seq_len < max_seq. This is the asym2/3/4 tile-kernel convention.
    // Symptom of the previous mismatch: at long context all heads except
    // h=0 produced NaN/garbage outputs (reduce reading uninitialized or
    // wrong-head stale partials), poisoning every subsequent layer and
    // producing 100% NaN logits at the first single-token decode step
    // after a long prefill (see fix/long-decode-nan).
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }

    // ═══ Phase D: V-weighted accumulation (loops over halves so all
    //              head_dim output dims are written, not just first 128) ═══
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        const int bi = d_base / 32;
        const int bj_base = d_base % 32;
        const int v_blk_off = (kv_head_block_start + bi) * 34;
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        int t_local = 0;
        #define ACCUM_V4(tl) do { \
            float w0 = scores[(tl) + 0]; \
            float w1 = scores[(tl) + 1]; \
            float w2 = scores[(tl) + 2]; \
            float w3 = scores[(tl) + 3]; \
            int t0 = tile_start + (tl) + 0; \
            int t1 = tile_start + (tl) + 1; \
            int t2 = tile_start + (tl) + 2; \
            int t3 = tile_start + (tl) + 3; \
            int s0 = (window > 0) ? (t0 % max_seq) : t0; \
            int s1 = (window > 0) ? (t1 % max_seq) : t1; \
            int s2 = (window > 0) ? (t2 % max_seq) : t2; \
            int s3 = (window > 0) ? (t3 % max_seq) : t3; \
            const unsigned char* vb0 = v_cache + (size_t)s0 * row_stride + v_blk_off; \
            const unsigned char* vb1 = v_cache + (size_t)s1 * row_stride + v_blk_off; \
            const unsigned char* vb2 = v_cache + (size_t)s2 * row_stride + v_blk_off; \
            const unsigned char* vb3 = v_cache + (size_t)s3 * row_stride + v_blk_off; \
            float vs0 = (float)*((const _Float16*)vb0); \
            float vs1 = (float)*((const _Float16*)vb1); \
            float vs2 = (float)*((const _Float16*)vb2); \
            float vs3 = (float)*((const _Float16*)vb3); \
            const signed char v00 = (signed char)vb0[2 + bj_base + 0]; \
            const signed char v01 = (signed char)vb0[2 + bj_base + 1]; \
            const signed char v02 = (signed char)vb0[2 + bj_base + 2]; \
            const signed char v03 = (signed char)vb0[2 + bj_base + 3]; \
            const signed char v10 = (signed char)vb1[2 + bj_base + 0]; \
            const signed char v11 = (signed char)vb1[2 + bj_base + 1]; \
            const signed char v12 = (signed char)vb1[2 + bj_base + 2]; \
            const signed char v13 = (signed char)vb1[2 + bj_base + 3]; \
            const signed char v20 = (signed char)vb2[2 + bj_base + 0]; \
            const signed char v21 = (signed char)vb2[2 + bj_base + 1]; \
            const signed char v22 = (signed char)vb2[2 + bj_base + 2]; \
            const signed char v23 = (signed char)vb2[2 + bj_base + 3]; \
            const signed char v30 = (signed char)vb3[2 + bj_base + 0]; \
            const signed char v31 = (signed char)vb3[2 + bj_base + 1]; \
            const signed char v32 = (signed char)vb3[2 + bj_base + 2]; \
            const signed char v33 = (signed char)vb3[2 + bj_base + 3]; \
            out0 += w0 * (vs0 * (float)v00); \
            out1 += w0 * (vs0 * (float)v01); \
            out2 += w0 * (vs0 * (float)v02); \
            out3 += w0 * (vs0 * (float)v03); \
            out0 += w1 * (vs1 * (float)v10); \
            out1 += w1 * (vs1 * (float)v11); \
            out2 += w1 * (vs1 * (float)v12); \
            out3 += w1 * (vs1 * (float)v13); \
            out0 += w2 * (vs2 * (float)v20); \
            out1 += w2 * (vs2 * (float)v21); \
            out2 += w2 * (vs2 * (float)v22); \
            out3 += w2 * (vs2 * (float)v23); \
            out0 += w3 * (vs3 * (float)v30); \
            out1 += w3 * (vs3 * (float)v31); \
            out2 += w3 * (vs3 * (float)v32); \
            out3 += w3 * (vs3 * (float)v33); \
        } while (0)
        #define ACCUM_V(tl) do { \
            float w = scores[(tl)]; \
            int t = tile_start + (tl); \
            int slot = (window > 0) ? (t % max_seq) : t; \
            const unsigned char* vb = v_cache + (size_t)slot * row_stride + v_blk_off; \
            float vs = (float)*((const _Float16*)vb); \
            out0 += w * (vs * (float)((signed char)vb[2 + bj_base + 0])); \
            out1 += w * (vs * (float)((signed char)vb[2 + bj_base + 1])); \
            out2 += w * (vs * (float)((signed char)vb[2 + bj_base + 2])); \
            out3 += w * (vs * (float)((signed char)vb[2 + bj_base + 3])); \
        } while (0)
        for (; t_local + 7 < tile_len; t_local += 8) {
            ACCUM_V4(t_local + 0);
            asm volatile("" : "+v"(out0), "+v"(out1), "+v"(out2), "+v"(out3) :: "memory");
            ACCUM_V4(t_local + 4);
        }
        for (; t_local + 3 < tile_len; t_local += 4) {
            ACCUM_V4(t_local);
        }
        for (; t_local < tile_len; t_local++) {
            ACCUM_V(t_local);
        }
        #undef ACCUM_V4
        #undef ACCUM_V
        p[2 + d_base + 0] = out0;
        p[2 + d_base + 1] = out1;
        p[2 + d_base + 2] = out2;
        p[2 + d_base + 3] = out3;
    }
}

// ═══ GQA-fused tile kernel ═══
//
// One block per (kv_head, tile) computes all G sibling query heads instead
// of one. The per-head kernel above launches [n_heads, n_tiles] blocks and
// every one of the n_heads/n_kv_heads sibling blocks re-reads the SAME K/V
// bytes of its kv head — at K2-Horizon's 32:8 ratio, 4x the KV traffic, at
// ~10 GB/s effective on gfx1100, which was the entire long-context decode
// decay (0.6 ms/layer at 6k ctx, growing linearly). Here K and V bytes are
// loaded once per key and shared across the G sibling heads: G times less
// KV traffic and G times fewer blocks, with the tile axis (grid y) still
// position-bound for replay.
//
// Per-head arithmetic — the 4-product sum expression, half iteration
// order, and the descending XOR wave reductions — is copied verbatim from
// attention_flash_q8_0_tile, so scores and partials are bit-exact per head
// and the shared `attention_flash_q8_0_reduce` consumes them unchanged.
// Instantiated for the common GQA ratios G ∈ {2, 4, 8} (compile-time G
// keeps the per-head accumulator arrays in registers); other ratios and
// MHA keep the per-head kernel. Opt out with HIPFIRE_Q8_FLASH_GQA=0.

#define HFIRE_Q8_GQA_TILE(G, MIN_WAVES)                                                        \
extern "C" __launch_bounds__(32, MIN_WAVES)                                                    \
__global__ void attention_flash_q8_0_tile_gqa##G(                                              \
    const float* __restrict__ q,                                                               \
    const unsigned char* __restrict__ k_cache,                                                 \
    const unsigned char* __restrict__ v_cache,                                                 \
    float* __restrict__ partials,                                                              \
    const int* __restrict__ pos_buf,                                                           \
    int n_heads, int n_kv_heads, int head_dim, int max_seq,                                    \
    float scale_attn, int tile_size, int window, int effective_seq_len) {                      \
    (void)effective_seq_len;                                                                   \
    const int logical_seq_len = pos_buf[0] + 1;                                                \
    const int seq_len = (window > 0) ? min(logical_seq_len, max_seq) : logical_seq_len;        \
    const int win_lo = (window > 0) ? (seq_len - window) : 0;                                  \
    const int kv_h = blockIdx.x;                                                               \
    if (kv_h >= n_kv_heads) return;                                                            \
    const int tile_id = blockIdx.y;                                                            \
    const int tile_start = tile_id * tile_size;                                                \
    const int tile_end = min(tile_start + tile_size, seq_len);                                 \
    if (tile_start >= seq_len) return;                                                         \
    const int tile_len = tile_end - tile_start;                                                \
    const int blocks_per_head = head_dim / 32;                                                 \
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;                             \
    const int kv_head_block_start = kv_h * blocks_per_head;                                    \
    const int row_stride = total_blocks_per_pos * 34;                                          \
    const int n_halves = (head_dim + 127) / 128;                                               \
    extern __shared__ float sdata[];                                                           \
    float* q_lds = sdata;                  /* [G][head_dim] */                                 \
    float* scores = sdata + (G) * head_dim; /* [G][tile_size] */                               \
    const int tid = threadIdx.x;                                                               \
    _Pragma("unroll")                                                                          \
    for (int g = 0; g < G; g++) {                                                              \
        const float* q_head = q + (kv_h * G + g) * head_dim;                                   \
        for (int half = 0; half < n_halves; half++) {                                          \
            const int d_base = half * 128 + tid * 4;                                           \
            q_lds[g * head_dim + d_base + 0] = q_head[d_base + 0];                             \
            q_lds[g * head_dim + d_base + 1] = q_head[d_base + 1];                             \
            q_lds[g * head_dim + d_base + 2] = q_head[d_base + 2];                             \
            q_lds[g * head_dim + d_base + 3] = q_head[d_base + 3];                             \
        }                                                                                      \
    }                                                                                          \
    __syncthreads();                                                                           \
    const int k_off = (tid % 8) * 4;                                                           \
    int valid_start = win_lo - tile_start;                                                     \
    if (valid_start < 0) valid_start = 0;                                                      \
    if (valid_start > tile_len) valid_start = tile_len;                                        \
    for (int t_local = 0; t_local < valid_start; t_local++) {                                  \
        if (tid == 0) {                                                                        \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) scores[g * tile_size + t_local] = -INFINITY;           \
        }                                                                                      \
    }                                                                                          \
    int t_local = valid_start;                                                                 \
    for (; t_local + 3 < tile_len; t_local += 4) {                                             \
        const int t0 = tile_start + t_local + 0;                                               \
        const int t1 = tile_start + t_local + 1;                                               \
        const int t2 = tile_start + t_local + 2;                                               \
        const int t3 = tile_start + t_local + 3;                                               \
        const int s0 = (window > 0) ? (t0 % max_seq) : t0;                                     \
        const int s1 = (window > 0) ? (t1 % max_seq) : t1;                                     \
        const int s2 = (window > 0) ? (t2 % max_seq) : t2;                                     \
        const int s3 = (window > 0) ? (t3 % max_seq) : t3;                                     \
        float partial[G][4];                                                                   \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) {                                                          \
            partial[g][0] = 0.0f; partial[g][1] = 0.0f;                                        \
            partial[g][2] = 0.0f; partial[g][3] = 0.0f;                                        \
        }                                                                                      \
        for (int half = 0; half < n_halves; half++) {                                          \
            const int k_bi = half * 4 + tid / 8;                                               \
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;                           \
            const unsigned char* kb0 = k_cache + (size_t)s0 * row_stride + k_blk_off;          \
            const unsigned char* kb1 = k_cache + (size_t)s1 * row_stride + k_blk_off;          \
            const unsigned char* kb2 = k_cache + (size_t)s2 * row_stride + k_blk_off;          \
            const unsigned char* kb3 = k_cache + (size_t)s3 * row_stride + k_blk_off;          \
            float ks0 = (float)*((const _Float16*)kb0);                                        \
            float ks1 = (float)*((const _Float16*)kb1);                                        \
            float ks2 = (float)*((const _Float16*)kb2);                                        \
            float ks3 = (float)*((const _Float16*)kb3);                                        \
            const signed char k00 = (signed char)kb0[2 + k_off + 0];                           \
            const signed char k01 = (signed char)kb0[2 + k_off + 1];                           \
            const signed char k02 = (signed char)kb0[2 + k_off + 2];                           \
            const signed char k03 = (signed char)kb0[2 + k_off + 3];                           \
            const signed char k10 = (signed char)kb1[2 + k_off + 0];                           \
            const signed char k11 = (signed char)kb1[2 + k_off + 1];                           \
            const signed char k12 = (signed char)kb1[2 + k_off + 2];                           \
            const signed char k13 = (signed char)kb1[2 + k_off + 3];                           \
            const signed char k20 = (signed char)kb2[2 + k_off + 0];                           \
            const signed char k21 = (signed char)kb2[2 + k_off + 1];                           \
            const signed char k22 = (signed char)kb2[2 + k_off + 2];                           \
            const signed char k23 = (signed char)kb2[2 + k_off + 3];                           \
            const signed char k30 = (signed char)kb3[2 + k_off + 0];                           \
            const signed char k31 = (signed char)kb3[2 + k_off + 1];                           \
            const signed char k32 = (signed char)kb3[2 + k_off + 2];                           \
            const signed char k33 = (signed char)kb3[2 + k_off + 3];                           \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) {                                                      \
                const float* qb_thread = q_lds + g * head_dim + k_bi * 32 + k_off;             \
                partial[g][0] += qb_thread[0] * (ks0 * (float)k00)                             \
                               + qb_thread[1] * (ks0 * (float)k01)                             \
                               + qb_thread[2] * (ks0 * (float)k02)                             \
                               + qb_thread[3] * (ks0 * (float)k03);                            \
                partial[g][1] += qb_thread[0] * (ks1 * (float)k10)                             \
                               + qb_thread[1] * (ks1 * (float)k11)                             \
                               + qb_thread[2] * (ks1 * (float)k12)                             \
                               + qb_thread[3] * (ks1 * (float)k13);                            \
                partial[g][2] += qb_thread[0] * (ks2 * (float)k20)                             \
                               + qb_thread[1] * (ks2 * (float)k21)                             \
                               + qb_thread[2] * (ks2 * (float)k22)                             \
                               + qb_thread[3] * (ks2 * (float)k23);                            \
                partial[g][3] += qb_thread[0] * (ks3 * (float)k30)                             \
                               + qb_thread[1] * (ks3 * (float)k31)                             \
                               + qb_thread[2] * (ks3 * (float)k32)                             \
                               + qb_thread[3] * (ks3 * (float)k33);                            \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) {                                                          \
            _Pragma("unroll")                                                                  \
            for (int i = 0; i < 4; i++) {                                                      \
                for (int off = 16; off > 0; off >>= 1)                                         \
                    partial[g][i] += __shfl_xor(partial[g][i], off);                           \
            }                                                                                  \
        }                                                                                      \
        if (tid == 0) {                                                                        \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) {                                                      \
                scores[g * tile_size + t_local + 0] = partial[g][0] * scale_attn;              \
                scores[g * tile_size + t_local + 1] = partial[g][1] * scale_attn;              \
                scores[g * tile_size + t_local + 2] = partial[g][2] * scale_attn;              \
                scores[g * tile_size + t_local + 3] = partial[g][3] * scale_attn;              \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
    for (; t_local < tile_len; t_local++) {                                                    \
        int t = tile_start + t_local;                                                          \
        const int slot = (window > 0) ? (t % max_seq) : t;                                     \
        float partial[G];                                                                      \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) partial[g] = 0.0f;                                         \
        for (int half = 0; half < n_halves; half++) {                                          \
            const int k_bi = half * 4 + tid / 8;                                               \
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;                           \
            const unsigned char* kb = k_cache + (size_t)slot * row_stride + k_blk_off;         \
            float ks = (float)*((const _Float16*)kb);                                          \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) {                                                      \
                const float* qb_thread = q_lds + g * head_dim + k_bi * 32 + k_off;             \
                partial[g] += qb_thread[0] * (ks * (float)((signed char)kb[2 + k_off + 0]))    \
                            + qb_thread[1] * (ks * (float)((signed char)kb[2 + k_off + 1]))    \
                            + qb_thread[2] * (ks * (float)((signed char)kb[2 + k_off + 2]))    \
                            + qb_thread[3] * (ks * (float)((signed char)kb[2 + k_off + 3]));   \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) {                                                          \
            for (int off = 16; off > 0; off >>= 1)                                             \
                partial[g] += __shfl_xor(partial[g], off);                                     \
        }                                                                                      \
        if (tid == 0) {                                                                        \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) scores[g * tile_size + t_local] = partial[g] * scale_attn; \
        }                                                                                      \
    }                                                                                          \
    __syncthreads();                                                                           \
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;                               \
    _Pragma("unroll")                                                                          \
    for (int g = 0; g < G; g++) {                                                              \
        float tile_max = -1e30f;                                                               \
        for (int i = tid; i < tile_len; i += 32)                                               \
            tile_max = fmaxf(tile_max, scores[g * tile_size + i]);                             \
        for (int off = 16; off > 0; off >>= 1)                                                 \
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));                             \
        float tile_sum = 0.0f;                                                                 \
        for (int i = tid; i < tile_len; i += 32) {                                             \
            float e = expf(scores[g * tile_size + i] - tile_max);                              \
            scores[g * tile_size + i] = e;                                                     \
            tile_sum += e;                                                                     \
        }                                                                                      \
        for (int off = 16; off > 0; off >>= 1)                                                 \
            tile_sum += __shfl_xor(tile_sum, off);                                             \
        if (tid == 0) {                                                                        \
            float* pg = partials + ((kv_h * G + g) * max_tiles + tile_id) * (2 + head_dim);    \
            pg[0] = tile_max;                                                                  \
            pg[1] = tile_sum;                                                                  \
        }                                                                                      \
    }                                                                                          \
    __syncthreads();                                                                           \
    for (int half = 0; half < n_halves; half++) {                                              \
        const int d_base = half * 128 + tid * 4;                                               \
        const int bi = d_base / 32;                                                            \
        const int bj_base = d_base % 32;                                                       \
        const int v_blk_off = (kv_head_block_start + bi) * 34;                                 \
        float out[G][4];                                                                       \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) {                                                          \
            out[g][0] = 0.0f; out[g][1] = 0.0f; out[g][2] = 0.0f; out[g][3] = 0.0f;            \
        }                                                                                      \
        int vt = 0;                                                                            \
        for (; vt + 3 < tile_len; vt += 4) {                                                   \
            int t0 = tile_start + vt + 0;                                                      \
            int t1 = tile_start + vt + 1;                                                      \
            int t2 = tile_start + vt + 2;                                                      \
            int t3 = tile_start + vt + 3;                                                      \
            int s0 = (window > 0) ? (t0 % max_seq) : t0;                                       \
            int s1 = (window > 0) ? (t1 % max_seq) : t1;                                       \
            int s2 = (window > 0) ? (t2 % max_seq) : t2;                                       \
            int s3 = (window > 0) ? (t3 % max_seq) : t3;                                       \
            const unsigned char* vb0 = v_cache + (size_t)s0 * row_stride + v_blk_off;          \
            const unsigned char* vb1 = v_cache + (size_t)s1 * row_stride + v_blk_off;          \
            const unsigned char* vb2 = v_cache + (size_t)s2 * row_stride + v_blk_off;          \
            const unsigned char* vb3 = v_cache + (size_t)s3 * row_stride + v_blk_off;          \
            float vs0 = (float)*((const _Float16*)vb0);                                        \
            float vs1 = (float)*((const _Float16*)vb1);                                        \
            float vs2 = (float)*((const _Float16*)vb2);                                        \
            float vs3 = (float)*((const _Float16*)vb3);                                        \
            const signed char v00 = (signed char)vb0[2 + bj_base + 0];                         \
            const signed char v01 = (signed char)vb0[2 + bj_base + 1];                         \
            const signed char v02 = (signed char)vb0[2 + bj_base + 2];                         \
            const signed char v03 = (signed char)vb0[2 + bj_base + 3];                         \
            const signed char v10 = (signed char)vb1[2 + bj_base + 0];                         \
            const signed char v11 = (signed char)vb1[2 + bj_base + 1];                         \
            const signed char v12 = (signed char)vb1[2 + bj_base + 2];                         \
            const signed char v13 = (signed char)vb1[2 + bj_base + 3];                         \
            const signed char v20 = (signed char)vb2[2 + bj_base + 0];                         \
            const signed char v21 = (signed char)vb2[2 + bj_base + 1];                         \
            const signed char v22 = (signed char)vb2[2 + bj_base + 2];                         \
            const signed char v23 = (signed char)vb2[2 + bj_base + 3];                         \
            const signed char v30 = (signed char)vb3[2 + bj_base + 0];                         \
            const signed char v31 = (signed char)vb3[2 + bj_base + 1];                         \
            const signed char v32 = (signed char)vb3[2 + bj_base + 2];                         \
            const signed char v33 = (signed char)vb3[2 + bj_base + 3];                         \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) {                                                      \
                float w0 = scores[g * tile_size + vt + 0];                                     \
                float w1 = scores[g * tile_size + vt + 1];                                     \
                float w2 = scores[g * tile_size + vt + 2];                                     \
                float w3 = scores[g * tile_size + vt + 3];                                     \
                out[g][0] += w0 * (vs0 * (float)v00);                                          \
                out[g][1] += w0 * (vs0 * (float)v01);                                          \
                out[g][2] += w0 * (vs0 * (float)v02);                                          \
                out[g][3] += w0 * (vs0 * (float)v03);                                          \
                out[g][0] += w1 * (vs1 * (float)v10);                                          \
                out[g][1] += w1 * (vs1 * (float)v11);                                          \
                out[g][2] += w1 * (vs1 * (float)v12);                                          \
                out[g][3] += w1 * (vs1 * (float)v13);                                          \
                out[g][0] += w2 * (vs2 * (float)v20);                                          \
                out[g][1] += w2 * (vs2 * (float)v21);                                          \
                out[g][2] += w2 * (vs2 * (float)v22);                                          \
                out[g][3] += w2 * (vs2 * (float)v23);                                          \
                out[g][0] += w3 * (vs3 * (float)v30);                                          \
                out[g][1] += w3 * (vs3 * (float)v31);                                          \
                out[g][2] += w3 * (vs3 * (float)v32);                                          \
                out[g][3] += w3 * (vs3 * (float)v33);                                          \
            }                                                                                  \
        }                                                                                      \
        for (; vt < tile_len; vt++) {                                                          \
            int t = tile_start + vt;                                                           \
            int slot = (window > 0) ? (t % max_seq) : t;                                       \
            const unsigned char* vb = v_cache + (size_t)slot * row_stride + v_blk_off;         \
            float vs = (float)*((const _Float16*)vb);                                          \
            _Pragma("unroll")                                                                  \
            for (int g = 0; g < G; g++) {                                                      \
                float w = scores[g * tile_size + vt];                                          \
                out[g][0] += w * (vs * (float)((signed char)vb[2 + bj_base + 0]));             \
                out[g][1] += w * (vs * (float)((signed char)vb[2 + bj_base + 1]));             \
                out[g][2] += w * (vs * (float)((signed char)vb[2 + bj_base + 2]));             \
                out[g][3] += w * (vs * (float)((signed char)vb[2 + bj_base + 3]));             \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (int g = 0; g < G; g++) {                                                          \
            float* pg = partials + ((kv_h * G + g) * max_tiles + tile_id) * (2 + head_dim);    \
            pg[2 + d_base + 0] = out[g][0];                                                    \
            pg[2 + d_base + 1] = out[g][1];                                                    \
            pg[2 + d_base + 2] = out[g][2];                                                    \
            pg[2 + d_base + 3] = out[g][3];                                                    \
        }                                                                                      \
    }                                                                                          \
}

HFIRE_Q8_GQA_TILE(2, 8)
HFIRE_Q8_GQA_TILE(4, 8)
HFIRE_Q8_GQA_TILE(8, 4)
