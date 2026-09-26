// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused grouped RMSNorm + FWHT rotation for MagnumQuant inputs.
// K2-Horizon (layernorm_num_groups=2): variance is computed per
// n/n_groups contiguous chunk, then each chunk's 256-element groups are
// FWHT-rotated exactly like fused_rmsnorm_mq_rotate.
//
// Grid: (batch * n_groups) blocks — one block per (row, group-chunk),
// matching grouped_rmsnorm_f32. Block: 256 threads.
// Requires chunk_len = n / n_groups to be a multiple of 256 (FWHT group).
//
// Writes BOTH outputs:
//   out   = x * weight * rms(chunk)          (the normed vector)
//   x_rot = FWHT(out) * signs                (the rotated vector)
// Math order matches grouped_rmsnorm_f32 + rotate_x_mq exactly:
//   ((x * w) * rms) * s1 → butterfly → * s2 * (1/16).

__launch_bounds__(256, 1)
extern "C" __global__ void grouped_rmsnorm_mq_rotate(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ out,
    float* __restrict__ x_rot,
    int n,
    int n_groups,
    float eps
) {
    extern __shared__ float smem[];
    float* reduce = smem;              // [256]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x; // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;      // 0..7
    const int num_warps = block_size >> 5;

    const int chunk_len = n / n_groups;
    const int row = blockIdx.x / n_groups;
    const int group = blockIdx.x % n_groups;
    const int offset = row * n + group * chunk_len;

    const float* x_c     = x     + offset;
          float* out_c   = out   + offset;
          float* x_rot_c = x_rot + offset;
    // weight is [n] shared across batch rows — index by global position.
    const float* w_c     = weight + group * chunk_len;

    // Phase 1a: sum-of-squares over this chunk.
    float local_sum = 0.0f;
    for (int i = tid; i < chunk_len; i += block_size) {
        float v = x_c[i];
        local_sum += v * v;
    }

    // Prefetch first FWHT group's x/weight (+ lane-local signs) so the
    // loads ride under the Phase-1b barriers.
    const int groups_total = chunk_len / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;
    const int d0 = lane_id * 8;

    float4 pref_x0, pref_x1, pref_w0, pref_w1;
    float4 s10, s11, s20, s21;
    const int pref_group = warp_group_start;
    const bool have_pref = (groups_per_warp > 0 && pref_group < groups_total);
    if (have_pref) {
        const int base = pref_group * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x_c + base);
        pref_x1 = *reinterpret_cast<const float4*>(x_c + base + 4);
        pref_w0 = *reinterpret_cast<const float4*>(w_c + base);
        pref_w1 = *reinterpret_cast<const float4*>(w_c + base + 4);
    }
    s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    // Phase 1b: block reduction -> per-chunk RMS.
    float warp_sum = local_sum;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);
    if (lane_id == 0) reduce[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        warp_sum = (tid < num_warps) ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)chunk_len + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

    // Phase 2: write normed output + FWHT per 256-group.
    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int g = warp_group_start + gi;
        if (g >= groups_total) break;

        const int base = g * 256 + d0;

        float4 x0, x1, w0, w1;
        if (gi == 0 && have_pref) {
            x0 = pref_x0; x1 = pref_x1;
            w0 = pref_w0; w1 = pref_w1;
        } else {
            x0 = *reinterpret_cast<const float4*>(x_c + base);
            x1 = *reinterpret_cast<const float4*>(x_c + base + 4);
            w0 = *reinterpret_cast<const float4*>(w_c + base);
            w1 = *reinterpret_cast<const float4*>(w_c + base + 4);
        }

        // Exact multiply order: ((x * weight) * rms) — this is `out`.
        float n0 = x0.x * w0.x * rms;
        float n1 = x0.y * w0.y * rms;
        float n2 = x0.z * w0.z * rms;
        float n3 = x0.w * w0.w * rms;
        float n4 = x1.x * w1.x * rms;
        float n5 = x1.y * w1.y * rms;
        float n6 = x1.z * w1.z * rms;
        float n7 = x1.w * w1.w * rms;

        float4 o0 = make_float4(n0, n1, n2, n3);
        float4 o1 = make_float4(n4, n5, n6, n7);
        *reinterpret_cast<float4*>(out_c + base) = o0;
        *reinterpret_cast<float4*>(out_c + base + 4) = o1;

        // Then * sign1, matching fused_rmsnorm_mq_rotate's order.
        float v0 = n0 * s10.x;
        float v1 = n1 * s10.y;
        float v2 = n2 * s10.z;
        float v3 = n3 * s10.w;
        float v4 = n4 * s11.x;
        float v5 = n5 * s11.y;
        float v6 = n6 * s11.z;
        float v7 = n7 * s11.w;

        // Local butterfly: strides 1, 2, 4.
        float t;
        t = v0; v0 = v0 + v1; v1 = t - v1;
        t = v2; v2 = v2 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v5; v5 = t - v5;
        t = v6; v6 = v6 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v2; v2 = t - v2;
        t = v1; v1 = v1 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v6; v6 = t - v6;
        t = v5; v5 = v5 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v4; v4 = t - v4;
        t = v1; v1 = v1 + v5; v5 = t - v5;
        t = v2; v2 = v2 + v6; v6 = t - v6;
        t = v3; v3 = v3 + v7; v7 = t - v7;

        #define HBFLY(v, pat, str) do { \
            float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8(pat, str) \
            HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
            HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

        HBFLY8(0x041F, 1);
        HBFLY8(0x081F, 2);
        HBFLY8(0x101F, 4);
        HBFLY8(0x201F, 8);
        HBFLY8(0x401F, 16);
        #undef HBFLY8
        #undef HBFLY

        const float s = 0.0625f;
        float4 r0, r1;
        r0.x = v0 * s * s20.x;
        r0.y = v1 * s * s20.y;
        r0.z = v2 * s * s20.z;
        r0.w = v3 * s * s20.w;
        r1.x = v4 * s * s21.x;
        r1.y = v5 * s * s21.y;
        r1.z = v6 * s * s21.z;
        r1.w = v7 * s * s21.w;
        *reinterpret_cast<float4*>(x_rot_c + base) = r0;
        *reinterpret_cast<float4*>(x_rot_c + base + 4) = r1;
    }
}
