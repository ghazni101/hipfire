// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched grouped RMSNorm: `rows` independent vectors of `groups * width`
// elements, each group normalized with its own affine slice. This is the
// K2-Horizon layernorm_num_groups>1 shape; one launch replaces the
// rows*groups grid of plain rmsnorm_f32 calls (4 rows x 4 groups = 16
// launches per site -> 1), which dominated the grouped-norm batch forward's
// launch count.
//
// Grid: rows*groups blocks (one (row, group) pair each). Weight layout is
// [groups*width] with group g's slice at g*width — matches the per-group
// sub_offset views LlamaConfig::rmsnorm already uses.

#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_grouped_batched(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int width, int groups, float eps
) {
    extern __shared__ float sdata[];
    const int group = blockIdx.x % groups;
    const int row = blockIdx.x / groups;
    const size_t base = (size_t)row * groups * width + (size_t)group * width;
    const float* xg = x + base;
    float* og = out + base;
    const float* wg = weight + (size_t)group * width;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = xg[i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        og[i] = xg[i] * wg[i] * rms;
    }
}
