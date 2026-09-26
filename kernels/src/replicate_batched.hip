// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Replicate [N × K] → [N × K_TOP × K] by copying each token's K-vector K_TOP times.
// Used by MoVA value routing in batched prefill: the indexed down GEMV expects
// rot_batch[bid][krank][...] but the rotated input is only [N × K] (shared
// across k_top experts for the same token).

extern "C" __global__ void replicate_batched_f32(
    const float* __restrict__ in,   // [N × K]
    float* __restrict__ out,         // [N × K_TOP × K]
    int K, int K_TOP, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K_TOP * K;
    if (idx >= total) return;
    int k = idx % K;
    int n = idx / (K_TOP * K);
    out[idx] = in[n * K + k];
}
