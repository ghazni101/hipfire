// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Squared ReLU: out = relu(x)^2 = max(0, x)^2. ZDTaichu-5.0's vision
// projector (mlp1) uses this between its two bias-free linears.
extern "C" __global__ void squared_relu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fmaxf(x[i], 0.0f);
    out[i] = v * v;
}
