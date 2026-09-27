// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact GELU (erf form, PyTorch default `nn.GELU()`): out = x * Phi(x).
// timm ViT blocks (incl. the C-RADIO v4-h tower in ZDTaichu-5.0) use the
// exact erf form — gelu_tanh_f32's tanh approximation drifts ~1e-3, which is
// material for a 32-layer vision tower feeding RMSNorm'd projector inputs.
extern "C" __global__ void gelu_erf_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    out[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
}
