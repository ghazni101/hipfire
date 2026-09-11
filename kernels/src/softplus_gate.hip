// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused softplus post-attention gate for K2-Horizon MoVA.
//
// Replaces the 4-kernel sequence:
//   scale_f32(gate, LN_2)        // gate *= beta
//   softplus_f32(gate)           // gate = log(1 + exp(gate))
//   scale_f32(gate, 1/LN_2)      // gate /= beta
//   mul_f32(attn_out, gate, attn_out)  // attn_out *= gate
//
// with a single kernel:
//   attn_out[i] *= log(1 + exp(LN_2 * gate[i])) / LN_2
//
// LN_2 = 0.6931...  The branched formulation matches softplus_f32 exactly
// after the pre/post scaling: for v = LN_2 * gate[i],
//   v > 20  → log(1+exp(v)) ≈ v        → result = gate[i]
//   v < -20 → log(1+exp(v)) ≈ exp(v)   → result = exp(LN_2*gate[i]) / LN_2
//   else    → log(1+exp(v)) / LN_2

extern "C" __global__ void softplus_gate_f32(
    const float* __restrict__ gate,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float v = 0.6931471805599453f * g;
        float sp;
        if (v > 20.0f) {
            sp = g;  // log(1+exp(LN2*g)) / LN2 ≈ g
        } else if (v < -20.0f) {
            sp = expf(v) / 0.6931471805599453f;
        } else {
            sp = logf(1.0f + expf(v)) / 0.6931471805599453f;
        }
        out[i] *= sp;
    }
}
