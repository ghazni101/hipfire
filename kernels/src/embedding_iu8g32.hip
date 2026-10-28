// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// IU8G32 (qt 59) embedding lookup: dequantize one row on GPU, output F32.
//
// This is the Q8 grid in the planar layout the iu4/iu8 products lift their
// embeddings onto: `[K signed codes][f16 scale] * (K / 32)`, no FWHT. The
// dequantized value is `w = f32(scale) * f32(i8(code))`, identical to
// `embedding_q8` on the interleaved Q8_0 block.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_iu8g32(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const long long bytes_per_row =
        (long long)dim + (long long)(blocks_per_row << 1);
    const unsigned char* row = table + (long long)token_id * bytes_per_row;
    const unsigned char* scales = row + dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        const int block_idx = elem >> 5;
        const float d = (float)*((const _Float16*)(scales + (block_idx << 1)));
        signed char qval = (signed char)row[elem];
        output[elem] = d * (float)qval;
    }
}
