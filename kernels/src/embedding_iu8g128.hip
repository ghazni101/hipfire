// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// IU8G128 (qt 58) embedding lookup: dequantize one row on GPU, output F32.
//
// This is the Q8 grid in the planar layout the iu4 products lift their
// embeddings onto: `[K signed codes][K / 128 f16 scales]`, no FWHT. The
// dequantized value is `w = f32(scale) * f32(i8(code))`, identical to
// `embedding_q8` on the interleaved Q8_0 block.

#include <hip/hip_runtime.h>

// qt58 rows are padded to a 16-byte multiple: K code bytes + K/64 scale
// bytes, rounded up (hipfire_runtime::iu_quant::q8_row_bytes).
#ifndef IU8_ROW_BYTES
#define IU8_ROW_BYTES(K) ((((long long)(K) + (long long)((K) >> 6)) + 15LL) & ~15LL)
#endif

extern "C" __global__ void embedding_iu8g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 128;
    const long long bytes_per_row = IU8_ROW_BYTES(dim);
    const unsigned char* row = table + (long long)token_id * bytes_per_row;
    const unsigned char* scales = row + dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        const int block_idx = elem >> 7;
        const float d = (float)*((const _Float16*)(scales + (block_idx << 1)));
        signed char qval = (signed char)row[elem];
        output[elem] = d * (float)qval;
    }
}
