// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched IU8G32 (qt 59) embedding lookup: N rows of the planar Q8 table
// `[K signed codes][f16 scale] * (K / 32)` into `output[n, dim]` F32, token
// ids from a device buffer (hipGraph-captureable). Per element identical to
// `embedding_iu8g32` and to `embedding_q8_batched` on the Q8_0 twin.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_iu8g32_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_ids,
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const long long bytes_per_row =
        (long long)dim + (long long)(blocks_per_row << 1);
    const unsigned char* row = table + (long long)token_ids[batch_idx] * bytes_per_row;
    const unsigned char* scales = row + dim;
    float* out_row = output + (long long)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        const float d = (float)*((const _Float16*)(scales + ((elem >> 5) << 1)));
        out_row[elem] = d * (float)(signed char)row[elem];
    }
}
