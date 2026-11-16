// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched IU8G128 (qt 61) embedding lookup: N rows of the compact table
// `[K signed codes][K / 128 f16 scales]` into `output[n, dim]` F32, token
// ids from a device buffer (hipGraph-captureable). Per element identical to
// `embedding_iu8g128` and to `embedding_q8_batched` on the Q8_0 twin.

#include <hip/hip_runtime.h>

// qt61 rows are padded to a 16-byte multiple: K code bytes + K/64 scale
// bytes, rounded up (hipfire_runtime::iu_quant::q8_row_bytes).
#ifndef IU8_ROW_BYTES
#define IU8_ROW_BYTES(K) ((((long long)(K) + (long long)((K) >> 6)) + 15LL) & ~15LL)
#endif

extern "C" __global__ void embedding_iu8g128_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_ids,
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 128;
    const long long bytes_per_row = IU8_ROW_BYTES(dim);
    const unsigned char* row = table + (long long)token_ids[batch_idx] * bytes_per_row;
    const unsigned char* scales = row + dim;
    float* out_row = output + (long long)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        const float d = (float)*((const _Float16*)(scales + ((elem >> 7) << 1)));
        out_row[elem] = d * (float)(signed char)row[elem];
    }
}
