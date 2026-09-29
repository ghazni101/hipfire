// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>


// Bidirectional ViT attention over INDEPENDENT row-segments: `seg_len` is the
// length of one segment (one InternVL tile's rows, or N for single-segment
// towers). Rows are grouped contiguously: segment s covers rows
// [s*seg_len, (s+1)*seg_len), and its softmax is bounded to that range —
// cross-segment attention would silently pollute every tile (each tile is an
// independent batch element in the HF reference; Taichu even duplicates the
// cls/register prefix per tile for exactly this reason).
// Grid: [num_heads, seg_len, N/seg_len]. seg_len must divide N evenly
// (host wrapper enforces; seg_len <= 0 degenerates to N = flat attention).
extern "C" __global__ void vit_attention_f32(
    const float* __restrict__ qkv,   // [N, 3*hidden] row-major
    float* __restrict__ out,          // [N, hidden] row-major
    int N, int hidden, int num_heads, int head_dim, float scale,
    int seg_len
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int qi_local = blockIdx.y;
    const int seg = blockIdx.z;
    const int n_seg = (seg_len > 0) ? seg_len : N;
    const int base = seg * n_seg;
    int qi = base + qi_local;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (head >= num_heads || qi_local >= n_seg) return;

    int q_off = head * head_dim;
    int k_off = hidden + head * head_dim;
    int v_off = 2 * hidden + head * head_dim;
    int stride = 3 * hidden;

    float* scores = sdata;

    // Phase 1: Q[qi] @ K[j]^T over this segment's keys only
    const float* q_ptr = qkv + qi * stride + q_off;
    float local_max = -1e30f;
    for (int j = tid; j < n_seg; j += nthreads) {
        const float* k_ptr = qkv + (base + j) * stride + k_off;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_ptr[d] * k_ptr[d];
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }

    float* ws = sdata + n_seg;
    ws[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < n_seg; j += nthreads) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float sum_val = ws[0];
    __syncthreads();

    for (int j = tid; j < n_seg; j += nthreads)
        scores[j] /= sum_val;
    __syncthreads();

    // Phase 2: weighted V sum over this segment's values only
    float* out_row = out + qi * hidden + head * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int j = 0; j < n_seg; j++)
            val += scores[j] * qkv[(base + j) * stride + v_off + d];
        out_row[d] = val;
    }
}
