// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Device-side lane-union planner of the sealed expert-slab decode (K1,
// `mq_i4_moe_gemv`). One workgroup turns the call's routed ids
// (`topk[S]`, `S = 10 * lanes`, slot = lane * 10 + rank) into the fixed-capacity
// per-codec-bucket descriptors the four K1 symbols read, with no host readback:
// the bucket counts are data dependent, so every bucket is laid out at its
// worst-case capacity `S` and its unused tail is empty (`n = 0` experts, which
// the K1 kernels skip).
//
// Buckets (K1's fixed codec order): 0 = W3 (codec byte 1), 1 = MQ4 (codec byte
// 0). `uniform_bucket >= 0` puts every slot in that bucket and ignores the
// codec table (a layer whose experts all share one codec carries no table).
//
// Output words (u32), for bucket b in {0, 1}:
//   expert_ids [b * S + e]            distinct experts of the bucket in
//                                      first-occurrence order, tail 0
//   expert_offsets [2 S + b (S+1) + e] run of S + 1 offsets from 0 to the
//                                      bucket's slot count n_b; tail = n_b
//   slot_ids [2 S + 2 (S+1) + b S + p] original slot ids, grouped by expert in
//                                      ascending slot order, tail 0
// Total 6 S + 2 words. Every word is written exactly once per invocation.

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "qwen4_slab_union_plan.gfx1201.hip is gfx1201-only"
#endif

#define SLAB_PLAN_MAX_SLOTS 128

extern "C" __global__ __launch_bounds__(128) void qwen4_slab_union_plan(
    const int* __restrict__ topk, const unsigned char* __restrict__ codec,
    unsigned* __restrict__ out, int S, int uniform_bucket) {
    __shared__ int s_id[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_b[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_first[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_total[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_eidx[SLAB_PLAN_MAX_SLOTS];
    const int t = threadIdx.x;
    if (t < S) {
        int id = topk[t];
        id = ((unsigned)id < 512u) ? id : 0;
        s_id[t] = id;
        s_b[t] = uniform_bucket >= 0 ? uniform_bucket : (codec[id] == 1 ? 0 : 1);
    }
    __syncthreads();
    int rank = 0, total = 0, f = t;
    if (t < S) {
        const int id = s_id[t];
        f = -1;
        for (int j = 0; j < S; ++j) {
            if (s_id[j] == id) {
                if (f < 0) f = j;
                ++total;
                if (j < t) ++rank;
            }
        }
        s_first[t] = rank == 0;
        s_total[t] = total;
    }
    __syncthreads();
    if (t < S && rank == 0) {
        const int b = s_b[t];
        int e = 0;
        for (int j = 0; j < t; ++j) e += s_first[j] && s_b[j] == b;
        s_eidx[t] = e;
    }
    __syncthreads();
    unsigned* ids = out;
    unsigned* offs = out + 2 * S;
    unsigned* slots = out + 2 * S + 2 * (S + 1);
    int ub0 = 0, ub1 = 0, nb0 = 0, nb1 = 0;
    for (int j = 0; j < S; ++j) {
        if (s_first[j]) {
            if (s_b[j] == 0) { ++ub0; nb0 += s_total[j]; } else { ++ub1; nb1 += s_total[j]; }
        }
    }
    if (t < S) {
        const int b = s_b[t];
        const int e = s_eidx[f];
        int start = 0;
        for (int j = 0; j < S; ++j) {
            if (s_first[j] && s_b[j] == b && s_eidx[j] < e) start += s_total[j];
        }
        slots[b * S + start + rank] = (unsigned)t;
        if (rank == 0) {
            ids[b * S + e] = (unsigned)s_id[t];
            offs[b * (S + 1) + e] = (unsigned)start;
            offs[b * (S + 1) + e + 1] = (unsigned)(start + total);
        }
        if (t >= ub0) { ids[t] = 0u; offs[t + 1] = (unsigned)nb0; }
        if (t >= ub1) { ids[S + t] = 0u; offs[(S + 1) + t + 1] = (unsigned)nb1; }
        if (t >= nb0) slots[t] = 0u;
        if (t >= nb1) slots[S + t] = 0u;
        if (t == 0) { offs[0] = 0u; offs[S + 1] = 0u; }
    }
}
