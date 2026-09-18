// SPDX-License-Identifier: Apache-2.0
// HIP port of ifm-ai/uno nano_vllm_uno/engine/fused_verify_kernel.py.
// Reference: https://github.com/ifm-ai/uno (Apache-2.0).
// Dual softmax, rejection acceptance, and Gumbel-max residual correction.
// RNG is request-seeded Philox4x32-10, not Triton's tl.rand bitstream.
#include <hip/hip_runtime.h>
#include <stdint.h>

// Counter domains: acceptance=0, correction=1; token and row never overlap.
__device__ uint32_t uno_random(uint32_t seed, uint32_t row, uint32_t token, uint32_t domain) {
    uint32_t a=token, b=row, c=domain, d=0, k0=seed, k1=0x554e4f31u;
    for (int round=0; round<10; ++round) {
        uint64_t x=uint64_t(0xd2511f53u)*a, y=uint64_t(0xcd9e8d57u)*c;
        uint32_t na=uint32_t(y>>32)^b^k0, nc=uint32_t(x>>32)^d^k1;
        b=uint32_t(y); d=uint32_t(x); a=na; c=nc;
        k0+=0x9e3779b9u; k1+=0xbb67ae85u;
    }
    return a;
}
__device__ float uno_uniform(uint32_t seed, uint32_t row, uint32_t token, uint32_t domain) {
    // Open interval in F32, including after rounding.
    return (float(uno_random(seed,row,token,domain)>>9)+0.5f)*(1.0f/8388608.0f);
}

// Grid: rows blocks, exactly 256 threads. Result pairs: [accepted, correction].
// accepted=2 denotes invalid input/numerics; never fabricate a token on failure.
extern "C" __global__ void uno_verify_logits(
    const float* p_logits, const float* q_logits, const uint32_t* proposals,
    uint32_t* result, int vocab, float temperature, uint32_t seed) {
    const int tid=threadIdx.x, row=blockIdx.x;
    const float* p=p_logits+size_t(row)*vocab;
    const float* q=q_logits+size_t(row)*vocab;
    __shared__ float pm[256], qm[256], ps[256], qs[256];
    __shared__ int bad[256];
    __shared__ uint32_t ids[256];
    float pmax=-INFINITY, qmax=-INFINITY, ptotal=0, qtotal=0;
    int invalid=0;
    // Each lane maintains dual online softmax accumulators over its strip.
    for (int i=tid; i<vocab; i+=256) {
        float pv=p[i], qv=q[i];
        if (!isfinite(pv) || !isfinite(qv)) { invalid=1; continue; }
        // Subtract before division to avoid overflow for extreme finite logits.
        float pn=fmaxf(pmax,pv), qn=fmaxf(qmax,qv);
        ptotal=ptotal*expf((pmax-pn)/temperature)+expf((pv-pn)/temperature);
        qtotal=qtotal*expf((qmax-qn)/temperature)+expf((qv-qn)/temperature);
        pmax=pn; qmax=qn;
    }
    pm[tid]=pmax; qm[tid]=qmax; ps[tid]=ptotal; qs[tid]=qtotal; bad[tid]=invalid;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (tid<s) {
            float pn=fmaxf(pm[tid],pm[tid+s]), qn=fmaxf(qm[tid],qm[tid+s]);
            float pa=ps[tid]>0 ? ps[tid]*expf((pm[tid]-pn)/temperature) : 0;
            float pb=ps[tid+s]>0 ? ps[tid+s]*expf((pm[tid+s]-pn)/temperature) : 0;
            float qa=qs[tid]>0 ? qs[tid]*expf((qm[tid]-qn)/temperature) : 0;
            float qb=qs[tid+s]>0 ? qs[tid+s]*expf((qm[tid+s]-qn)/temperature) : 0;
            ps[tid]=pa+pb; qs[tid]=qa+qb; pm[tid]=pn; qm[tid]=qn;
            bad[tid]|=bad[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) {
        uint32_t t=proposals[row];
        uint32_t status=2;
        if (!bad[0] && t<uint32_t(vocab) && ps[0]>0 && qs[0]>0 && isfinite(ps[0]) && isfinite(qs[0])) {
            float pv=expf((p[t]-pm[0])/temperature)/ps[0];
            float qv=expf((q[t]-qm[0])/temperature)/qs[0];
            float ratio=qv>0 ? pv/qv : 0.0f;
            status=(ratio>=1.0f || uno_uniform(seed,row,0,0)<ratio) ? 1 : 0;
        }
        bad[0]=int(status);
        result[2*row]=status; result[2*row+1]=0;
    }
    __syncthreads();
    if (bad[0]!=0) return; // All lanes take the same branch.
    float best=-INFINITY; uint32_t pick=uint32_t(vocab);
    for (int i=tid; i<vocab; i+=256) {
        float pv=expf((p[i]-pm[0])/temperature)/ps[0];
        float qv=expf((q[i]-qm[0])/temperature)/qs[0];
        float residual=fmaxf(pv-qv,0.0f);
        // Match upstream's tiny-mass cutoff; no arbitrary argmax fallback.
        if (residual>1e-20f) {
            float u=uno_uniform(seed,row,uint32_t(i),1);
            float score=logf(residual)-logf(-logf(u));
            if (score>best || (score==best && uint32_t(i)<pick)) { best=score; pick=i; }
        }
    }
    // pm is no longer needed. Barrier before reusing it as reduction scratch.
    __syncthreads();
    pm[tid]=best; ids[tid]=pick;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (tid<s && (pm[tid+s]>pm[tid] || (pm[tid+s]==pm[tid] && ids[tid+s]<ids[tid]))) {
            pm[tid]=pm[tid+s]; ids[tid]=ids[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) {
        if (ids[0]>=uint32_t(vocab)) result[2*row]=2;
        else result[2*row+1]=ids[0];
    }
}

// Per-row logsumexp over temperature-scaled logits (Uno tree candidate
// masses): out[row] = max/temp + log(sum exp((x-max)/temp)). One block per
// row, 256 threads, online two-pass (max then sum) over the row.
extern "C" __global__ void uno_row_lse(
    const float* __restrict__ logits, float* __restrict__ out,
    int vocab, float inv_temp) {
    const int tid=threadIdx.x, row=blockIdx.x;
    const float* p=logits+size_t(row)*vocab;
    __shared__ float sm[256], ss[256];
    float tmax=-INFINITY, tsum=0.0f;
    for (int i=tid; i<vocab; i+=256) {
        float v=p[i]*inv_temp;
        float n=fmaxf(tmax, v);
        tsum=tsum*expf(tmax-n)+expf(v-n);
        tmax=n;
    }
    sm[tid]=tmax; ss[tid]=tsum;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (tid<s) {
            float n=fmaxf(sm[tid], sm[tid+s]);
            float a=ss[tid]>0 ? ss[tid]*expf(sm[tid]-n) : 0.0f;
            float b=ss[tid+s]>0 ? ss[tid+s]*expf(sm[tid+s]-n) : 0.0f;
            ss[tid]=a+b; sm[tid]=n;
        }
        __syncthreads();
    }
    if (tid==0) out[row]=sm[0]+logf(ss[0]);
}

// One masked-argmax round of an iterative top-k: each row block finds its
// strict-> lowest-index maximum, writes [idx, f32bits(val)] to out[row*2..],
// and knocks that lane down to -INF so the next round yields the runner-up.
// Callers loop k rounds on host (k tiny: candidate width of the draft tree).
extern "C" __global__ void uno_topk_round(
    float* __restrict__ logits, float* __restrict__ out,
    int vocab) {
    const int tid=threadIdx.x, row=blockIdx.x;
    float* p=logits+size_t(row)*vocab;
    __shared__ float vm[256];
    __shared__ int vi[256];
    float best=-INFINITY; int idx=0;
    for (int i=tid; i<vocab; i+=256) {
        float v=p[i];
        if (v>best) { best=v; idx=i; }
    }
    vm[tid]=best; vi[tid]=idx;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (tid<s && vm[tid+s]>vm[tid]) {
            vm[tid]=vm[tid+s]; vi[tid]=vi[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) {
        out[2*row]=(float)vi[0];
        out[2*row+1]=vm[0];
        p[vi[0]]=-INFINITY;
    }
}

// In-place top-K mask for Uno filtered verification: reduce each row to its
// top-K support, flooring every other lane to a FINITE -1e30 (not -INF, so
// the fused verifier's isfinite guard stays satisfied; exp(-1e30/temp)==0
// and the online-softmax max-scale recurrence carries it to zero). Keeping
// the top-K ORIGINAL values means the fused dual-softmax computes the
// filtered p/q exactly — the upstream build_sparse_top_k_probs support.
// top_k must be in [1, 1024].
extern "C" __global__ void uno_apply_topk_mask(
    float* __restrict__ logits, int vocab, int top_k) {
    const int tid=threadIdx.x, row=blockIdx.x;
    float* p=logits+size_t(row)*vocab;
    __shared__ float vm[256];
    __shared__ int vi[256];
    __shared__ int keep_idx[1024];
    __shared__ float keep_val[1024];
    for (int r=0; r<top_k; ++r) {
        float best=-INFINITY; int idx=vocab;
        for (int i=tid; i<vocab; i+=256) {
            float v=p[i];
            if (v>best) { best=v; idx=i; }
        }
        vm[tid]=best; vi[tid]=idx;
        __syncthreads();
        for (int s=128; s; s>>=1) {
            if (tid<s && vm[tid+s]>vm[tid]) { vm[tid]=vm[tid+s]; vi[tid]=vi[tid+s]; }
            __syncthreads();
        }
        if (tid==0 && r<1024 && vi[0]<vocab) {
            keep_idx[r]=vi[0];
            keep_val[r]=vm[0];
            p[vi[0]]=-INFINITY;
        }
        __syncthreads();
    }
    for (int i=tid; i<vocab; i+=256) p[i]=-1e30f;
    __syncthreads();
    for (int r=tid; r<top_k && r<1024; r+=256) {
        if (keep_idx[r]<vocab) p[keep_idx[r]]=keep_val[r];
    }
}
