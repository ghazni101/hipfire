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

// Filtered Uno verification via candidate gather — the device-resident
// nucleus (top-p) path. Two kernels mirror the fused AR sampler's exact law
// (softmax probabilities are monotone in logit, so the top-K gather by logit
// value is already in probability order):
//
//   `uno_filter_gather`  — per row, the row's top-64 logits by value (the
//                          sampler's partial+merge selection), written to
//                          global scratch for the target (p) and draft (q)
//                          rows.
//   `uno_filter_finalize` — per row, build the two sampling laws with the
//                          sampler's arithmetic — softmax at `temperature`,
//                          cap at `top_k_req` candidates (request top_k, else
//                          the sampler's 20 when the request sets no top_k),
//                          nucleus-truncate at `top_p` over the descending
//                          list, renormalize over the truncated support (the
//                          sampler's `trunc_sum`) — then decide accept/correct.
//
// The proposals were drawn by that same law (`sample_row` →
// `sample_top_p_pf`), so the acceptance ratio p(t)/q(t) and the (p−q)+ residual
// correction emit exactly the target's AR-sampler-law distribution. Emission
// law therefore matches what the daemon's non-speculative AR decode produces
// at the same request config (temp + top_p + top_k), replacing the old host
// Distribution law (full-vocab nucleus) that forced a full logits download per
// row per forward.
//
// `uno_filter_gather` grid: rows blocks, 128 threads; dynamic smem =
// 128*64*8 bytes (64 KiB, the RDNA wave32 group-segment limit — same footprint
// as the wide sampler partial; no static smem on top, so the launch fits).
// `uno_filter_finalize` grid: rows blocks, 1 thread; no smem. Result pairs:
// [accepted, correction]; accepted=2 is a fail-closed numeric error (mirrors
// `uno_verify_logits`).
#define UNO_FILTER_KW 64
extern "C" __global__ void uno_filter_gather(
    const float* __restrict__ logits, float* __restrict__ out_val,
    int* __restrict__ out_idx, int vocab) {
    extern __shared__ char smem[];
    const int tid=threadIdx.x, nthreads=blockDim.x, row=blockIdx.x;
    const float* p=logits+size_t(row)*vocab;
    float* s_val=(float*)smem;                       // [nthreads*KW]
    int*   s_idx=(int*)(s_val+(size_t)nthreads*UNO_FILTER_KW);
    float my_val[UNO_FILTER_KW];
    int   my_idx[UNO_FILTER_KW];
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) { my_val[k]=-1e30f; my_idx[k]=0; }
    int min_pos=0;
    float min_val=-1e30f;
    for (int i=tid; i<vocab; i+=nthreads) {
        float v=p[i];
        if (v>min_val) {
            my_val[min_pos]=v;
            my_idx[min_pos]=i;
            min_val=my_val[0]; min_pos=0;
            #pragma unroll
            for (int k=1; k<UNO_FILTER_KW; ++k) {
                if (my_val[k]<min_val) { min_val=my_val[k]; min_pos=k; }
            }
        }
    }
    // Block merge (the sampler's non-fast merge: nthreads x KW -> block top-KW).
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) {
        s_val[(size_t)tid*UNO_FILTER_KW+k]=my_val[k];
        s_idx[(size_t)tid*UNO_FILTER_KW+k]=my_idx[k];
    }
    __syncthreads();
    for (int stride=nthreads/2; stride; stride>>=1) {
        if (tid<stride) {
            float* a_val=s_val+(size_t)tid*UNO_FILTER_KW;
            int*   a_idx=s_idx+(size_t)tid*UNO_FILTER_KW;
            const float* b_val=s_val+(size_t)(tid+stride)*UNO_FILTER_KW;
            const int*   b_idx=s_idx+(size_t)(tid+stride)*UNO_FILTER_KW;
            for (int j=0; j<UNO_FILTER_KW; ++j) {
                float bv=b_val[j]; int bi=b_idx[j];
                float amin=a_val[0]; int amin_pos=0;
                #pragma unroll
                for (int k=1; k<UNO_FILTER_KW; ++k) {
                    if (a_val[k]<amin) { amin=a_val[k]; amin_pos=k; }
                }
                if (bv>amin) { a_val[amin_pos]=bv; a_idx[amin_pos]=bi; }
            }
        }
        __syncthreads();
    }
    if (tid==0) {
        float* o_val=out_val+size_t(row)*UNO_FILTER_KW;
        int*   o_idx=out_idx+size_t(row)*UNO_FILTER_KW;
        #pragma unroll
        for (int k=0; k<UNO_FILTER_KW; ++k) { o_val[k]=s_val[k]; o_idx[k]=s_idx[k]; }
    }
}

extern "C" __global__ void uno_filter_finalize(
    const float* __restrict__ p_val, const int* __restrict__ p_idx,
    const float* __restrict__ q_val, const int* __restrict__ q_idx,
    const uint32_t* __restrict__ proposals, uint32_t* __restrict__ result,
    int vocab, float temperature, float top_p, int top_k_req, uint32_t seed) {
    const int row=blockIdx.x;
    const float* pv_all=p_val+size_t(row)*UNO_FILTER_KW;
    const int*   pi_all=p_idx+size_t(row)*UNO_FILTER_KW;
    const float* qv_all=q_val+size_t(row)*UNO_FILTER_KW;
    const int*   qi_all=q_idx+size_t(row)*UNO_FILTER_KW;
    int cap_lo=top_k_req; if (cap_lo<1) cap_lo=1;
    if (cap_lo>UNO_FILTER_KW) cap_lo=UNO_FILTER_KW;
    float inv_temp=1.0f/temperature;

    // ── p law: descending logit order == descending probability order.
    float pv[UNO_FILTER_KW]; int pi[UNO_FILTER_KW];
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) { pv[k]=pv_all[k]; pi[k]=pi_all[k]; }
    for (int i=1; i<UNO_FILTER_KW; ++i) {
        float cv=pv[i]; int ci=pi[i];
        int j=i;
        while (j>0 && pv[j-1]<cv) { pv[j]=pv[j-1]; pi[j]=pi[j-1]; --j; }
        pv[j]=cv; pi[j]=ci;
    }
    float pmax=pv[0];
    float pprobs[UNO_FILTER_KW];
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) {
        pprobs[k]=(pv[k]<=-1e29f) ? 0.0f : expf((pv[k]-pmax)*inv_temp);
    }
    float psum=0.0f;
    for (int i=0; i<cap_lo; ++i) psum+=pprobs[i];
    float pthresh=top_p*psum;
    float pcum=0.0f;
    int ptrunc=cap_lo;
    for (int i=0; i<cap_lo; ++i) {
        pcum+=pprobs[i];
        if (pcum>=pthresh) { ptrunc=i+1; break; }
    }
    float pnorm=0.0f;
    for (int i=0; i<ptrunc; ++i) pnorm+=pprobs[i];

    // ── q law: identical arithmetic over the draft row.
    float qv[UNO_FILTER_KW]; int qi[UNO_FILTER_KW];
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) { qv[k]=qv_all[k]; qi[k]=qi_all[k]; }
    for (int i=1; i<UNO_FILTER_KW; ++i) {
        float cv=qv[i]; int ci=qi[i];
        int j=i;
        while (j>0 && qv[j-1]<cv) { qv[j]=qv[j-1]; qi[j]=qi[j-1]; --j; }
        qv[j]=cv; qi[j]=ci;
    }
    float qmax=qv[0];
    float qprobs[UNO_FILTER_KW];
    #pragma unroll
    for (int k=0; k<UNO_FILTER_KW; ++k) {
        qprobs[k]=(qv[k]<=-1e29f) ? 0.0f : expf((qv[k]-qmax)*inv_temp);
    }
    float qsum=0.0f;
    for (int i=0; i<cap_lo; ++i) qsum+=qprobs[i];
    float qthresh=top_p*qsum;
    float qcum=0.0f;
    int qtrunc=cap_lo;
    for (int i=0; i<cap_lo; ++i) {
        qcum+=qprobs[i];
        if (qcum>=qthresh) { qtrunc=i+1; break; }
    }
    float qnorm=0.0f;
    for (int i=0; i<qtrunc; ++i) qnorm+=qprobs[i];

    const uint32_t t=proposals[row];
    float pt=0.0f, qt=0.0f;
    for (int i=0; i<ptrunc; ++i) {
        if ((uint32_t)pi[i]==t) { pt=pprobs[i]/pnorm; break; }
    }
    for (int i=0; i<qtrunc; ++i) {
        if ((uint32_t)qi[i]==t) { qt=qprobs[i]/qnorm; break; }
    }
    uint32_t status=2;
    if (pmax>-1e29f && qmax>-1e29f && pnorm>0.0f && qnorm>0.0f
        && isfinite(pnorm) && isfinite(qnorm) && t<(uint32_t)vocab) {
        float ratio=(qt>0.0f) ? (pt/qt) : 0.0f;
        status=(ratio>=1.0f || uno_uniform(seed,row,0,0)<ratio) ? 1 : 0;
    }
    result[2*row]=status; result[2*row+1]=0;
    if (status!=0) return;
    // Residual corrective draw: (p−q)+ is nonzero only on the p support
    // (q-only tokens have p=0 <= q). Gumbel-max over the truncated p support
    // with per-token q lookup — mirrors uno_verify_logits.
    float best=-INFINITY;
    uint32_t pick=(uint32_t)vocab;
    for (int i=0; i<ptrunc; ++i) {
        uint32_t id=(uint32_t)pi[i];
        float qv_at=0.0f;
        for (int j=0; j<qtrunc; ++j) {
            if ((uint32_t)qi[j]==id) { qv_at=qprobs[j]/qnorm; break; }
        }
        float residual=fmaxf(pprobs[i]/pnorm-qv_at, 0.0f);
        if (residual>1e-20f) {
            float u=uno_uniform(seed,row,id,1);
            float score=logf(residual)-logf(-logf(u));
            if (score>best || (score==best && id<pick)) { best=score; pick=id; }
        }
    }
    if (pick>=uint32_t(vocab)) result[2*row]=2;
    else result[2*row+1]=pick;
}
