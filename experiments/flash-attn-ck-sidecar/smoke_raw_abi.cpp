// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
        std::exit(2);
    }
}

size_t offset(int b, int s, int h, int d, int seqlen, int heads, int hdim)
{
    return ((static_cast<size_t>(b) * seqlen + s) * heads + h) * hdim + d;
}

float load(const std::vector<__half>& values,
           int b,
           int s,
           int h,
           int d,
           int seqlen,
           int heads,
           int hdim)
{
    return __half2float(values[offset(b, s, h, d, seqlen, heads, hdim)]);
}

void run_case(const char* name, int nhead_q, int nhead_k, bool causal, bool non_default_stream)
{
    constexpr int batch = 1;
    constexpr int seqlen_q = 64;
    constexpr int seqlen_k = 96;
    constexpr int hdim = 64;
    constexpr float scale = 1.0f / 8.0f;
    const int groups = nhead_q / nhead_k;

    const size_t q_count = static_cast<size_t>(batch) * seqlen_q * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(batch) * seqlen_k * nhead_k * hdim;
    std::vector<__half> q(q_count), k(kv_count), v(kv_count), output(q_count);
    std::vector<float> expected(q_count, 0.0f);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for(auto* values : {&q, &k, &v})
    {
        for(auto& value : *values)
        {
            value = __float2half(distribution(rng));
        }
    }

    for(int b = 0; b < batch; ++b)
    {
        for(int hq = 0; hq < nhead_q; ++hq)
        {
            const int hk = hq / groups;
            for(int sq = 0; sq < seqlen_q; ++sq)
            {
                std::vector<float> scores(seqlen_k);
                float maximum = -INFINITY;
                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    if(causal && sk > sq + seqlen_k - seqlen_q)
                    {
                        scores[sk] = -INFINITY;
                        continue;
                    }
                    float score = 0.0f;
                    for(int d = 0; d < hdim; ++d)
                    {
                        score += load(q, b, sq, hq, d, seqlen_q, nhead_q, hdim) *
                                 load(k, b, sk, hk, d, seqlen_k, nhead_k, hdim);
                    }
                    scores[sk] = score * scale;
                    maximum = std::max(maximum, scores[sk]);
                }
                float denominator = 0.0f;
                for(float& score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for(int d = 0; d < hdim; ++d)
                {
                    float value = 0.0f;
                    for(int sk = 0; sk < seqlen_k; ++sk)
                    {
                        value += scores[sk] / denominator *
                                 load(v, b, sk, hk, d, seqlen_k, nhead_k, hdim);
                    }
                    expected[offset(b, sq, hq, d, seqlen_q, nhead_q, hdim)] = value;
                }
            }
        }
    }

    void* q_device = nullptr;
    void* k_device = nullptr;
    void* v_device = nullptr;
    void* out_device = nullptr;
    check_hip(hipMalloc(&q_device, q_count * sizeof(__half)), "hipMalloc(q)");
    check_hip(hipMalloc(&k_device, kv_count * sizeof(__half)), "hipMalloc(k)");
    check_hip(hipMalloc(&v_device, kv_count * sizeof(__half)), "hipMalloc(v)");
    check_hip(hipMalloc(&out_device, q_count * sizeof(__half)), "hipMalloc(out)");
    hipStream_t stream = nullptr;
    if(non_default_stream)
    {
        check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    }
    check_hip(hipMemcpyAsync(q_device,
                            q.data(),
                            q_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(q)");
    check_hip(hipMemcpyAsync(k_device,
                            k.data(),
                            kv_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(k)");
    check_hip(hipMemcpyAsync(v_device,
                            v.data(),
                            kv_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(v)");

    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = q_device;
    params.k = k_device;
    params.v = v_device;
    params.out = out_device;
    params.stream = reinterpret_cast<void*>(stream);
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F16;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    params.batch = batch;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q;
    params.nhead_k = nhead_k;
    params.head_dim = hdim;
    params.causal = causal ? 1 : 0;
    params.softmax_scale = scale;
    params.stride_q = nhead_q * hdim;
    params.stride_k = nhead_k * hdim;
    params.stride_v = nhead_k * hdim;
    params.stride_out = nhead_q * hdim;
    params.nhead_stride_q = hdim;
    params.nhead_stride_k = hdim;
    params.nhead_stride_v = hdim;
    params.nhead_stride_out = hdim;
    params.batch_stride_q = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = seqlen_k * nhead_k * hdim;
    params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.batch_stride_out = seqlen_q * nhead_q * hdim;

    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0)
    {
        std::fprintf(stderr, "sidecar status=%d: %s\n", status, error);
        std::exit(3);
    }
    check_hip(hipMemcpyAsync(output.data(),
                            out_device,
                            q_count * sizeof(__half),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(out)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t i = 0; i < q_count; ++i)
    {
        const float delta = std::abs(__half2float(output[i]) - expected[i]);
        max_abs = std::max(max_abs, delta);
        mean_abs += delta;
    }
    mean_abs /= q_count;
    std::printf("case=%s dtype=fp16 q_heads=%d kv_heads=%d causal=%d stream=%s "
                "max_abs=%.7g mean_abs=%.7g\n",
                name,
                nhead_q,
                nhead_k,
                causal ? 1 : 0,
                non_default_stream ? "non-default" : "default",
                max_abs,
                mean_abs);
    if(max_abs > 0.02f)
    {
        std::exit(4);
    }

    check_hip(hipFree(q_device), "hipFree(q)");
    check_hip(hipFree(k_device), "hipFree(k)");
    check_hip(hipFree(v_device), "hipFree(v)");
    check_hip(hipFree(out_device), "hipFree(out)");
    if(non_default_stream)
    {
        check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    }
}

void pack_q8(const std::vector<float>& input,
             std::vector<uint8_t>& packed,
             std::vector<float>& decoded,
             int rows,
             int heads)
{
    constexpr int hdim = 256;
    constexpr int head_bytes = 272;
    for(int row = 0; row < rows; ++row)
    {
        for(int head = 0; head < heads; ++head)
        {
            for(int block = 0; block < 8; ++block)
            {
                const size_t input_base = (static_cast<size_t>(row) * heads + head) * hdim + block * 32;
                const size_t packed_base = (static_cast<size_t>(row) * heads + head) * head_bytes + block * 34;
                float maximum = 0.0f;
                for(int index = 0; index < 32; ++index)
                    maximum = std::max(maximum, std::abs(input[input_base + index]));
                const float scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
                const __half scale_half = __float2half(scale);
                std::memcpy(packed.data() + packed_base, &scale_half, sizeof(scale_half));
                const float stored_scale = __half2float(scale_half);
                for(int index = 0; index < 32; ++index)
                {
                    const int quantized = stored_scale > 0.0f
                                              ? static_cast<int>(std::nearbyint(input[input_base + index] / stored_scale))
                                              : 0;
                    const int8_t value = static_cast<int8_t>(std::clamp(quantized, -127, 127));
                    std::memcpy(packed.data() + packed_base + 2 + index, &value, 1);
                    decoded[input_base + index] = stored_scale * static_cast<float>(value);
                }
            }
        }
    }
}

void run_q8_d256_case()
{
    constexpr int seqlen_q = 16;
    constexpr int seqlen_k = 32;
    constexpr int nhead_q = 4;
    constexpr int nhead_k = 2;
    constexpr int hdim = 256;
    constexpr int head_bytes = 272;
    constexpr float scale = 1.0f / 16.0f;
    const size_t q_count = static_cast<size_t>(seqlen_q) * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(seqlen_k) * nhead_k * hdim;
    std::vector<float> q(q_count), k(kv_count), v(kv_count), decoded_k(kv_count),
        decoded_v(kv_count), output(q_count), expected(q_count);
    std::vector<uint8_t> packed_k(static_cast<size_t>(seqlen_k) * nhead_k * head_bytes);
    std::vector<uint8_t> packed_v(packed_k.size());
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for(auto* values : {&q, &k, &v})
        for(float& value : *values) value = distribution(rng);
    pack_q8(k, packed_k, decoded_k, seqlen_k, nhead_k);
    pack_q8(v, packed_v, decoded_v, seqlen_k, nhead_k);

    for(int hq = 0; hq < nhead_q; ++hq)
    {
        const int hk = hq / (nhead_q / nhead_k);
        for(int sq = 0; sq < seqlen_q; ++sq)
        {
            std::vector<float> scores(seqlen_k, -INFINITY);
            float maximum = -INFINITY;
            const int last_key = sq + seqlen_k - seqlen_q;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                float score = 0.0f;
                for(int d = 0; d < hdim; ++d)
                    score += q[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] *
                             decoded_k[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
                scores[sk] = score * scale;
                maximum = std::max(maximum, scores[sk]);
            }
            float denominator = 0.0f;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                scores[sk] = std::exp(scores[sk] - maximum);
                denominator += scores[sk];
            }
            for(int d = 0; d < hdim; ++d)
                for(int sk = 0; sk <= last_key; ++sk)
                    expected[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] +=
                        scores[sk] / denominator *
                        decoded_v[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
        }
    }

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *workspace = nullptr;
    check_hip(hipMalloc(&dq, q_count * sizeof(float)), "hipMalloc(q8 q)");
    check_hip(hipMalloc(&dk, packed_k.size()), "hipMalloc(q8 k)");
    check_hip(hipMalloc(&dv, packed_v.size()), "hipMalloc(q8 v)");
    check_hip(hipMalloc(&dout, q_count * sizeof(float)), "hipMalloc(q8 out)");
    check_hip(hipMemcpy(dq, q.data(), q_count * sizeof(float), hipMemcpyHostToDevice), "copy q8 q");
    check_hip(hipMemcpy(dk, packed_k.data(), packed_k.size(), hipMemcpyHostToDevice), "copy q8 k");
    check_hip(hipMemcpy(dv, packed_v.data(), packed_v.size(), hipMemcpyHostToDevice), "copy q8 v");

    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = dq; params.k = dk; params.v = dv; params.out = dout;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q; params.nhead_k = nhead_k; params.head_dim = hdim;
    params.causal = 1; params.softmax_scale = scale;
    params.stride_q = nhead_q * hdim; params.stride_k = nhead_k * hdim;
    params.stride_v = nhead_k * hdim; params.stride_out = nhead_q * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v =
        params.nhead_stride_out = hdim;
    params.batch_stride_q = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.batch_stride_out = params.batch_stride_q;
    params.packed_k_row_stride_bytes = params.packed_v_row_stride_bytes = nhead_k * head_bytes;
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    check_hip(hipMalloc(&workspace, params.workspace_bytes), "hipMalloc(q8 workspace)");
    params.workspace = workspace;
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0) { std::fprintf(stderr, "q8 sidecar status=%d: %s\n", status, error); std::exit(5); }
    check_hip(hipMemcpy(output.data(), dout, q_count * sizeof(float), hipMemcpyDeviceToHost), "copy q8 out");
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t index = 0; index < q_count; ++index)
    {
        const float delta = std::abs(output[index] - expected[index]);
        max_abs = std::max(max_abs, delta); mean_abs += delta;
    }
    mean_abs /= q_count;
    std::printf("case=q8-d256-gqa-causal max_abs=%.7g mean_abs=%.7g workspace=%zu\n",
                max_abs, mean_abs, params.workspace_bytes);
    if(max_abs > 0.01f) std::exit(6);
    check_hip(hipFree(dq), "hipFree(q8 q)");
    check_hip(hipFree(dk), "hipFree(q8 k)");
    check_hip(hipFree(dv), "hipFree(q8 v)");
    check_hip(hipFree(dout), "hipFree(q8 out)");
    check_hip(hipFree(workspace), "hipFree(q8 workspace)");
}

} // namespace

int main()
{
    if(hipfire_flash_attn_ck_abi_version() != HIPFIRE_FLASH_ATTN_CK_ABI_VERSION)
    {
        std::fprintf(stderr, "sidecar ABI mismatch\n");
        return 1;
    }
    run_case("gqa-noncausal", 4, 2, false, false);
    run_case("gqa-causal", 4, 2, true, true);
    run_case("mha-noncausal", 2, 2, false, false);
    run_case("mqa-noncausal", 4, 1, false, false);
    run_q8_d256_case();
    return 0;
}
