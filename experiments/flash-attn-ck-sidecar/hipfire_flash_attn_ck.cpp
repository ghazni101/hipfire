// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include "fmha_fwd.hpp"
#include "mask.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

static_assert(sizeof(hipfire_flash_attn_ck_fwd_params) == 224,
              "FlashAttention CK ABI parameter layout changed");
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, q) == 8);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, workspace) == 40);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, dtype) == 64);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, softmax_scale) == 104);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, stride_q) == 112);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, batch_stride_out) == 200);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, packed_k_row_stride_bytes) == 208);
static_assert(sizeof(hipfire_flash_attn_ck_capability) == 32);

namespace {

constexpr size_t kWorkspaceAlignment = 256;

size_t align_up(size_t value)
{
    return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

bool is_q8_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return p->dtype == HIPFIRE_FLASH_ATTN_CK_F32 &&
           p->k_format == HIPFIRE_FLASH_ATTN_CK_Q8 &&
           p->v_format == HIPFIRE_FLASH_ATTN_CK_Q8;
}

size_t q8_workspace_bytes(const hipfire_flash_attn_ck_fwd_params* p)
{
    const size_t q = static_cast<size_t>(p->batch) * p->seqlen_q * p->nhead_q * p->head_dim;
    const size_t kv = static_cast<size_t>(p->batch) * p->seqlen_k * p->nhead_k * p->head_dim;
    return align_up(q * sizeof(__half)) + align_up(kv * sizeof(__half)) * 2 +
           align_up(q * sizeof(__half));
}

__global__ void convert_f32_to_f16(const float* input, __half* output, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count) output[index] = __float2half_rn(input[index]);
}

__global__ void convert_f16_to_f32(const __half* input, float* output, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count) output[index] = __half2float(input[index]);
}

__global__ void decode_q8_kv_d256(const uint8_t* packed_k,
                                   const uint8_t* packed_v,
                                   __half* dense_k,
                                   __half* dense_v,
                                   int rows,
                                   int kv_heads,
                                   int64_t k_row_stride_bytes,
                                   int64_t v_row_stride_bytes)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= kv_heads || lane >= 32) return;

    const int block = lane >> 2;
    const int lane_in_block = lane & 3;
    const uint8_t* k_block = packed_k + static_cast<size_t>(row) * k_row_stride_bytes +
                             head * 272 + block * 34;
    const uint8_t* v_block = packed_v + static_cast<size_t>(row) * v_row_stride_bytes +
                             head * 272 + block * 34;
    const float k_scale = __half2float(*reinterpret_cast<const __half*>(k_block));
    const float v_scale = __half2float(*reinterpret_cast<const __half*>(v_block));
    const int8_t* k_values = reinterpret_cast<const int8_t*>(k_block + 2) + lane_in_block * 8;
    const int8_t* v_values = reinterpret_cast<const int8_t*>(v_block + 2) + lane_in_block * 8;
    const size_t output = (static_cast<size_t>(row) * kv_heads + head) * 256 + lane * 8;
#pragma unroll
    for(int index = 0; index < 8; ++index)
    {
        dense_k[output + index] = __float2half_rn(k_scale * static_cast<float>(k_values[index]));
        dense_v[output + index] = __float2half_rn(v_scale * static_cast<float>(v_values[index]));
    }
}

void set_error(char* error, size_t capacity, const std::string& message)
{
    if(error == nullptr || capacity == 0)
    {
        return;
    }
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

int validate(const hipfire_flash_attn_ck_fwd_params* p, char* error, size_t error_capacity)
{
    if(p == nullptr)
    {
        set_error(error, error_capacity, "params is null");
        return 1;
    }
    if(p->abi_version != HIPFIRE_FLASH_ATTN_CK_ABI_VERSION)
    {
        set_error(error, error_capacity, "unsupported ABI version");
        return 1;
    }
    if(p->struct_size < sizeof(hipfire_flash_attn_ck_fwd_params))
    {
        set_error(error, error_capacity, "parameter struct is too small");
        return 1;
    }
    if(p->q == nullptr || p->k == nullptr || p->v == nullptr || p->out == nullptr)
    {
        set_error(error, error_capacity, "q, k, v, and out must be non-null");
        return 1;
    }
    const bool dense = p->dtype == HIPFIRE_FLASH_ATTN_CK_F16 &&
                       p->k_format == HIPFIRE_FLASH_ATTN_CK_DENSE_F16 &&
                       p->v_format == HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    const bool q8 = is_q8_cell(p);
    if(!dense && !q8)
    {
        set_error(error, error_capacity, "unsupported dtype and K/V format cell");
        return 1;
    }
    if(p->workspace_bytes > 0 && p->workspace == nullptr)
    {
        set_error(error, error_capacity, "workspace must be non-null when workspace_bytes is non-zero");
        return 1;
    }
    if(p->batch <= 0 || p->seqlen_q <= 0 || p->seqlen_k <= 0 ||
       p->nhead_q <= 0 || p->nhead_k <= 0)
    {
        set_error(error, error_capacity, "batch, sequence lengths, and head counts must be positive");
        return 1;
    }
    if((dense && p->head_dim != 64) || (q8 && p->head_dim != 256))
    {
        set_error(error, error_capacity, "unsupported head dimension for selected cell");
        return 1;
    }
    if(p->nhead_q % p->nhead_k != 0)
    {
        set_error(error, error_capacity, "nhead_k must divide nhead_q");
        return 1;
    }
    if(p->causal != 0 && p->causal != 1)
    {
        set_error(error, error_capacity, "causal must be 0 or 1");
        return 1;
    }
    if(!(p->softmax_scale > 0.0f))
    {
        set_error(error, error_capacity, "softmax_scale must be positive");
        return 1;
    }
    const int64_t strides[] = {
        p->stride_q,
        p->stride_k,
        p->stride_v,
        p->stride_out,
        p->nhead_stride_q,
        p->nhead_stride_k,
        p->nhead_stride_v,
        p->nhead_stride_out,
        p->batch_stride_q,
        p->batch_stride_k,
        p->batch_stride_v,
        p->batch_stride_out,
    };
    for(const int64_t stride : strides)
    {
        if(stride <= 0 || stride > INT32_MAX)
        {
            set_error(error, error_capacity, "all element strides must be in (0, INT32_MAX]");
            return 1;
        }
    }
    if(q8)
    {
        const int64_t minimum_row = static_cast<int64_t>(p->nhead_k) * 272;
        if(p->batch != 1 || p->causal != 1 ||
           p->packed_k_row_stride_bytes < minimum_row ||
           p->packed_v_row_stride_bytes < minimum_row)
        {
            set_error(error, error_capacity, "Q8 D256 requires batch=1, causal, and valid packed row strides");
            return 1;
        }
        if(p->workspace == nullptr || p->workspace_bytes < q8_workspace_bytes(p))
        {
            set_error(error, error_capacity, "caller workspace is too small for Q8 staging");
            return 1;
        }
    }
    set_error(error, error_capacity, "");
    return 0;
}

} // namespace

extern "C" uint32_t hipfire_flash_attn_ck_abi_version(void)
{
    return HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
}

extern "C" size_t hipfire_flash_attn_ck_capabilities(
    hipfire_flash_attn_ck_capability* capabilities,
    size_t capacity)
{
    static const hipfire_flash_attn_ck_capability cells[] = {
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#elif defined(HIPFIRE_CK_TARGET_GFX1151)
        HIPFIRE_FLASH_ATTN_CK_GFX1151,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F16,
        HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
        HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
        64,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
#if defined(HIPFIRE_CK_TARGET_GFX1100)
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
#endif
    };
    constexpr size_t count = sizeof(cells) / sizeof(cells[0]);
    if(capabilities != nullptr && capacity > 0)
    {
        const size_t written = std::min(capacity, count);
        std::memcpy(capabilities, cells, written * sizeof(cells[0]));
        return written;
    }
    return count;
}

extern "C" size_t hipfire_flash_attn_ck_fwd_workspace_bytes(
    const hipfire_flash_attn_ck_fwd_params* params)
{
    return params != nullptr && is_q8_cell(params) ? q8_workspace_bytes(params) : 0;
}

extern "C" int hipfire_flash_attn_ck_fwd_supported(
    const hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity)
{
    return validate(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_fwd(
    const hipfire_flash_attn_ck_fwd_params* p,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate(p, error, error_capacity); status != 0)
    {
        return status;
    }

    try
    {
        const bool q8 = is_q8_cell(p);
        const void* q_ptr = p->q;
        const void* k_ptr = p->k;
        const void* v_ptr = p->v;
        void* out_ptr = p->out;
        int64_t stride_q = p->stride_q;
        int64_t stride_k = p->stride_k;
        int64_t stride_v = p->stride_v;
        int64_t stride_out = p->stride_out;
        int64_t nhead_stride_q = p->nhead_stride_q;
        int64_t nhead_stride_k = p->nhead_stride_k;
        int64_t nhead_stride_v = p->nhead_stride_v;
        int64_t nhead_stride_out = p->nhead_stride_out;
        int64_t batch_stride_q = p->batch_stride_q;
        int64_t batch_stride_k = p->batch_stride_k;
        int64_t batch_stride_v = p->batch_stride_v;
        int64_t batch_stride_out = p->batch_stride_out;
        __half* staged_out = nullptr;
        hipStream_t stream = reinterpret_cast<hipStream_t>(p->stream);
        const size_t q_count = static_cast<size_t>(p->batch) * p->seqlen_q * p->nhead_q * p->head_dim;
        if(q8)
        {
            uint8_t* cursor = static_cast<uint8_t*>(p->workspace);
            __half* staged_q = reinterpret_cast<__half*>(cursor);
            cursor += align_up(q_count * sizeof(__half));
            const size_t kv_count = static_cast<size_t>(p->seqlen_k) * p->nhead_k * p->head_dim;
            __half* staged_k = reinterpret_cast<__half*>(cursor);
            cursor += align_up(kv_count * sizeof(__half));
            __half* staged_v = reinterpret_cast<__half*>(cursor);
            cursor += align_up(kv_count * sizeof(__half));
            staged_out = reinterpret_cast<__half*>(cursor);

            const int threads = 256;
            convert_f32_to_f16<<<(q_count + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<const float*>(p->q), staged_q, q_count);
            decode_q8_kv_d256<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                static_cast<const uint8_t*>(p->k), static_cast<const uint8_t*>(p->v),
                staged_k, staged_v, p->seqlen_k, p->nhead_k,
                p->packed_k_row_stride_bytes, p->packed_v_row_stride_bytes);
            q_ptr = staged_q;
            k_ptr = staged_k;
            v_ptr = staged_v;
            out_ptr = staged_out;
            stride_q = p->nhead_q * p->head_dim;
            stride_k = p->nhead_k * p->head_dim;
            stride_v = p->nhead_k * p->head_dim;
            stride_out = p->nhead_q * p->head_dim;
            nhead_stride_q = nhead_stride_k = nhead_stride_v = nhead_stride_out = p->head_dim;
            batch_stride_q = p->seqlen_q * stride_q;
            batch_stride_k = p->seqlen_k * stride_k;
            batch_stride_v = p->seqlen_k * stride_v;
            batch_stride_out = p->seqlen_q * stride_out;
        }
        const std::string dtype = "fp16";
        const std::string mask_id = p->causal != 0 ? "b:-1,0" : "0";
        const mask_info mask = mask_info::decode(mask_id, p->seqlen_q, p->seqlen_k);

        fmha_fwd_traits traits{
            p->head_dim,
            p->head_dim,
            dtype,
            false,
            true,
            false,
            mask.type,
            bias_enum::no_bias,
            false,
            false,
            quant_scale_enum::no_scale,
            false,
        };

        fmha_fwd_args args{};
        args.q_ptr = q_ptr;
        args.k_ptr = k_ptr;
        args.v_ptr = v_ptr;
        args.o_ptr = out_ptr;
        args.seqlen_q = p->seqlen_q;
        args.seqlen_k = p->seqlen_k;
        args.batch = p->batch;
        args.max_seqlen_q = p->seqlen_q;
        args.hdim_q = p->head_dim;
        args.hdim_v = p->head_dim;
        args.nhead_q = p->nhead_q;
        args.nhead_k = p->nhead_k;
        args.scale_s = p->softmax_scale;
        args.logits_soft_cap = 0.0f;
        args.stride_q = static_cast<ck_tile::index_t>(stride_q);
        args.stride_k = static_cast<ck_tile::index_t>(stride_k);
        args.stride_v = static_cast<ck_tile::index_t>(stride_v);
        args.stride_o = static_cast<ck_tile::index_t>(stride_out);
        args.nhead_stride_q = static_cast<ck_tile::index_t>(nhead_stride_q);
        args.nhead_stride_k = static_cast<ck_tile::index_t>(nhead_stride_k);
        args.nhead_stride_v = static_cast<ck_tile::index_t>(nhead_stride_v);
        args.nhead_stride_o = static_cast<ck_tile::index_t>(nhead_stride_out);
        args.batch_stride_q = static_cast<ck_tile::index_t>(batch_stride_q);
        args.batch_stride_k = static_cast<ck_tile::index_t>(batch_stride_k);
        args.batch_stride_v = static_cast<ck_tile::index_t>(batch_stride_v);
        args.batch_stride_o = static_cast<ck_tile::index_t>(batch_stride_out);
        args.window_size_left = -1;
        args.window_size_right = p->causal != 0 ? 0 : -1;
        args.mask_type = static_cast<ck_tile::index_t>(mask.type);
        args.min_seqlen_q = 0;
        args.p_drop = 0.0f;
        args.s_randval = false;
        args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

        ck_tile::stream_config stream_config{
            reinterpret_cast<hipStream_t>(p->stream),
        };
        const float result = fmha_fwd(traits, args, stream_config);
        if(result < 0.0f)
        {
            set_error(error, error_capacity, "CK found no matching forward kernel");
            return 2;
        }
        if(q8)
        {
            const int threads = 256;
            convert_f16_to_f32<<<(q_count + threads - 1) / threads, threads, 0, stream>>>(
                staged_out, static_cast<float*>(p->out), q_count);
        }
        set_error(error, error_capacity, "");
        return 0;
    }
    catch(const std::exception& exception)
    {
        set_error(error, error_capacity, exception.what());
        return 3;
    }
    catch(...)
    {
        set_error(error, error_capacity, "unknown C++ exception");
        return 3;
    }
}
