#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

#ifdef __CUDACC__
#ifndef _HOST_FALLBACKS_H_
#define _HOST_FALLBACKS_H_
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#endif
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tensor.h"
#include "model_config.h"
#include "tokenizer.h"
#include "gguf_types_cuda.h"
#include "gguf_reader_cuda.h"
#include "gguf_loader_cuda.h"

// Matemaatikateegid
#include "tensor_math_quantized_cuda.cuh"
#include "wmma_tensor.cuh"
#include "gguf_cuda_tensor/tensor_math_quantized_wmma.cuh"
#include "flash_attention_batched.cuh"

struct RuntimeConfig {
    float temperature = 0.7f; float top_p = 0.9f; float rep_penalty = 1.18f;
    int min_tokens = 50; int max_tokens = 300; int penalty_window = 64;
};

RuntimeConfig load_runtime_config(const std::string& filepath) {
    RuntimeConfig cfg; std::ifstream file(filepath);
    if (!file.is_open()) return cfg;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string key, assign;
        if (ss >> key >> assign && assign == "=") {
            if (key == "temperature") ss >> cfg.temperature;
            else if (key == "top_p") ss >> cfg.top_p;
            else if (key == "rep_penalty") ss >> cfg.rep_penalty;
            else if (key == "min_tokens") ss >> cfg.min_tokens;
            else if (key == "max_tokens") ss >> cfg.max_tokens;
            else if (key == "penalty_window") ss >> cfg.penalty_window;
        }
    }
    return cfg;
}

std::string load_system_prompt(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[FATAL] runtime_config.json ei leitud teelt: " << filepath << "\n";
        std::exit(1);
    }
    std::stringstream buffer; buffer << file.rdbuf();
    std::string content = buffer.str();
    size_t key_pos = content.find("\"system_prompt\"");
    if (key_pos == std::string::npos) {
        std::cerr << "[FATAL] \"system_prompt\" votit ei leitud runtime_config.json seest\n";
        std::exit(1);
    }
    size_t colon_pos = content.find(":", key_pos);
    size_t start_quote = content.find("\"", colon_pos);
    size_t end_quote = content.find("\"", start_quote + 1);
    if (start_quote == std::string::npos || end_quote == std::string::npos) {
        std::cerr << "[FATAL] \"system_prompt\" vaartust ei onnestunud parsida\n";
        std::exit(1);
    }
    return content.substr(start_quote + 1, end_quote - start_quote - 1);
}

// =============================================================================
// N=1 KERNELID (Decode faas / CUDA Graphs Ankur)
// =============================================================================
__global__ void fetch_embedding_kernel(float* __restrict__ x, const float* __restrict__ emb_table, const int* __restrict__ d_token_id, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) x[idx] = emb_table[(*d_token_id) * dim + idx];
}

// 🔥 L2 CACHE LUKUSTUS: PTX st.cg väldib VRAM-i reostamist
__global__ void rmsnorm_kernel_ptx_l2_locked(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x; int block_size = blockDim.x; int n4 = n / 4;
    float partial = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < n4; i += block_size) {
        float4 v = x4[i]; partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    sdata[tid] = partial; __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sdata[0] = rsqrtf((sdata[0] / static_cast<float>(n)) + eps);
    __syncthreads();
    float inv_rms = sdata[0];
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n4; i += block_size) {
        float4 v = x4[i]; float4 w = w4[i];
        float4 res;
        res.x = v.x * inv_rms * w.x; res.y = v.y * inv_rms * w.y;
        res.z = v.z * inv_rms * w.z; res.w = v.w * inv_rms * w.w;
        asm volatile ("st.cg.global.v4.f32 [%0], {%1, %2, %3, %4};"
                      : : "l"(&out4[i]), "f"(res.x), "f"(res.y), "f"(res.z), "f"(res.w) : "memory");
    }
}

// 🔥 L2 CACHE LUKUSTUS (Residual + Norm)
__global__ void add_and_rmsnorm_kernel_ptx_l2_locked(float* __restrict__ x, const float* __restrict__ residual, const float* __restrict__ weight, float* __restrict__ out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x; int block_size = blockDim.x; int n4 = n / 4;
    float partial = 0.0f;
    float4* x4 = reinterpret_cast<float4*>(x);
    const float4* res4 = reinterpret_cast<const float4*>(residual);
    for (int i = tid; i < n4; i += block_size) {
        float4 vx = x4[i]; float4 vr = res4[i];
        vx.x += vr.x; vx.y += vr.y; vx.z += vr.z; vx.w += vr.w; x4[i] = vx;
        partial += vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w;
    }
    sdata[tid] = partial; __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sdata[0] = rsqrtf((sdata[0] / static_cast<float>(n)) + eps);
    __syncthreads();
    float inv_rms = sdata[0];
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n4; i += block_size) {
        float4 vx = x4[i]; float4 w = w4[i];
        float4 res;
        res.x = vx.x * inv_rms * w.x; res.y = vx.y * inv_rms * w.y;
        res.z = vx.z * inv_rms * w.z; res.w = vx.w * inv_rms * w.w;
        asm volatile ("st.cg.global.v4.f32 [%0], {%1, %2, %3, %4};"
                      : : "l"(&out4[i]), "f"(res.x), "f"(res.y), "f"(res.z), "f"(res.w) : "memory");
    }
}

__global__ void fused_rope_and_cache_kv_kernel(
    float* __restrict__ q, const float* __restrict__ k_in, const float* __restrict__ v_in,
    half* __restrict__ k_cache_base, half* __restrict__ v_cache_base,
    int total_heads, int num_kv_heads, int head_dim, int kv_dim,
    const int* __restrict__ d_pos, float base)
{
    int pos = *d_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q_pairs = total_heads * half_dim;

    if (idx < total_q_pairs) {
        int h = idx / half_dim; int i = idx % half_dim;
        float freq = 1.0f / powf(base, (2.0f * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(pos) * freq;
        float cos_val = cosf(val), sin_val = sinf(val);

        float q0 = q[h * head_dim + 2 * i]; float q1 = q[h * head_dim + 2 * i + 1];
        q[h * head_dim + 2 * i] = q0 * cos_val - q1 * sin_val;
        q[h * head_dim + 2 * i + 1] = q0 * sin_val + q1 * cos_val;

        if (idx < num_kv_heads * half_dim) {
            int kv_h = h;
            float x0 = k_in[kv_h * head_dim + 2 * i]; float x1 = k_in[kv_h * head_dim + 2 * i + 1];
            float k_rope0 = x0 * cos_val - x1 * sin_val; float k_rope1 = x0 * sin_val + x1 * cos_val;
            int cache_offset = pos * kv_dim + kv_h * head_dim;
            reinterpret_cast<half2*>(k_cache_base + cache_offset)[i] = __floats2half2_rn(k_rope0, k_rope1);
            float v0 = v_in[kv_h * head_dim + 2 * i]; float v1 = v_in[kv_h * head_dim + 2 * i + 1];
            reinterpret_cast<half2*>(v_cache_base + cache_offset)[i] = __floats2half2_rn(v0, v1);
        }
    }
}

// 🔥🔥 THE BEAST V3 (Optimized Decode Attention)
__global__ void mha_kv_cache_kernel(const float* __restrict__ Q, const half* __restrict__ K_cache_base, const half* __restrict__ V_cache_base, float* __restrict__ out, int num_heads, int head_dim, int kv_dim, int heads_per_kv_group, const int* __restrict__ d_pos, float scale) {
    int pos = *d_pos; int current_seq_len = pos + 1;
    int h = blockIdx.x; if (h >= num_heads) return;
    int kv_head = h / heads_per_kv_group;
    const float* q_ptr = Q + h * head_dim;
    extern __shared__ float smem[]; float* scores = smem;

    int tid = threadIdx.x; int warp_id = tid / 32; int lane_id = tid % 32;
    int warps_per_block = blockDim.x / 32;

    float q0 = 0.0f, q1 = 0.0f;
    if (lane_id < 32) { q0 = q_ptr[lane_id * 2]; q1 = q_ptr[lane_id * 2 + 1]; }

    int t = warp_id;
    for (; t <= current_seq_len - (warps_per_block * 2); t += warps_per_block * 2) {
        const uint32_t* k_ptr1 = reinterpret_cast<const uint32_t*>(K_cache_base + t * kv_dim + kv_head * head_dim);
        const uint32_t* k_ptr2 = reinterpret_cast<const uint32_t*>(K_cache_base + (t + warps_per_block) * kv_dim + kv_head * head_dim);

        half2 k_h2_1 = *reinterpret_cast<const half2*>(&k_ptr1[lane_id]);
        half2 k_h2_2 = *reinterpret_cast<const half2*>(&k_ptr2[lane_id]);

        float2 k_vec1 = __half22float2(k_h2_1); float2 k_vec2 = __half22float2(k_h2_2);
        float dot1 = q0 * k_vec1.x + q1 * k_vec1.y; float dot2 = q0 * k_vec2.x + q1 * k_vec2.y;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot1 += __shfl_down_sync(0xffffffff, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffff, dot2, offset);
        }
        if (lane_id == 0) { scores[t] = dot1 * scale; scores[t + warps_per_block] = dot2 * scale; }
    }
    for (; t < current_seq_len; t += warps_per_block) {
        const uint32_t* k_ptr = reinterpret_cast<const uint32_t*>(K_cache_base + t * kv_dim + kv_head * head_dim);
        half2 k_h2 = *reinterpret_cast<const half2*>(&k_ptr[lane_id]);
        float2 k_vec = __half22float2(k_h2);
        float dot = q0 * k_vec.x + q1 * k_vec.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane_id == 0) scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = -1e30f; float local_sum = 0.0f;
    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        float s = scores[i];
        if (s > local_max) {
            float e = expf(local_max - s);
            local_sum = local_sum * e + 1.0f; local_max = s;
        } else { local_sum += expf(s - local_max); }
    }

    __shared__ float s_max[32]; __shared__ float s_sum[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float m_j = __shfl_down_sync(0xffffffff, local_max, offset);
        float l_j = __shfl_down_sync(0xffffffff, local_sum, offset);
        if (local_max > m_j) { local_sum = local_sum + l_j * expf(m_j - local_max); }
        else { local_sum = local_sum * expf(local_max - m_j) + l_j; local_max = m_j; }
    }
    if (lane_id == 0) { s_max[warp_id] = local_max; s_sum[warp_id] = local_sum; }
    __syncthreads();

    if (tid < 32) {
        float m_j = (tid < warps_per_block) ? s_max[tid] : -1e30f;
        float l_j = (tid < warps_per_block) ? s_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float m_shfl = __shfl_down_sync(0xffffffff, m_j, offset);
            float l_shfl = __shfl_down_sync(0xffffffff, l_j, offset);
            if (m_j > m_shfl) { l_j = l_j + l_shfl * expf(m_shfl - m_j); }
            else { l_j = l_j * expf(m_j - m_shfl) + l_shfl; m_j = m_shfl; }
        }
        if (tid == 0) { s_max[0] = m_j; s_sum[0] = l_j; }
    }
    __syncthreads();

    float global_max = s_max[0]; float global_sum = s_sum[0];
    for (int i = tid; i < current_seq_len; i += blockDim.x) scores[i] = expf(scores[i] - global_max) / global_sum;
    __syncthreads();

    float sum_v0 = 0.0f; float sum_v1 = 0.0f;
    t = warp_id;
    for (; t <= current_seq_len - (warps_per_block * 2); t += warps_per_block * 2) {
        float s1 = scores[t]; float s2 = scores[t + warps_per_block];
        const uint32_t* v_ptr1 = reinterpret_cast<const uint32_t*>(V_cache_base + t * kv_dim + kv_head * head_dim);
        const uint32_t* v_ptr2 = reinterpret_cast<const uint32_t*>(V_cache_base + (t + warps_per_block) * kv_dim + kv_head * head_dim);

        float2 v_vec1 = __half22float2(*reinterpret_cast<const half2*>(&v_ptr1[lane_id]));
        float2 v_vec2 = __half22float2(*reinterpret_cast<const half2*>(&v_ptr2[lane_id]));

        sum_v0 += s1 * v_vec1.x + s2 * v_vec2.x;
        sum_v1 += s1 * v_vec1.y + s2 * v_vec2.y;
    }
    for (; t < current_seq_len; t += warps_per_block) {
        float s = scores[t];
        const uint32_t* v_ptr = reinterpret_cast<const uint32_t*>(V_cache_base + t * kv_dim + kv_head * head_dim);
        float2 v_vec = __half22float2(*reinterpret_cast<const half2*>(&v_ptr[lane_id]));
        sum_v0 += s * v_vec.x; sum_v1 += s * v_vec.y;
    }

    float* v_reduce = smem;
    v_reduce[warp_id * 64 + lane_id * 2] = sum_v0;
    v_reduce[warp_id * 64 + lane_id * 2 + 1] = sum_v1;
    __syncthreads();

    if (tid < head_dim) {
        float final_v = 0.0f;
        #pragma unroll
        for (int w = 0; w < warps_per_block; ++w) final_v += v_reduce[w * 64 + tid];
        out[h * head_dim + tid] = final_v;
    }
}

__global__ void gemv_f32_kernel_warp_float4(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ out, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32; int row = warp_id;
    if (row >= M) return;
    float local_sum = 0.0f;
    const float4* w_row_4 = reinterpret_cast<const float4*>(W + row * K);
    const float4* x_4 = reinterpret_cast<const float4*>(x);
    int K4 = K / 4;
    for (int k = lane_id; k < K4; k += 32) {
        float4 w_vec = w_row_4[k]; float4 x_vec = x_4[k];
        local_sum += w_vec.x * x_vec.x + w_vec.y * x_vec.y + w_vec.z * x_vec.z + w_vec.w * x_vec.w;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) out[row] = local_sum;
}


// =============================================================================
// BATCHED KERNELID (Prefill faas)
// =============================================================================
__global__ void fetch_embedding_batched_kernel(float* __restrict__ x, const float* __restrict__ emb_table, const int* __restrict__ d_tokens, int dim, int N) {
    int token_idx = blockIdx.y; int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < N && dim_idx < dim) { int token_id = d_tokens[token_idx]; x[token_idx * dim + dim_idx] = emb_table[token_id * dim + dim_idx]; }
}
__global__ void rmsnorm_batched_kernel_float4(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int dim, int N, float eps) {
    int token_idx = blockIdx.y; if (token_idx >= N) return; extern __shared__ float sdata[];
    int tid = threadIdx.x; int block_size = blockDim.x; int dim4 = dim / 4; float partial = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(x + token_idx * dim);
    for (int i = tid; i < dim4; i += block_size) { float4 v = x4[i]; partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
    sdata[tid] = partial; __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) { if (tid < stride) sdata[tid] += sdata[tid + stride]; __syncthreads(); }
    if (tid == 0) sdata[0] = rsqrtf((sdata[0] / static_cast<float>(dim)) + eps); __syncthreads();
    float inv_rms = sdata[0]; const float4* w4 = reinterpret_cast<const float4*>(weight); float4* out4 = reinterpret_cast<float4*>(out + token_idx * dim);
    for (int i = tid; i < dim4; i += block_size) { float4 v = x4[i]; float4 w = w4[i]; float4 res; res.x = v.x * inv_rms * w.x; res.y = v.y * inv_rms * w.y; res.z = v.z * inv_rms * w.z; res.w = v.w * inv_rms * w.w; out4[i] = res; }
}
__global__ void add_and_rmsnorm_batched_kernel_float4(float* __restrict__ x, const float* __restrict__ residual_in, const float* __restrict__ weight, float* __restrict__ out, int dim, int N, float eps) {
    int token_idx = blockIdx.y; if (token_idx >= N) return; extern __shared__ float sdata[];
    int tid = threadIdx.x; int block_size = blockDim.x; int dim4 = dim / 4; float partial = 0.0f;
    float4* x4 = reinterpret_cast<float4*>(x + token_idx * dim); const float4* res4 = reinterpret_cast<const float4*>(residual_in + token_idx * dim);
    for (int i = tid; i < dim4; i += block_size) { float4 vx = x4[i]; float4 vr = res4[i]; vx.x += vr.x; vx.y += vr.y; vx.z += vr.z; vx.w += vr.w; x4[i] = vx; partial += vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w; }
    sdata[tid] = partial; __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) { if (tid < stride) sdata[tid] += sdata[tid + stride]; __syncthreads(); }
    if (tid == 0) sdata[0] = rsqrtf((sdata[0] / static_cast<float>(dim)) + eps); __syncthreads();
    float inv_rms = sdata[0]; const float4* w4 = reinterpret_cast<const float4*>(weight); float4* out4 = reinterpret_cast<float4*>(out + token_idx * dim);
    for (int i = tid; i < dim4; i += block_size) { float4 vx = x4[i]; float4 w = w4[i]; float4 res; res.x = vx.x * inv_rms * w.x; res.y = vx.y * inv_rms * w.y; res.z = vx.z * inv_rms * w.z; res.w = vx.w * inv_rms * w.w; out4[i] = res; }
}
__global__ void rope_q_batched_kernel(float* q, int num_heads, int head_dim, int N, float base) {
    int token_idx = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x; int half_dim = head_dim / 2;
    if (token_idx < N && idx < num_heads * half_dim) {
        int h = idx / half_dim; int i = idx % half_dim; float freq = 1.0f / powf(base, (2.0f * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(token_idx) * freq; float cos_val = cosf(val), sin_val = sinf(val);
        float* p = q + token_idx * (num_heads * head_dim) + h * head_dim; float x0 = p[2 * i], x1 = p[2 * i + 1];
        p[2 * i] = x0 * cos_val - x1 * sin_val; p[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}
__global__ void rope_k_inplace_batched_kernel(float* k, int kv_heads, int head_dim, int kv_dim, int N, float base) {
    int token_idx = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x; int half_dim = head_dim / 2;
    if (token_idx < N && idx < kv_heads * half_dim) {
        int h = idx / half_dim; int i = idx % half_dim; float freq = 1.0f / powf(base, (2.0f * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(token_idx) * freq; float cos_val = cosf(val), sin_val = sinf(val);
        int offset = token_idx * kv_dim + h * head_dim; float x0 = k[offset + 2 * i]; float x1 = k[offset + 2 * i + 1];
        k[offset + 2 * i] = x0 * cos_val - x1 * sin_val; k[offset + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}
__global__ void float2half_copy_batched_kernel(const float* __restrict__ in, half* __restrict__ out, int stride, int N) {
    int token_idx = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < N && idx < stride / 2) {
        float v0 = in[token_idx * stride + 2 * idx]; float v1 = in[token_idx * stride + 2 * idx + 1];
        reinterpret_cast<half2*>(out + token_idx * stride)[idx] = __floats2half2_rn(v0, v1);
    }
}
__global__ void swiglu_batched_kernel(const float* w1_out, const float* w3_out, float* final_out, int hidden_dim, int N) {
    int token_idx = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < N && idx < hidden_dim) { int offset = token_idx * hidden_dim + idx; float v1 = w1_out[offset]; float v3 = w3_out[offset]; final_out[offset] = (v1 / (1.0f + expf(-v1))) * v3; }
}
__global__ void vector_add_batched_kernel_float4(float* a, const float* b, int dim, int N) {
    int token_idx = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x; int dim4 = dim / 4;
    if (token_idx < N && idx < dim4) { int offset = token_idx * dim4 + idx; float4 va = reinterpret_cast<float4*>(a)[offset]; float4 vb = reinterpret_cast<const float4*>(b)[offset]; va.x += vb.x; va.y += vb.y; va.z += vb.z; va.w += vb.w; reinterpret_cast<float4*>(a)[offset] = va; }
}
__global__ void float2half_batched_kernel(const float* __restrict__ in, half* __restrict__ out, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < elements) out[idx] = __float2half(in[idx]);
}

// =============================================================================
// SÜSTEEMI PUHVRID
// =============================================================================
struct CUDABuffersQ4 {
    float* x = nullptr; float* xb = nullptr;
    float* q = nullptr; float* k = nullptr; float* v = nullptr;
    float* attn_concat = nullptr; float* attn_out = nullptr;
    float* ffn_norm = nullptr; float* swiglu_out = nullptr; float* w3_out = nullptr;
    float* ffn_final = nullptr; float* final_hidden = nullptr;
    float* logits = nullptr;

    half* xb_fp16 = nullptr;
    half* hidden_fp16 = nullptr;

    std::vector<half*> k_cache; std::vector<half*> v_cache;

    int* d_pos = nullptr; int* d_token_id = nullptr; int* d_tokens_array = nullptr;

    cudaGraph_t graph = nullptr; cudaGraphExec_t instance = nullptr;
    bool graph_created = false; cudaStream_t stream;

    void allocate(const ModelConfig& cfg) {
        int dim = cfg.dim; int hidden_dim = cfg.hidden_dim;
        int num_heads = cfg.num_heads; int num_kv_heads = cfg.num_kv_heads;
        int kv_dim = (dim / num_heads) * num_kv_heads; int max_n = cfg.seq_len;

        cudaMalloc(&x, max_n * dim * sizeof(float)); cudaMalloc(&xb, max_n * dim * sizeof(float));
        cudaMalloc(&xb_fp16, max_n * dim * sizeof(half)); cudaMalloc(&q, max_n * dim * sizeof(float));
        cudaMalloc(&k, max_n * kv_dim * sizeof(float)); cudaMalloc(&v, max_n * kv_dim * sizeof(float));
        cudaMalloc(&attn_concat, max_n * dim * sizeof(float)); cudaMalloc(&attn_out, max_n * dim * sizeof(float));
        cudaMalloc(&ffn_norm, max_n * dim * sizeof(float)); cudaMalloc(&swiglu_out, max_n * hidden_dim * sizeof(float));
        cudaMalloc(&w3_out, max_n * hidden_dim * sizeof(float)); cudaMalloc(&hidden_fp16, max_n * hidden_dim * sizeof(half));
        cudaMalloc(&ffn_final, max_n * dim * sizeof(float)); cudaMalloc(&final_hidden, max_n * dim * sizeof(float));
        cudaMalloc(&logits, max_n * cfg.vocab_size * sizeof(float));

        k_cache.resize(cfg.num_layers); v_cache.resize(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) {
            cudaMalloc(&k_cache[l], cfg.seq_len * kv_dim * sizeof(half));
            cudaMalloc(&v_cache[l], cfg.seq_len * kv_dim * sizeof(half));
        }
        cudaMalloc(&d_pos, sizeof(int)); cudaMalloc(&d_token_id, sizeof(int));
        cudaMalloc(&d_tokens_array, max_n * sizeof(int)); stream = cudaStreamPerThread;
    }

    void reset_kv_cache(const ModelConfig& cfg) {
        int kv_dim = (cfg.dim / cfg.num_heads) * cfg.num_kv_heads;
        for (int l = 0; l < cfg.num_layers; ++l) {
            cudaMemset(k_cache[l], 0, cfg.seq_len * kv_dim * sizeof(half));
            cudaMemset(v_cache[l], 0, cfg.seq_len * kv_dim * sizeof(half));
        }
    }
};

// =============================================================================
// TACTICAL ENGINE 1: CUDA GRAPHS (Decode) - [L2 CACHE HACK AKTIVEERITUD - V1 KULDSTANDARD]
// =============================================================================
void forward_pass_decode_cuda(const CUDATransformerModelQ4& model, CUDABuffersQ4& bufs) {
    const auto& cfg = model.config;
    int dim = cfg.dim; int hidden_dim = cfg.hidden_dim;
    int num_heads = cfg.num_heads; int num_kv_heads = cfg.num_kv_heads;
    int head_dim = dim / num_heads; int kv_dim = head_dim * num_kv_heads;
    int block_256 = 256; size_t norm_smem = 256 * sizeof(float);

    fetch_embedding_kernel<<<(dim + 255) / 256, 256, 0, bufs.stream>>>(bufs.x, model.token_embedding_table, bufs.d_token_id, dim);

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];

        // 1. PTX RMSNorm (Kirjutab bufs.xb L2 cache'i)
        rmsnorm_kernel_ptx_l2_locked<<<1, 256, norm_smem, bufs.stream>>>(bufs.x, lw.attention_norm, bufs.xb, dim, 1e-5f);

        // 2. QKV GEMV (Loeb bufs.xb otse L2-st, kasutab originaalset optimeeritud teeki!)
        tensor_math_quantized_cuda::launch_gemv_qkv_fused(lw.wq, lw.wk, lw.wv, bufs.xb, bufs.q, bufs.k, bufs.v, dim, kv_dim, bufs.stream);

        // 3. RoPE & MHA (The Beast)
        fused_rope_and_cache_kv_kernel<<<(num_heads * (head_dim / 2) + block_256 - 1) / block_256, block_256, 0, bufs.stream>>>(
            bufs.q, bufs.k, bufs.v, bufs.k_cache[l], bufs.v_cache[l], num_heads, num_kv_heads, head_dim, kv_dim, bufs.d_pos, 10000.0f
        );

        mha_kv_cache_kernel<<<num_heads, 256, cfg.seq_len * sizeof(float), bufs.stream>>>(bufs.q, bufs.k_cache[l], bufs.v_cache[l], bufs.attn_concat, num_heads, head_dim, kv_dim, num_heads / num_kv_heads, bufs.d_pos, 1.0f / sqrtf(static_cast<float>(head_dim)));

        // 4. WO GEMV
        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.wo, bufs.attn_concat, bufs.attn_out, dim, dim);

        // 5. PTX Residual+RMSNorm (Lukustatud L2 cache'i!)
        add_and_rmsnorm_kernel_ptx_l2_locked<<<1, 256, norm_smem, bufs.stream>>>(bufs.x, bufs.attn_out, lw.ffn_norm, bufs.ffn_norm, dim, 1e-5f);

        // 6. W1 & W3 (Loeb L2-st)
        tensor_math_quantized_cuda::launch_gemv_q4_0_w1_w3_swiglu(lw.w1, lw.w3, bufs.ffn_norm, bufs.swiglu_out, hidden_dim, dim);

        // 7. W2 Residual
        tensor_math_quantized_cuda::launch_gemv_q4_0_fused_residual(lw.w2, bufs.swiglu_out, bufs.x, dim, hidden_dim, bufs.stream);
    }

    rmsnorm_kernel_ptx_l2_locked<<<1, 256, norm_smem, bufs.stream>>>(bufs.x, model.final_norm, bufs.final_hidden, dim, 1e-5f);
    gemv_f32_kernel_warp_float4<<< (cfg.vocab_size + 3) / 4, 128, 0, bufs.stream>>>(model.output_weights, bufs.final_hidden, bufs.logits, cfg.vocab_size, dim);
}

// =============================================================================
// TACTICAL ENGINE 2: BATCHED PREFILL (FUSED QKV)
// =============================================================================
void forward_pass_batched_prefill(const CUDATransformerModelQ4& model, const std::vector<int>& tokens, int& pos, CUDABuffersQ4& bufs, float* host_logits) {
    int N = tokens.size(); if (N == 0) return;
    const auto& cfg = model.config;
    int dim = cfg.dim; int hidden_dim = cfg.hidden_dim;
    int kv_dim = (dim / cfg.num_heads) * cfg.num_kv_heads; int head_dim = dim / cfg.num_heads;

    cudaMemcpyAsync(bufs.d_tokens_array, tokens.data(), N * sizeof(int), cudaMemcpyHostToDevice, bufs.stream);

    dim3 grid_emb((dim + 255) / 256, N);
    fetch_embedding_batched_kernel<<<grid_emb, 256, 0, bufs.stream>>>(bufs.x, model.token_embedding_table, bufs.d_tokens_array, dim, N);

    size_t norm_smem = 256 * sizeof(float);
    int tot_dim = N * dim; int tot_hid = N * hidden_dim;

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];
        dim3 grid_norm(1, N);
        rmsnorm_batched_kernel_float4<<<grid_norm, 256, norm_smem, bufs.stream>>>(bufs.x, lw.attention_norm, bufs.xb, dim, N, 1e-5f);
        float2half_batched_kernel<<<(tot_dim + 255) / 256, 256, 0, bufs.stream>>>(bufs.xb, bufs.xb_fp16, tot_dim);

        // 🚀 TOTAALNE FUSIOON: ÜKS LÖÖK!
        tensor_math_quantized_wmma::launch_gemm_qkv_fused_wmma(
            lw.wq, lw.wk, lw.wv, bufs.xb_fp16,
            bufs.q, bufs.k, bufs.v,
            dim, kv_dim, N, dim, bufs.stream
        );

        dim3 grid_rq((cfg.num_heads * (head_dim / 2) + 255) / 256, N);
        rope_q_batched_kernel<<<grid_rq, 256, 0, bufs.stream>>>(bufs.q, cfg.num_heads, head_dim, N, 10000.0f);

        dim3 grid_rk((cfg.num_kv_heads * (head_dim / 2) + 255) / 256, N);
        rope_k_inplace_batched_kernel<<<grid_rk, 256, 0, bufs.stream>>>(bufs.k, cfg.num_kv_heads, head_dim, kv_dim, N, 10000.0f);

        flash_attention_cuda::launch_flash_attention_batched_causal(bufs.q, bufs.k, bufs.v, bufs.attn_out, N, cfg.num_heads, cfg.num_kv_heads, head_dim, bufs.stream);

        dim3 grid_kv((kv_dim / 2 + 255) / 256, N);
        float2half_copy_batched_kernel<<<grid_kv, 256, 0, bufs.stream>>>(bufs.k, bufs.k_cache[l], kv_dim, N);
        float2half_copy_batched_kernel<<<grid_kv, 256, 0, bufs.stream>>>(bufs.v, bufs.v_cache[l], kv_dim, N);

        float2half_batched_kernel<<<(tot_dim + 255) / 256, 256, 0, bufs.stream>>>(bufs.attn_out, bufs.xb_fp16, tot_dim);
        tensor_math_quantized_wmma::launch_gemm_q4_0_wmma(lw.wo, bufs.xb_fp16, bufs.attn_concat, dim, N, dim, bufs.stream);

        add_and_rmsnorm_batched_kernel_float4<<<grid_norm, 256, norm_smem, bufs.stream>>>(bufs.x, bufs.attn_concat, lw.ffn_norm, bufs.ffn_norm, dim, N, 1e-5f);

        float2half_batched_kernel<<<(tot_dim + 255) / 256, 256, 0, bufs.stream>>>(bufs.ffn_norm, bufs.xb_fp16, tot_dim);
        tensor_math_quantized_wmma::launch_gemm_q4_0_wmma(lw.w1, bufs.xb_fp16, bufs.swiglu_out, hidden_dim, N, dim, bufs.stream);
        tensor_math_quantized_wmma::launch_gemm_q4_0_wmma(lw.w3, bufs.xb_fp16, bufs.w3_out, hidden_dim, N, dim, bufs.stream);

        dim3 grid_swiglu((hidden_dim + 255) / 256, N);
        swiglu_batched_kernel<<<grid_swiglu, 256, 0, bufs.stream>>>(bufs.swiglu_out, bufs.w3_out, bufs.swiglu_out, hidden_dim, N);

        float2half_batched_kernel<<<(tot_hid + 255) / 256, 256, 0, bufs.stream>>>(bufs.swiglu_out, bufs.hidden_fp16, tot_hid);
        tensor_math_quantized_wmma::launch_gemm_q4_0_wmma(lw.w2, bufs.hidden_fp16, bufs.ffn_final, dim, N, hidden_dim, bufs.stream);

        dim3 grid_add((dim / 4 + 255) / 256, N);
        vector_add_batched_kernel_float4<<<grid_add, 256, 0, bufs.stream>>>(bufs.x, bufs.ffn_final, dim, N);
    }

    pos += N;

    rmsnorm_kernel_ptx_l2_locked<<<1, 256, norm_smem, bufs.stream>>>(bufs.x + (N - 1) * dim, model.final_norm, bufs.final_hidden, dim, 1e-5f);

    gemv_f32_kernel_warp_float4<<< (cfg.vocab_size + 3) / 4, 128, 0, bufs.stream>>>(model.output_weights, bufs.final_hidden, bufs.logits, cfg.vocab_size, dim);

    cudaStreamSynchronize(bufs.stream);
    cudaMemcpyAsync(host_logits, bufs.logits, cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost, bufs.stream);
    cudaStreamSynchronize(bufs.stream);
}

int sample_token(float* logits, int vocab_size, float temperature, float topp, float rep_penalty, const std::vector<int>& recent_tokens, std::mt19937& rng) {
    if (rep_penalty != 1.0f) {
        for (int token : recent_tokens) {
            if (token >= 0 && token < vocab_size) {
                if (logits[token] < 0) logits[token] *= rep_penalty;
                else logits[token] /= rep_penalty;
            }
        }
    }
    if (temperature == 0.0f) {
        int max_i = 0; float max_v = logits[0];
        #pragma omp parallel for
        for (int i = 1; i < vocab_size; ++i) {
            #pragma omp critical
            {
                if (logits[i] > max_v) { max_v = logits[i]; max_i = i; }
            }
        }
        return max_i;
    }

    float max_val = logits[0] / temperature;
    #pragma omp parallel for
    for (int i = 1; i < vocab_size; ++i) {
        float val = logits[i] / temperature;
        #pragma omp critical
        {
            if (val > max_val) max_val = val;
        }
    }

    float sum = 0.0f; std::vector<std::pair<float, int>> vec; vec.reserve(vocab_size / 10);
    float threshold = 1e-4f;
    for (int i = 0; i < vocab_size; ++i) {
        float prob = std::exp((logits[i] / temperature) - max_val);
        sum += prob;
        if (prob > threshold) vec.push_back({prob, i});
    }
    for (auto& pair : vec) pair.first /= sum;
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (topp < 1.0f) {
        float cumulative_prob = 0.0f; int last_idx = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            cumulative_prob += vec[i].first; last_idx = static_cast<int>(i);
            if (cumulative_prob > topp) break;
        }
        vec.resize(last_idx + 1);
        float p_sum = 0.0f; for (const auto& pair : vec) p_sum += pair.first;
        for (auto& pair : vec) pair.first /= p_sum;
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng); float cdf = 0.0f;
    for (const auto& pair : vec) {
        cdf += pair.first; if (r <= cdf) return pair.second;
    }
    return vec.back().second;
}


int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001); SetConsoleCP(65001);
#endif
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    std::cout << "=== Sovereign Kernel: DUAL ENGINE (V1 GOLD STANDARD + FUSED PREFILL) ===\n";
    try {
        std::string gguf_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
        std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";
        std::string config_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/src_cuda_tensor/runtime_config.txt";
        std::string prompt_config_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/src_cuda_tensor/runtime_config.json";

        ModelConfig config;
        config.dim = 2048; config.hidden_dim = 5632; config.num_layers = 22;
        config.num_heads = 32; config.num_kv_heads = 4;
        config.vocab_size = 32000; config.seq_len = 1024;

        CUDATransformerModelQ4 cuda_model; cuda_model.config = config;
        std::cout << "[INFO] Booting GGUFLoaderCUDA...\n";
        GGUFLoaderCUDA::load_weights_to_vram(gguf_path, cuda_model);

        Tokenizer tokenizer; tokenizer.load(tokenizer_path, config.vocab_size);
        std::cout << "[INFO] Tokenizer initialized.\n";

        CUDABuffersQ4 bufs; bufs.allocate(config);
        std::vector<float> host_logits(config.vocab_size);
        std::random_device rd; std::mt19937 rng(rd());

        std::cout << "\n=======================================================\n";
        std::cout << " V1 N=1 INFERENCE READY (Puhas üks-rida režiim) \n";
        std::cout << "=======================================================\n\n";

        while (true) {
            std::cout << "\n[Darth SulxX] > ";
            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;

            if (user_input == "exit" || user_input == "quit") break;

            // Puhastame võimalikud tühikud/reavahetused servadest
            while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r' || user_input.back() == ' ')) {
                user_input.pop_back();
            }

            if (user_input.empty()) continue;

            RuntimeConfig rcfg = load_runtime_config(config_path);

            std::string sys_prompt = load_system_prompt(prompt_config_path);
            std::string formatted_prompt = "<|system|>\n" + sys_prompt + "</s>\n<|user|>\n" + user_input + "</s>\n<|assistant|>\n";
            std::vector<int> tokens = tokenizer.encode(formatted_prompt);

            std::cout << "\n[Misha Hybrid Output]: " << std::flush;
            bufs.reset_kv_cache(config);
            int pos = 0;

            auto prefill_start = std::chrono::high_resolution_clock::now();
            forward_pass_batched_prefill(cuda_model, tokens, pos, bufs, host_logits.data());
            auto prefill_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> prefill_ms = prefill_end - prefill_start;

            int generated_token_count = 0; double total_forward_ms = 0.0;
            auto gen_start = std::chrono::high_resolution_clock::now();
            auto window_start = gen_start;
            std::vector<int> recent_tokens;

            std::string token_buffer = ""; int buffer_count = 0;
            std::string stop_sequence = "<|user|>";
            std::string rolling_check_buffer = "";

            for (int step = 0; step < rcfg.max_tokens; ++step) {
                int next_token = sample_token(host_logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);
                if (next_token == 2 && generated_token_count < rcfg.min_tokens) {
                    host_logits[2] = -1e9f;
                    next_token = sample_token(host_logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);
                }
                if (next_token == 2 && generated_token_count >= rcfg.min_tokens) break;

                recent_tokens.push_back(next_token);
                if (recent_tokens.size() > static_cast<size_t>(rcfg.penalty_window)) recent_tokens.erase(recent_tokens.begin());

                std::string piece = tokenizer.decode(next_token);
                token_buffer += piece; buffer_count++; generated_token_count++;

                rolling_check_buffer += piece;
                if (rolling_check_buffer.size() > 64) {
                    rolling_check_buffer.erase(0, rolling_check_buffer.size() - 64);
                }
                size_t stop_pos = rolling_check_buffer.find(stop_sequence);
                if (stop_pos != std::string::npos) {
                    size_t trim_amount = rolling_check_buffer.size() - stop_pos;
                    if (token_buffer.size() >= trim_amount) token_buffer.erase(token_buffer.size() - trim_amount);
                    else token_buffer.clear();
                    if (!token_buffer.empty()) { std::cout << token_buffer << std::flush; token_buffer.clear(); }
                    break;
                }

                if (buffer_count >= 8) {
                    std::cout << token_buffer << std::flush;
                    token_buffer = ""; buffer_count = 0;
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                cudaMemcpyAsync(bufs.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, bufs.stream);
                cudaMemcpyAsync(bufs.d_token_id, &next_token, sizeof(int), cudaMemcpyHostToDevice, bufs.stream);

                if (!bufs.graph_created) {
                    cudaStreamBeginCapture(bufs.stream, cudaStreamCaptureModeGlobal);
                    forward_pass_decode_cuda(cuda_model, bufs);
                    cudaStreamEndCapture(bufs.stream, &bufs.graph);
                    cudaGraphInstantiate(&bufs.instance, bufs.graph, NULL, NULL, 0);
                    bufs.graph_created = true;
                }

                cudaGraphLaunch(bufs.instance, bufs.stream);
                cudaMemcpyAsync(host_logits.data(), bufs.logits, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost, bufs.stream);
                cudaStreamSynchronize(bufs.stream);

                auto t1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> fp_time = t1 - t0;
                total_forward_ms += fp_time.count();

                if (generated_token_count % 20 == 0) {
                    auto window_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> window_time = window_end - window_start;
                    double window_ms_per_token = window_time.count() / 20.0;
                    double window_tok_sec = 1000.0 / window_ms_per_token;

                    std::cout << "\n  [Telemetry] Tokens " << (generated_token_count - 19) << "-" << generated_token_count
                              << " | Speed: " << window_tok_sec << " tok/s | Avg: " << window_ms_per_token << " ms/token\n";
                    window_start = std::chrono::high_resolution_clock::now();
                }

                pos++; if (pos >= config.seq_len) break;
            }

            if (!token_buffer.empty()) std::cout << token_buffer << std::flush;

            auto gen_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = gen_end - gen_start;
            float tokens_per_sec = (elapsed.count() > 0.0f) ? (static_cast<float>(generated_token_count) / elapsed.count()) : 0.0f;
            double avg_fp_ms = (generated_token_count > 0) ? (total_forward_ms / generated_token_count) : 0.0;

            std::cout << "\n\n[CUDA Performance Stats] Generated: " << generated_token_count
                      << " tokens | Decode Speed: " << tokens_per_sec << " tok/s | Avg Decode Pass: " << avg_fp_ms << " ms"
                      << " | Prefill Latency: " << prefill_ms.count() << " ms\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL CUDA ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}