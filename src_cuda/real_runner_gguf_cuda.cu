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
#endif
#endif

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "tensor.h"
#include "model_config.h"
#include "tokenizer.h"
#include "gguf_types_cuda.h"
#include "gguf_reader_cuda.h"
#include "gguf_loader_cuda.h"
#include "tensor_math_quantized_cuda.cuh"

struct RuntimeConfig {
    float temperature = 0.7f;
    float top_p = 0.9f;
    float rep_penalty = 1.18f;
    int min_tokens = 50;
    int max_tokens = 300;
    int penalty_window = 64;
};

RuntimeConfig load_runtime_config(const std::string& filepath) {
    RuntimeConfig cfg;
    std::ifstream file(filepath);
    if (!file.is_open()) return cfg;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key, assign;
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
        return "You are Darth Misha, an elite cybernetic intelligence.";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    size_t key_pos = content.find("\"system_prompt\"");
    if (key_pos == std::string::npos) return "You are Darth Misha.";
    size_t colon_pos = content.find(":", key_pos);
    size_t start_quote = content.find("\"", colon_pos);
    size_t end_quote = content.find("\"", start_quote + 1);

    if (start_quote != std::string::npos && end_quote != std::string::npos) {
        return content.substr(start_quote + 1, end_quote - start_quote - 1);
    }
    return "You are Darth Misha.";
}

// UUS: 16-baidine FLOAT4 Vectorized RMSNorm
__global__ void rmsnorm_kernel_float4(const float* __restrict__ x, const float* __restrict__ weight,
                                      float* __restrict__ out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int n4 = n / 4;

    float partial = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int i = tid; i < n4; i += block_size) {
        float4 v = x4[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = rsqrtf((sdata[0] / static_cast<float>(n)) + eps);
    }
    __syncthreads();
    float inv_rms = sdata[0];

    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n4; i += block_size) {
        float4 v = x4[i];
        float4 w = w4[i];
        float4 res;
        res.x = v.x * inv_rms * w.x;
        res.y = v.y * inv_rms * w.y;
        res.z = v.z * inv_rms * w.z;
        res.w = v.w * inv_rms * w.w;
        out4[i] = res;
    }
}

// UUS: 16-baidine FLOAT4 Vectorized Add + RMSNorm Fusion
__global__ void add_and_rmsnorm_kernel_float4(float* __restrict__ x, const float* __restrict__ residual,
                                              const float* __restrict__ weight, float* __restrict__ out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int n4 = n / 4;

    float partial = 0.0f;
    float4* x4 = reinterpret_cast<float4*>(x);
    const float4* res4 = reinterpret_cast<const float4*>(residual);

    for (int i = tid; i < n4; i += block_size) {
        float4 vx = x4[i];
        float4 vr = res4[i];
        vx.x += vr.x;
        vx.y += vr.y;
        vx.z += vr.z;
        vx.w += vr.w;
        x4[i] = vx;

        partial += vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = rsqrtf((sdata[0] / static_cast<float>(n)) + eps);
    }
    __syncthreads();
    float inv_rms = sdata[0];

    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n4; i += block_size) {
        float4 vx = x4[i];
        float4 w = w4[i];
        float4 res;
        res.x = vx.x * inv_rms * w.x;
        res.y = vx.y * inv_rms * w.y;
        res.z = vx.z * inv_rms * w.z;
        res.w = vx.w * inv_rms * w.w;
        out4[i] = res;
    }
}

// FlashAttention-lite (Online Softmax)
__global__ void mha_kv_cache_kernel(
    const float* __restrict__ Q, const float* __restrict__ K_cache, const float* __restrict__ V_cache,
    float* __restrict__ out, int num_heads, int head_dim, int kv_dim,
    int heads_per_kv_group, int current_seq_len, float scale)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_head = h / heads_per_kv_group;
    const float* q_ptr = Q + h * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;

    int tid = threadIdx.x;

    float local_max = -1e30f;
    float local_sum = 0.0f;

    for (int t = tid; t < current_seq_len; t += blockDim.x) {
        const float* k_ptr = K_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;

        #pragma unroll(8)
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float score = dot * scale;
        scores[t] = score;

        if (score > local_max) {
            float local_exp = expf(local_max - score);
            local_sum = local_sum * local_exp + 1.0f;
            local_max = score;
        } else {
            local_sum += expf(score - local_max);
        }
    }

    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m1 = s_max[tid];
            float m2 = s_max[tid + s];
            float s1 = s_sum[tid];
            float s2 = s_sum[tid + s];

            if (m1 > m2) {
                s_sum[tid] = s1 + s2 * expf(m2 - m1);
                s_max[tid] = m1;
            } else {
                s_sum[tid] = s1 * expf(m1 - m2) + s2;
                s_max[tid] = m2;
            }
        }
        __syncthreads();
    }

    float global_max = s_max[0];
    float global_sum = s_sum[0];
    __syncthreads();

    for (int t = tid; t < current_seq_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - global_max) / global_sum;
    }
    __syncthreads();

    float* head_out = out + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum_v = 0.0f;
        for (int t = 0; t < current_seq_len; ++t) {
            sum_v += scores[t] * V_cache[t * kv_dim + kv_head * head_dim + d];
        }
        head_out[d] = sum_v;
    }
}

// FLOAT4 Vector Add (16-baidised laksud)
__global__ void vector_add_kernel_float4(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (idx < n4) {
        float4 va = reinterpret_cast<float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        va.x += vb.x;
        va.y += vb.y;
        va.z += vb.z;
        va.w += vb.w;
        reinterpret_cast<float4*>(a)[idx] = va;
    }
}

// FLOAT4 FP32 Logits Warp-Coalesced
__global__ void gemv_f32_kernel_warp_float4(const float* __restrict__ W, const float* __restrict__ x,
                                     float* __restrict__ out, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    float local_sum = 0.0f;
    const float4* w_row_4 = reinterpret_cast<const float4*>(W + row * K);
    const float4* x_4 = reinterpret_cast<const float4*>(x);
    int K4 = K / 4;

    for (int k = lane_id; k < K4; k += 32) {
        float4 w_vec = w_row_4[k];
        float4 x_vec = x_4[k];
        local_sum += w_vec.x * x_vec.x + w_vec.y * x_vec.y + w_vec.z * x_vec.z + w_vec.w * x_vec.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane_id == 0) {
        out[row] = local_sum;
    }
}

__global__ void rope_adjacent_kernel(float* __restrict__ data, int total_heads,
                                     int head_dim, int pos, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = total_heads * half_dim;

    if (idx < total_pairs) {
        int h = idx / half_dim;
        int i = idx % half_dim;

        float freq = 1.0f / powf(base, (2.0f * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(pos) * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float* head_ptr = data + h * head_dim;
        float x0 = head_ptr[2 * i];
        float x1 = head_ptr[2 * i + 1];

        head_ptr[2 * i]     = x0 * cos_val - x1 * sin_val;
        head_ptr[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

__global__ void rope_and_cache_k_kernel(
    const float* __restrict__ k_in,
    float* __restrict__ k_cache_out,
    int total_heads, int head_dim, int pos, float base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = total_heads * half_dim;

    if (idx < total_pairs) {
        int h = idx / half_dim;
        int i = idx % half_dim;

        float freq = 1.0f / powf(base, (2.0f * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(pos) * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float x0 = k_in[h * head_dim + 2 * i];
        float x1 = k_in[h * head_dim + 2 * i + 1];

        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;

        int cache_offset = h * head_dim;
        k_cache_out[cache_offset + 2 * i] = out0;
        k_cache_out[cache_offset + 2 * i + 1] = out1;
    }
}

struct CUDABuffersQ4 {
    float* x = nullptr;
    float* xb = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_concat = nullptr;
    float* attn_out = nullptr;
    float* ffn_norm = nullptr;
    float* w1 = nullptr;
    float* w3 = nullptr;
    float* swiglu_out = nullptr;
    float* ffn_final = nullptr;
    float* final_hidden = nullptr;
    float* logits = nullptr;

    std::vector<float*> k_cache;
    std::vector<float*> v_cache;

    void allocate(const ModelConfig& cfg) {
        int dim = cfg.dim;
        int hidden_dim = cfg.hidden_dim;
        int num_heads = cfg.num_heads;
        int num_kv_heads = cfg.num_kv_heads;
        int head_dim = dim / num_heads;
        int kv_dim = head_dim * num_kv_heads;

        cudaMalloc(&x, dim * sizeof(float));
        cudaMalloc(&xb, dim * sizeof(float));
        cudaMalloc(&q, dim * sizeof(float));
        cudaMalloc(&k, kv_dim * sizeof(float));
        cudaMalloc(&v, kv_dim * sizeof(float));
        cudaMalloc(&attn_concat, dim * sizeof(float));
        cudaMalloc(&attn_out, dim * sizeof(float));
        cudaMalloc(&ffn_norm, dim * sizeof(float));
        cudaMalloc(&w1, hidden_dim * sizeof(float));
        cudaMalloc(&w3, hidden_dim * sizeof(float));
        cudaMalloc(&swiglu_out, hidden_dim * sizeof(float));
        cudaMalloc(&ffn_final, dim * sizeof(float));
        cudaMalloc(&final_hidden, dim * sizeof(float));
        cudaMalloc(&logits, cfg.vocab_size * sizeof(float));

        k_cache.resize(cfg.num_layers);
        v_cache.resize(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) {
            cudaMalloc(&k_cache[l], cfg.seq_len * kv_dim * sizeof(float));
            cudaMalloc(&v_cache[l], cfg.seq_len * kv_dim * sizeof(float));
        }
    }

    void reset_kv_cache(const ModelConfig& cfg) {
        int kv_dim = (cfg.dim / cfg.num_heads) * cfg.num_kv_heads;
        for (int l = 0; l < cfg.num_layers; ++l) {
            cudaMemset(k_cache[l], 0, cfg.seq_len * kv_dim * sizeof(float));
            cudaMemset(v_cache[l], 0, cfg.seq_len * kv_dim * sizeof(float));
        }
    }
};

void forward_pass_gguf_cuda(const CUDATransformerModelQ4& model, int token_id, int pos,
                            CUDABuffersQ4& bufs, float* host_logits) {
    const auto& cfg = model.config;
    int dim = cfg.dim;
    int hidden_dim = cfg.hidden_dim;
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = dim / num_heads;
    int kv_dim = head_dim * num_kv_heads;
    int heads_per_kv_group = num_heads / num_kv_heads;

    int block_256 = 256;
    size_t norm_smem = 256 * sizeof(float);

    const float* emb_ptr = model.token_embedding_table + static_cast<size_t>(token_id) * dim;
    cudaMemcpyAsync(bufs.x, emb_ptr, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];

        // FLOAT4 RMSNorm
        rmsnorm_kernel_float4<<<1, 256, norm_smem>>>(bufs.x, lw.attention_norm, bufs.xb, dim, 1e-5f);

        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.wq, bufs.xb, bufs.q, dim, dim);
        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.wk, bufs.xb, bufs.k, kv_dim, dim);

        float* v_dest = bufs.v_cache[l] + static_cast<size_t>(pos) * kv_dim;
        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.wv, bufs.xb, v_dest, kv_dim, dim);

        int q_pairs = num_heads * (head_dim / 2);
        rope_adjacent_kernel<<<(q_pairs + block_256 - 1) / block_256, block_256>>>(bufs.q, num_heads, head_dim, pos, 10000.0f);

        float* k_dest = bufs.k_cache[l] + static_cast<size_t>(pos) * kv_dim;
        int k_pairs = num_kv_heads * (head_dim / 2);
        rope_and_cache_k_kernel<<<(k_pairs + block_256 - 1) / block_256, block_256>>>(
            bufs.k, k_dest, num_kv_heads, head_dim, pos, 10000.0f);

        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        int current_seq_len = pos + 1;
        size_t mha_smem = current_seq_len * sizeof(float);

        mha_kv_cache_kernel<<<num_heads, 256, mha_smem>>>(
            bufs.q, bufs.k_cache[l], bufs.v_cache[l], bufs.attn_concat,
            num_heads, head_dim, kv_dim, heads_per_kv_group, current_seq_len, scale);

        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.wo, bufs.attn_concat, bufs.attn_out, dim, dim);

        // FLOAT4 Mega-Fusion (Add + RMSNorm)
        add_and_rmsnorm_kernel_float4<<<1, 256, norm_smem>>>(bufs.x, bufs.attn_out, lw.ffn_norm, bufs.ffn_norm, dim, 1e-5f);

        // FFN W1 + W3 + SwiGLU (Q4_0 Vectorized!)
        tensor_math_quantized_cuda::launch_gemv_q4_0_w1_w3_swiglu(lw.w1, lw.w3, bufs.ffn_norm, bufs.swiglu_out, hidden_dim, dim);

        tensor_math_quantized_cuda::launch_gemv_q4_0(lw.w2, bufs.swiglu_out, bufs.ffn_final, dim, hidden_dim);

        // FLOAT4 Vector Add
        vector_add_kernel_float4<<<(dim / 4 + block_256 - 1) / block_256, block_256>>>(bufs.x, bufs.ffn_final, dim);
    }

    // FLOAT4 Viimane RMSNorm
    rmsnorm_kernel_float4<<<1, 256, norm_smem>>>(bufs.x, model.final_norm, bufs.final_hidden, dim, 1e-5f);

    // FLOAT4 FP32 Logits
    int f32_threads = 128;
    int f32_blocks = (cfg.vocab_size + (f32_threads / 32) - 1) / (f32_threads / 32);
    gemv_f32_kernel_warp_float4<<<f32_blocks, f32_threads>>>(
        model.output_weights, bufs.final_hidden, bufs.logits, cfg.vocab_size, dim);

    cudaMemcpy(host_logits, bufs.logits, cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
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
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_v) { max_v = logits[i]; max_i = i; }
        }
        return max_i;
    }

    float max_val = logits[0] / temperature;
    for (int i = 1; i < vocab_size; ++i) {
        float val = logits[i] / temperature;
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    std::vector<std::pair<float, int>> vec;
    vec.reserve(vocab_size / 10);

    float threshold = 1e-4f;
    for (int i = 0; i < vocab_size; ++i) {
        float prob = std::exp((logits[i] / temperature) - max_val);
        sum += prob;
        if (prob > threshold) {
            vec.push_back({prob, i});
        }
    }

    for (auto& pair : vec) {
        pair.first /= sum;
    }

    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (topp < 1.0f) {
        float cumulative_prob = 0.0f;
        int last_idx = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            cumulative_prob += vec[i].first;
            last_idx = static_cast<int>(i);
            if (cumulative_prob > topp) break;
        }
        vec.resize(last_idx + 1);

        float p_sum = 0.0f;
        for (const auto& pair : vec) p_sum += pair.first;
        for (auto& pair : vec) pair.first /= p_sum;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cdf = 0.0f;
    for (const auto& pair : vec) {
        cdf += pair.first;
        if (r <= cdf) return pair.second;
    }

    return vec.back().second;
}

int main() {
    std::cout << "=== Sovereign Kernel: GGUF CUDA Q4_0 Engine (TinyLlama 1.1B) ===\n";

    try {
        std::string gguf_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
        std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";
        std::string config_path = "runtime_config.txt";
        std::string prompt_config_path = "runtime_config.json";

        ModelConfig config;
        config.dim = 2048;
        config.hidden_dim = 5632;
        config.num_layers = 22;
        config.num_heads = 32;
        config.num_kv_heads = 4;
        config.vocab_size = 32000;
        config.seq_len = 1024;

        CUDATransformerModelQ4 cuda_model;
        cuda_model.config = config;

        std::cout << "[INFO] Booting GGUFLoaderCUDA...\n";
        GGUFLoaderCUDA::load_weights_to_vram(gguf_path, cuda_model);

        Tokenizer tokenizer;
        tokenizer.load(tokenizer_path, config.vocab_size);
        std::cout << "[INFO] Tokenizer initialized.\n";

        CUDABuffersQ4 bufs;
        bufs.allocate(config);

        std::vector<float> host_logits(config.vocab_size);
        std::random_device rd;
        std::mt19937 rng(rd());

        std::cout << "\n=======================================================\n";
        std::cout << " GGUF CUDA Inference Ready \n";
        std::cout << "=======================================================\n\n";

        std::string user_input;
        while (true) {
            std::cout << "\n[Darth SulxX] > ";
            if (!std::getline(std::cin, user_input)) break;
            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;

            RuntimeConfig rcfg = load_runtime_config(config_path);
            std::string sys_prompt = load_system_prompt(prompt_config_path);

            std::string formatted_prompt = "<|system|>\n" + sys_prompt + "</s>\n<|user|>\n" + user_input + "</s>\n<|assistant|>\n";
            std::vector<int> tokens = tokenizer.encode(formatted_prompt);

            std::cout << "\n[Misha GGUF CUDA Output]: " << std::flush;

            bufs.reset_kv_cache(config);
            int pos = 0;

            for (size_t i = 0; i < tokens.size(); ++i) {
                forward_pass_gguf_cuda(cuda_model, tokens[i], pos, bufs, host_logits.data());
                pos++;
            }

            int generated_token_count = 0;
            double total_forward_ms = 0.0;
            auto gen_start = std::chrono::high_resolution_clock::now();
            std::vector<int> recent_tokens;

            for (int step = 0; step < rcfg.max_tokens; ++step) {
                int next_token = sample_token(host_logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);

                if (next_token == 2 && generated_token_count < rcfg.min_tokens) {
                    host_logits[2] = -1e9f;
                    next_token = sample_token(host_logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);
                }
                if (next_token == 2 && generated_token_count >= rcfg.min_tokens) break;

                recent_tokens.push_back(next_token);
                if (recent_tokens.size() > static_cast<size_t>(rcfg.penalty_window)) {
                    recent_tokens.erase(recent_tokens.begin());
                }

                std::string piece = tokenizer.decode(next_token);
                std::cout << piece << std::flush;
                generated_token_count++;

                auto t0 = std::chrono::high_resolution_clock::now();
                forward_pass_gguf_cuda(cuda_model, next_token, pos, bufs, host_logits.data());
                auto t1 = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> fp_time = t1 - t0;
                total_forward_ms += fp_time.count();

                pos++;
                if (pos >= config.seq_len) break;
            }

            auto gen_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = gen_end - gen_start;
            float tokens_per_sec = (elapsed.count() > 0.0f) ? (static_cast<float>(generated_token_count) / elapsed.count()) : 0.0f;
            double avg_fp_ms = (generated_token_count > 0) ? (total_forward_ms / generated_token_count) : 0.0;

            std::cout << "\n\n[CUDA Performance Stats] Generated: " << generated_token_count
                      << " tokens | Time: " << elapsed.count() << "s | Speed: " << tokens_per_sec << " tok/s"
                      << " | Avg Forward Pass: " << avg_fp_ms << " ms\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL CUDA ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}