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

#include "tensor.h"
#include "model_config.h"
#include "model.h"
#include "model_loader.h"
#include "tokenizer.h"

// =========================================================================
// CUDA KERNELID: Kogu Transformeri matemaatika VRAM-is
// =========================================================================

// 1. RMSNorm
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                               float* __restrict__ out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float partial = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float v = x[i];
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq = sdata[0] / static_cast<float>(n);
        sdata[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float inv_rms = sdata[0];

    for (int i = tid; i < n; i += block_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// 2. GEMV: Matrix-Vector korrutis (y = W * x)
__global__ void gemv_kernel(const float* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ out, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        const float* w_row = W + row * K;
        for (int k = 0; k < K; ++k) {
            sum += w_row[k] * x[k];
        }
        out[row] = sum;
    }
}

// 3. RoPE (HuggingFace Split Mode - 1:1 vastavus CPU referentsile)
__global__ void rope_hf_split_kernel(float* __restrict__ data, int total_heads,
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
        float x0 = head_ptr[i];
        float x1 = head_ptr[i + half_dim];

        head_ptr[i]            = x0 * cos_val - x1 * sin_val;
        head_ptr[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// 4. Multi-Head Attention üle VRAM KV-Cache'i (Decoder Autoregressive Step)
__global__ void mha_kv_cache_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ out,
    int num_heads,
    int head_dim,
    int kv_dim,
    int heads_per_kv_group,
    int current_seq_len,
    float scale)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_head = h / heads_per_kv_group;
    const float* q_ptr = Q + h * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;

    // 4.1. Q * K^T arvutamine iga t in [0, current_seq_len - 1] kohta
    for (int t = threadIdx.x; t < current_seq_len; t += blockDim.x) {
        const float* k_ptr = K_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // 4.2. Softmax: Leia max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < current_seq_len; t += blockDim.x) {
        if (scores[t] > local_max) local_max = scores[t];
    }
    __shared__ float s_reduce[256];
    s_reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = s_reduce[0];
    __syncthreads();

    // 4.3. Softmax: Exp ja summa
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < current_seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_val = s_reduce[0];
    __syncthreads();

    // 4.4. Normaliseeri
    for (int t = threadIdx.x; t < current_seq_len; t += blockDim.x) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // 4.5. Weighted Sum üle V
    float* head_out = out + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum_v = 0.0f;
        for (int t = 0; t < current_seq_len; ++t) {
            float w = scores[t];
            const float* v_ptr = V_cache + t * kv_dim + kv_head * head_dim;
            sum_v += w * v_ptr[d];
        }
        head_out[d] = sum_v;
    }
}

// 5. SwiGLU aktivatsioon
__global__ void swiglu_kernel(const float* __restrict__ w1, const float* __restrict__ w3,
                              float* __restrict__ out, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float a = w1[idx];
        float b = w3[idx];
        float silu_a = a / (1.0f + expf(-a));
        out[idx] = silu_a * b;
    }
}

// 6. Residual Add (a += b)
__global__ void vector_add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

// =========================================================================
// CUDA MUDELI STRUKTUURID JA MÄLUHALDUS
// =========================================================================

struct CUDALayerWeights {
    float* attention_norm = nullptr;
    float* wq = nullptr;
    float* wk = nullptr;
    float* wv = nullptr;
    float* wo = nullptr;
    float* ffn_norm = nullptr;
    float* w1 = nullptr;
    float* w2 = nullptr;
    float* w3 = nullptr;
};

struct CUDATransformerModel {
    ModelConfig config;
    float* token_embedding_table = nullptr;
    std::vector<CUDALayerWeights> layers;
    float* final_norm = nullptr;
    float* output_weights = nullptr;
};

struct CUDABuffers {
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
        int vocab_size = cfg.vocab_size;
        int max_seq = cfg.seq_len;

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
        cudaMalloc(&logits, vocab_size * sizeof(float));

        k_cache.resize(cfg.num_layers);
        v_cache.resize(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) {
            cudaMalloc(&k_cache[l], max_seq * kv_dim * sizeof(float));
            cudaMalloc(&v_cache[l], max_seq * kv_dim * sizeof(float));
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

float* upload_tensor_to_device(const Tensor& t) {
    float* d_ptr = nullptr;
    size_t bytes = t.num_elements() * sizeof(float);
    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, t.data(), bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

void upload_model_to_cuda(const TransformerModel& cpu_model, CUDATransformerModel& cuda_model) {
    cuda_model.config = cpu_model.config;
    cuda_model.token_embedding_table = upload_tensor_to_device(cpu_model.token_embedding_table);
    cuda_model.final_norm = upload_tensor_to_device(cpu_model.final_norm);

    if (cpu_model.config.output_weights_tied) {
        cuda_model.output_weights = cuda_model.token_embedding_table;
    } else {
        cuda_model.output_weights = upload_tensor_to_device(cpu_model.output_weights);
    }

    cuda_model.layers.resize(cpu_model.config.num_layers);
    for (int l = 0; l < cpu_model.config.num_layers; ++l) {
        const auto& cl = cpu_model.layers[l];
        auto& dl = cuda_model.layers[l];
        dl.attention_norm = upload_tensor_to_device(cl.attention_norm);
        dl.wq = upload_tensor_to_device(cl.wq);
        dl.wk = upload_tensor_to_device(cl.wk);
        dl.wv = upload_tensor_to_device(cl.wv);
        dl.wo = upload_tensor_to_device(cl.wo);
        dl.ffn_norm = upload_tensor_to_device(cl.ffn_norm);
        dl.w1 = upload_tensor_to_device(cl.w1);
        dl.w2 = upload_tensor_to_device(cl.w2);
        dl.w3 = upload_tensor_to_device(cl.w3);
    }
}

// =========================================================================
// CUDA FORWARD PASS
// =========================================================================

void forward_pass_cuda(const CUDATransformerModel& model, int token_id, int pos,
                       CUDABuffers& bufs, float* host_logits) {
    const auto& cfg = model.config;
    int dim = cfg.dim;
    int hidden_dim = cfg.hidden_dim;
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = dim / num_heads;
    int kv_dim = head_dim * num_kv_heads;
    int heads_per_kv_group = num_heads / num_kv_heads;
    int vocab_size = cfg.vocab_size;

    // 1. Embedding lookup otse VRAM-is
    const float* emb_ptr = model.token_embedding_table + static_cast<size_t>(token_id) * dim;
    cudaMemcpyAsync(bufs.x, emb_ptr, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    int block_256 = 256;

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];

        // 2. RMSNorm enne Attentionit
        size_t norm_smem = 256 * sizeof(float);
        rmsnorm_kernel<<<1, 256, norm_smem>>>(bufs.x, lw.attention_norm, bufs.xb, dim, 1e-5f);

        // 3. Q, K, V Projections (GEMV)
        gemv_kernel<<<(dim + block_256 - 1) / block_256, block_256>>>(lw.wq, bufs.xb, bufs.q, dim, dim);
        gemv_kernel<<<(kv_dim + block_256 - 1) / block_256, block_256>>>(lw.wk, bufs.xb, bufs.k, kv_dim, dim);
        gemv_kernel<<<(kv_dim + block_256 - 1) / block_256, block_256>>>(lw.wv, bufs.xb, bufs.v, kv_dim, dim);

        // 4. RoPE
        int q_pairs = num_heads * (head_dim / 2);
        rope_hf_split_kernel<<<(q_pairs + block_256 - 1) / block_256, block_256>>>(
            bufs.q, num_heads, head_dim, pos, 10000.0f);

        int k_pairs = num_kv_heads * (head_dim / 2);
        rope_hf_split_kernel<<<(k_pairs + block_256 - 1) / block_256, block_256>>>(
            bufs.k, num_kv_heads, head_dim, pos, 10000.0f);

        // 5. KV Cache Append (Kirjutame praeguse tokeni K ja V ajalukku)
        float* k_dest = bufs.k_cache[l] + static_cast<size_t>(pos) * kv_dim;
        float* v_dest = bufs.v_cache[l] + static_cast<size_t>(pos) * kv_dim;
        cudaMemcpyAsync(k_dest, bufs.k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(v_dest, bufs.v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // 6. Multi-Head Attention
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        int current_seq_len = pos + 1;
        size_t mha_smem = current_seq_len * sizeof(float);

        mha_kv_cache_kernel<<<num_heads, 256, mha_smem>>>(
            bufs.q, bufs.k_cache[l], bufs.v_cache[l], bufs.attn_concat,
            num_heads, head_dim, kv_dim, heads_per_kv_group, current_seq_len, scale);

        // 7. Output projection (W_o)
        gemv_kernel<<<(dim + block_256 - 1) / block_256, block_256>>>(lw.wo, bufs.attn_concat, bufs.attn_out, dim, dim);

        // 8. Residual Add (x = x + attn_out)
        vector_add_kernel<<<(dim + block_256 - 1) / block_256, block_256>>>(bufs.x, bufs.attn_out, dim);

        // 9. FFN RMSNorm
        rmsnorm_kernel<<<1, 256, norm_smem>>>(bufs.x, lw.ffn_norm, bufs.ffn_norm, dim, 1e-5f);

        // 10. W1 ja W3
        gemv_kernel<<<(hidden_dim + block_256 - 1) / block_256, block_256>>>(lw.w1, bufs.ffn_norm, bufs.w1, hidden_dim, dim);
        gemv_kernel<<<(hidden_dim + block_256 - 1) / block_256, block_256>>>(lw.w3, bufs.ffn_norm, bufs.w3, hidden_dim, dim);

        // 11. SwiGLU
        swiglu_kernel<<<(hidden_dim + block_256 - 1) / block_256, block_256>>>(bufs.w1, bufs.w3, bufs.swiglu_out, hidden_dim);

        // 12. W2 (Down projection)
        gemv_kernel<<<(dim + block_256 - 1) / block_256, block_256>>>(lw.w2, bufs.swiglu_out, bufs.ffn_final, dim, hidden_dim);

        // 13. Residual Add (x = x + ffn_final)
        vector_add_kernel<<<(dim + block_256 - 1) / block_256, block_256>>>(bufs.x, bufs.ffn_final, dim);
    }

    // 14. Final RMSNorm
    size_t norm_smem = 256 * sizeof(float);
    rmsnorm_kernel<<<1, 256, norm_smem>>>(bufs.x, model.final_norm, bufs.final_hidden, dim, 1e-5f);

    // 15. Logits projection
    gemv_kernel<<<(vocab_size + block_256 - 1) / block_256, block_256>>>(
        model.output_weights, bufs.final_hidden, bufs.logits, vocab_size, dim);

    // 16. Kopeerime CPU-sse AINULT viimased logitid sämplimiseks
    cudaMemcpy(host_logits, bufs.logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// =========================================================================
// SÄMPLIMINE (CPU)
// =========================================================================

int sample_token(float* logits, int vocab_size, float temperature, float topp, std::mt19937& rng) {
    if (temperature == 0.0f) {
        int max_i = 0;
        float max_v = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_v) {
                max_v = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    std::vector<std::pair<float, int>> vec(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        vec[i] = {logits[i] / temperature, i};
    }

    float max_val = vec[0].first;
    for (int i = 1; i < vocab_size; ++i) {
        if (vec[i].first > max_val) max_val = vec[i].first;
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        vec[i].first = std::exp(vec[i].first - max_val);
        sum += vec[i].first;
    }

    for (int i = 0; i < vocab_size; ++i) {
        vec[i].first /= sum;
    }

    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    if (topp < 1.0f) {
        float cumulative_prob = 0.0f;
        int last_idx = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            cumulative_prob += vec[i].first;
            last_idx = static_cast<int>(i);
            if (cumulative_prob > topp) break;
        }
        vec.resize(last_idx + 1);

        sum = 0.0f;
        for (size_t i = 0; i < vec.size(); ++i) sum += vec[i].first;
        for (size_t i = 0; i < vec.size(); ++i) vec[i].first /= sum;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cdf = 0.0f;
    for (size_t i = 0; i < vec.size(); ++i) {
        cdf += vec[i].first;
        if (r <= cdf) return vec[i].second;
    }

    return vec.back().second;
}

// =========================================================================
// MAIN RUNNER
// =========================================================================

int main() {
    std::cout << "=== Sovereign Kernel: CUDA 110M TinyLlama Inference Engine ===\n";

    try {
        std::string weights_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/stories110M.bin";
        std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";

        TransformerCheckpointHeader hdr = peek_model_header(weights_path);
        ModelConfig config;
        config.dim = hdr.dim;
        config.hidden_dim = hdr.hidden_dim;
        config.num_layers = hdr.num_layers;
        config.num_heads = hdr.num_heads;
        config.num_kv_heads = hdr.num_kv_heads;
        config.vocab_size = std::abs(hdr.vocab_size);
        config.output_weights_tied = (hdr.vocab_size >= 0);
        config.seq_len = hdr.seq_len;

        std::cout << "[INFO] Loading CPU binary weights...\n";
        TransformerModel cpu_model(config);
        load_model_weights(weights_path, cpu_model);

        std::cout << "[INFO] Transferring entire model weights to VRAM (Device::CUDA)...\n";
        CUDATransformerModel cuda_model;
        upload_model_to_cuda(cpu_model, cuda_model);
        std::cout << "[INFO] Model successfully locked in VRAM!\n";

        Tokenizer tokenizer;
        tokenizer.load(tokenizer_path, config.vocab_size);
        std::cout << "[INFO] Tokenizer initialized.\n";

        CUDABuffers bufs;
        bufs.allocate(config);

        std::vector<float> host_logits(config.vocab_size);
        std::random_device rd;
        std::mt19937 rng(rd());

        std::cout << "\n=======================================================\n";
        std::cout << " CUDA Inference Ready (Temperature: 0.8, Top-P: 0.9)\n";
        std::cout << "=======================================================\n\n";

        std::string user_input;
        while (true) {
            std::cout << "\n[Darth SulxX] > ";
            if (!std::getline(std::cin, user_input)) break;
            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;

            std::cout << "\n[Misha CUDA Output]: " << std::flush;

            bufs.reset_kv_cache(config);
            std::vector<int> tokens = tokenizer.encode(user_input);

            int pos = 0;
            for (size_t i = 0; i < tokens.size(); ++i) {
                forward_pass_cuda(cuda_model, tokens[i], pos, bufs, host_logits.data());
                pos++;
            }

            int generated_token_count = 0;
            double total_forward_ms = 0.0;
            auto gen_start = std::chrono::high_resolution_clock::now();

            int max_tokens = 1000;
            for (int step = 0; step < max_tokens; ++step) {
                int next_token = sample_token(host_logits.data(), config.vocab_size, 0.8f, 0.9f, rng);
                if (next_token == 2) break; // EOS

                std::string piece = tokenizer.decode(next_token);
                std::cout << piece << std::flush;
                generated_token_count++;

                auto t0 = std::chrono::high_resolution_clock::now();
                forward_pass_cuda(cuda_model, next_token, pos, bufs, host_logits.data());
                cudaDeviceSynchronize();
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