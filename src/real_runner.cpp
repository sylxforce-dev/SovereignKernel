#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

#include "tensor.h"
#include "model_config.h"
#include "model.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "tensor_math_cpu.h"
#include "tensor_math_transformer.h"
#include "tensor_kv_cache.h"

#ifdef _OPENMP
#include <omp.h>
#endif

std::string load_external_system_prompt(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "[Warning] Hoiatus: Ei suutnud avada süsteemiprompti faili (" << filepath << "). Kasutan vaikeseadet.\n";
        return "You are Darth Misha, the sovereign core of the Sovereign Kernel. You operate entirely through pure logic, structural analysis, and system architecture.";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline int vocab_size_safe(const ModelConfig& cfg) {
    return cfg.vocab_size;
}

void forward_pass_with_attention(const TransformerModel& model, int token_id, int pos,
                                  std::vector<TensorKVCache>& kv_caches, Tensor& logits) {
    const auto& cfg = model.config;
    int dim = cfg.dim;
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = dim / num_heads;
    int kv_dim = head_dim * num_kv_heads;
    int heads_per_kv_group = num_heads / num_kv_heads;
    int hidden_dim = cfg.hidden_dim;

    Tensor x({dim}, Device::CPU);
    const float* emb_ptr = model.token_embedding_table.data() + static_cast<size_t>(token_id) * dim;
    std::copy(emb_ptr, emb_ptr + dim, x.data());

    // ===== PARANDUS: matmul_inplace + eelnevalt allokeeritud scratch-puhvrid =====
    static thread_local std::vector<float> q_buf, k_buf, v_buf;
    static thread_local std::vector<float> attn_out_buf;
    static thread_local std::vector<float> w1_buf, w3_buf;
    static thread_local std::vector<float> ffn_final_buf;

    if ((int)q_buf.size() < dim) q_buf.resize(dim);
    if ((int)k_buf.size() < kv_dim) k_buf.resize(kv_dim);
    if ((int)v_buf.size() < kv_dim) v_buf.resize(kv_dim);
    if ((int)attn_out_buf.size() < dim) attn_out_buf.resize(dim);
    if ((int)w1_buf.size() < hidden_dim) w1_buf.resize(hidden_dim);
    if ((int)w3_buf.size() < hidden_dim) w3_buf.resize(hidden_dim);
    if ((int)ffn_final_buf.size() < dim) ffn_final_buf.resize(dim);

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];

        Tensor xb = tensor_math_transformer::rmsnorm(x, lw.attention_norm);

        tensor_math_cpu::matmul_inplace(lw.wq, xb.data(), q_buf.data(), dim, dim);
        tensor_math_cpu::matmul_inplace(lw.wk, xb.data(), k_buf.data(), kv_dim, dim);
        tensor_math_cpu::matmul_inplace(lw.wv, xb.data(), v_buf.data(), kv_dim, dim);

        Tensor q_rot({dim}, Device::CPU);
        Tensor k_rot({kv_dim}, Device::CPU);

        // Režiim 1 (HF split) - originaalne kood kutsus rope(q_head, pos) vaikeväärtusega
        for (int h = 0; h < num_heads; ++h) {
            tensor_math_transformer::rope_inplace_hf_split(
                q_buf.data() + h * head_dim, q_rot.data() + h * head_dim, head_dim, pos);
        }
        for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
            tensor_math_transformer::rope_inplace_hf_split(
                k_buf.data() + kvh * head_dim, k_rot.data() + kvh * head_dim, head_dim, pos);
        }

        Tensor v({kv_dim}, Device::CPU);
        std::copy(v_buf.data(), v_buf.data() + kv_dim, v.data());

        kv_caches[l].append(k_rot, v);

        const auto& k_hist = kv_caches[l].keys();
        const auto& v_hist = kv_caches[l].values();
        size_t hist_len = k_hist.size();

        Tensor attn_concat({dim}, Device::CPU);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        #pragma omp parallel for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
            static thread_local std::vector<float> local_scores;
            static thread_local std::vector<float> local_weights;
            if (local_scores.size() < hist_len) local_scores.resize(hist_len);
            if (local_weights.size() < hist_len) local_weights.resize(hist_len);

            int kv_head = h / heads_per_kv_group;
            const float* q_head_ptr = q_rot.data() + h * head_dim;

            for (size_t t = 0; t < hist_len; ++t) {
                const float* k_head_ptr = k_hist[t].data() + kv_head * head_dim;
                float dotv = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dotv += q_head_ptr[d] * k_head_ptr[d];
                }
                local_scores[t] = dotv * scale;
            }

            float max_val = local_scores[0];
            for (size_t t = 1; t < hist_len; ++t) {
                if (local_scores[t] > max_val) max_val = local_scores[t];
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < hist_len; ++t) {
                local_weights[t] = std::exp(local_scores[t] - max_val);
                sum_exp += local_weights[t];
            }
            for (size_t t = 0; t < hist_len; ++t) {
                local_weights[t] /= sum_exp;
            }

            float* head_out_ptr = attn_concat.data() + h * head_dim;
            std::fill(head_out_ptr, head_out_ptr + head_dim, 0.0f);
            for (size_t t = 0; t < hist_len; ++t) {
                float w = local_weights[t];
                const float* v_t = v_hist[t].data() + kv_head * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    head_out_ptr[d] += w * v_t[d];
                }
            }
        }

        tensor_math_cpu::matmul_inplace(lw.wo, attn_concat.data(), attn_out_buf.data(), dim, dim);

        for (int i = 0; i < dim; ++i) {
            x.data()[i] += attn_out_buf[i];
        }

        Tensor ffn_norm = tensor_math_transformer::rmsnorm(x, lw.ffn_norm);

        tensor_math_cpu::matmul_inplace(lw.w1, ffn_norm.data(), w1_buf.data(), hidden_dim, dim);
        tensor_math_cpu::matmul_inplace(lw.w3, ffn_norm.data(), w3_buf.data(), hidden_dim, dim);

        Tensor swiglu_in({hidden_dim * 2}, Device::CPU);
        std::copy(w1_buf.data(), w1_buf.data() + hidden_dim, swiglu_in.data());
        std::copy(w3_buf.data(), w3_buf.data() + hidden_dim, swiglu_in.data() + hidden_dim);

        Tensor swiglu_out = tensor_math_transformer::swiglu(swiglu_in);

        tensor_math_cpu::matmul_inplace(lw.w2, swiglu_out.data(), ffn_final_buf.data(), dim, hidden_dim);

        for (int i = 0; i < dim; ++i) {
            x.data()[i] += ffn_final_buf[i];
        }
    }

    Tensor final_hidden = tensor_math_transformer::rmsnorm(x, model.final_norm);

    Tensor hidden_2d({dim, 1}, Device::CPU);
    std::copy(final_hidden.data(), final_hidden.data() + dim, hidden_2d.data());
    Tensor logits_2d = tensor_math_cpu::matmul(model.output_weights, hidden_2d);

    float* out_ptr = logits.data();
    for (int v = 0; v < vocab_size_safe(cfg); ++v) {
        out_ptr[v] = logits_2d.data()[v * 1 + 0];
    }
}

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

int main() {
#ifdef _OPENMP
    omp_set_num_threads(8); // Ryzen 7700 füüsiliste tuumade arv
#endif

    std::cout << "=== Sovereign Kernel: Sampling CLI (Base & Chat Ready) ===\n";

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "[OpenMP Active] Ryzen 7 tuumad rakkes, aktiivseid lõimi: " << omp_get_num_threads() << "\n";
    }
#else
    std::cout << "[OpenMP OFF] Hoiatus: Kood jookseb ainult ühel tuumal!\n";
#endif

    try {
        std::string weights_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/stories110M.bin";
        std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";
        std::string prompt_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/system_prompt.txt";

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

        TransformerModel model(config);

        std::cout << "Loading binary weights into Sovereign Kernels Tensor memory...\n";
        load_model_weights(weights_path, model);
        std::cout << "Weights mapped successfully!\n";

        Tokenizer tokenizer;
        tokenizer.load(tokenizer_path, config.vocab_size);
        std::cout << "Tokenizer initialized.\n\n========================================\n";
        std::cout << " CLI Ready with Timers (Temp: 0.2, Top-P: 0.9)\n========================================\n\n";

        std::random_device rd;
        std::mt19937 rng(rd());

        std::string user_input;
        while (true) {
            std::cout << "\n[Darth SulxX] > ";
            if (!std::getline(std::cin, user_input)) break;
            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;

            std::cout << "\n[Misha Model Output]: " << std::flush;

            std::string prompt = user_input;
            std::vector<int> tokens = tokenizer.encode(prompt);

            Tensor logits({config.vocab_size}, Device::CPU);

            std::vector<TensorKVCache> kv_caches;
            kv_caches.reserve(config.num_layers);
            for (int l = 0; l < config.num_layers; ++l) {
                kv_caches.emplace_back(static_cast<size_t>(config.seq_len));
            }

            int pos = 0;
            for (size_t i = 0; i < tokens.size(); ++i) {
                forward_pass_with_attention(model, tokens[i], pos, kv_caches, logits);
                pos++;
            }

            std::string accumulated_output = "";
            int generated_token_count = 0;
            double total_forward_ms = 0.0;

            auto gen_start = std::chrono::high_resolution_clock::now();

            int max_tokens = 100;
            for (int step = 0; step < max_tokens; ++step) {
                int next_token = sample_token(logits.data(), config.vocab_size, 0.8f, 0.9f, rng);

                if (next_token == 2) break;

                std::string piece = tokenizer.decode(next_token);
                accumulated_output += piece;

                std::cout << piece << std::flush;
                generated_token_count++;

                auto t0 = std::chrono::high_resolution_clock::now();
                forward_pass_with_attention(model, next_token, pos, kv_caches, logits);
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

            std::cout << "\n\n[Performance Stats] Generated tokens: " << generated_token_count
                      << " | Time: " << elapsed.count() << "s | Speed: " << tokens_per_sec << " tok/s"
                      << " | Avg Forward Pass: " << avg_fp_ms << " ms\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL TENSOR ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}