#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <unordered_map>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "tensor.h"
#include "model_config.h"
#include "model.h"
#include "tokenizer.h"
#include "tensor_math_cpu.h"
#include "tensor_math_transformer.h"
#include "tensor_kv_cache.h"

#include "gguf/gguf_reader.h"
#include "gguf/gguf_types.h"
#include "gguf/gguf_loader.h"
#include "gguf/tensor_math_quantized.h"

#ifdef _OPENMP
#include <omp.h>
#endif

struct RuntimeConfig {
    float temperature = 0.3f;
    float top_p = 0.9f;
    float rep_penalty = 1.15f;
    int min_tokens = 50;
    int max_tokens = 100;
    int penalty_window = 64;
};

RuntimeConfig load_runtime_config(const std::string& filepath) {
    RuntimeConfig cfg;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return cfg;
    }

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

inline int vocab_size_safe(const ModelConfig& cfg) {
    return cfg.vocab_size;
}

void inspect_tensor(const Tensor& t, const std::string& name) {
    float max_val = -1e9f;
    float min_val = 1e9f;
    bool has_nan = false;
    for(size_t i = 0; i < t.num_elements(); i++) {
        float v = t.data()[i];
        if(std::isnan(v) || std::isinf(v)) has_nan = true;
        if(v > max_val) max_val = v;
        if(v < min_val) min_val = v;
    }
    std::cout << "[Sovereign Telemetry] " << name << " -> Min: " << min_val << " | Max: " << max_val << " | Mürgitatud: " << (has_nan ? "JAH (NaN/Inf)" : "EI") << "\n";
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

    if (pos == 0) inspect_tensor(x, "1. Token Embedding (Start)");

    // ===== PARANDUS v3: matmul_q4_0_inplace + eelnevalt allokeeritud scratch-puhvrid =====
    // v2 kaotas attention allocation'id ja parallelize'is head-tsükli, aga jättis 7 matmul_q4_0
    // kutset (Q,K,V,O,W1,W3,W2), mis IGA KORD lõi uue Tensori (154x/token). Samuti kaotati
    // asjatud 1D->2D reshape-koopiad (xb_2d, attn_2d, ffn_2d), kuna inplace võtab otse pointeri.
    // Kõik puhvrid on thread_local static - kasvatatakse vajadusel, ei looda uuesti iga tokeni kohta.
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
    // ===== PARANDUSE ALGUS LÕPPENUD, ülejäänud loogika kihi-tsükli sees allpool =====

    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = model.layers[l];

        Tensor xb = tensor_math_transformer::rmsnorm(x, lw.attention_norm);

        if (pos == 0 && l == 0) inspect_tensor(xb, "2. RMSNorm Out (Kiht 0)");

        // Q, K, V - otse xb.data() pealt, tulemus kirjutatakse eelallokeeritud puhvritesse
        tensor_math_quantized::matmul_q4_0_inplace(lw.wq.data(), xb.data(), q_buf.data(), dim, dim);
        tensor_math_quantized::matmul_q4_0_inplace(lw.wk.data(), xb.data(), k_buf.data(), kv_dim, dim);
        tensor_math_quantized::matmul_q4_0_inplace(lw.wv.data(), xb.data(), v_buf.data(), kv_dim, dim);

        if (pos == 0 && l == 0) {
            Tensor q_dbg({dim}, Device::CPU);
            std::copy(q_buf.data(), q_buf.data() + dim, q_dbg.data());
            inspect_tensor(q_dbg, "3. Matmul Q4_0 WQ (Kiht 0)");
        }

        // v (KV cache jaoks Tensor vajalik, koopia siin on vältimatu)
        Tensor v({kv_dim}, Device::CPU);
        std::copy(v_buf.data(), v_buf.data() + kv_dim, v.data());

        Tensor q_rot({dim}, Device::CPU);
        Tensor k_rot({kv_dim}, Device::CPU);

        // ===== PARANDUS v4: RoPE otse q_buf/k_buf pealt, ilma per-head Tensor allocationita =====
        // Varem: iga head kohta 2 Tensorit (q_head + q_head_rot), kokku 32*2 + 4*2 = 72
        // allocation'it kihi kohta. Nüüd: rope_inplace_adjacent loeb otse q_buf/k_buf pointerist
        // ja kirjutab otse q_rot/k_rot pointerisse, matemaatika identne rope(..., false) harule.
        for (int h = 0; h < num_heads; ++h) {
            tensor_math_transformer::rope_inplace_adjacent(
                q_buf.data() + h * head_dim, q_rot.data() + h * head_dim, head_dim, pos);
        }
        for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
            tensor_math_transformer::rope_inplace_adjacent(
                k_buf.data() + kvh * head_dim, k_rot.data() + kvh * head_dim, head_dim, pos);
        }
        // ===== PARANDUSE LÕPP =====

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
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_head_ptr[d] * k_head_ptr[d];
                }
                local_scores[t] = dot * scale;
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

        // Attention output projection (WO) - inplace, otse attn_concat.data() pealt
        tensor_math_quantized::matmul_q4_0_inplace(lw.wo.data(), attn_concat.data(), attn_out_buf.data(), dim, dim);

        for (int i = 0; i < dim; ++i) {
            x.data()[i] += attn_out_buf[i];
        }

        Tensor ffn_norm = tensor_math_transformer::rmsnorm(x, lw.ffn_norm);

        // FFN W1, W3 - inplace, otse ffn_norm.data() pealt
        tensor_math_quantized::matmul_q4_0_inplace(lw.w1.data(), ffn_norm.data(), w1_buf.data(), hidden_dim, dim);
        tensor_math_quantized::matmul_q4_0_inplace(lw.w3.data(), ffn_norm.data(), w3_buf.data(), hidden_dim, dim);

        Tensor swiglu_in({hidden_dim * 2}, Device::CPU);
        std::copy(w1_buf.data(), w1_buf.data() + hidden_dim, swiglu_in.data());
        std::copy(w3_buf.data(), w3_buf.data() + hidden_dim, swiglu_in.data() + hidden_dim);

        Tensor swiglu_out = tensor_math_transformer::swiglu(swiglu_in);

        // FFN W2 - inplace, otse swiglu_out.data() pealt
        tensor_math_quantized::matmul_q4_0_inplace(lw.w2.data(), swiglu_out.data(), ffn_final_buf.data(), dim, hidden_dim);

        for (int i = 0; i < dim; ++i) {
            x.data()[i] += ffn_final_buf[i];
        }
    }

    Tensor final_hidden = tensor_math_transformer::rmsnorm(x, model.final_norm);

    float* out_ptr = logits.data();
    const float* w_ptr = model.output_weights.data();
    const float* x_ptr = final_hidden.data();

#pragma omp parallel for if(vocab_size_safe(cfg) > 1000)
    for (int i = 0; i < vocab_size_safe(cfg); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        out_ptr[i] = sum;
    }

    if (pos == 0) inspect_tensor(logits, "4. Lõplikud Logits");
}

int sample_token(float* logits, int vocab_size, float temperature, float topp, float rep_penalty, const std::vector<int>& recent_tokens, std::mt19937& rng) {
    if (rep_penalty != 1.0f) {
        for (int token : recent_tokens) {
            if (token >= 0 && token < vocab_size) {
                if (logits[token] < 0) {
                    logits[token] *= rep_penalty;
                } else {
                    logits[token] /= rep_penalty;
                }
            }
        }
    }

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
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

#ifdef _OPENMP
    omp_set_num_threads(8); // Ryzen 7700 füüsiliste tuumade arv - hoiab ära 16-thread sync overhead'i
#endif

    std::cout << "=== Sovereign Kernel: GGUF Quantized Runner (model_runner2) ===\n";

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
        std::string gguf_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
        std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";
        std::string config_path = "runtime_config.txt";

        std::cout << "Loading GGUF model from: " << gguf_path << "\n";

        ModelConfig config;
        config.dim = 2048;
        config.hidden_dim = 5632;
        config.num_layers = 22;
        config.num_heads = 32;
        config.num_kv_heads = 4;
        config.vocab_size = 32000;
        config.seq_len = 256;

        TransformerModel model(config);

        load_gguf_model(gguf_path, model);
        std::cout << "GGUF weights loaded and mapped successfully!\n";

        Tokenizer tokenizer;
        tokenizer.load(tokenizer_path, config.vocab_size);
        std::cout << "Tokenizer initialized.\n\n========================================\n";
        std::cout << " GGUF CLI Ready (Configured via runtime_config.txt)\n========================================\n\n";

        std::random_device rd;
        std::mt19937 rng(rd());

        std::string user_input;
        while (true) {
            std::cout << "\n[Darth SulxX] > ";
            if (!std::getline(std::cin, user_input)) break;
            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;

            RuntimeConfig rcfg = load_runtime_config(config_path);

            std::vector<int> tokens = tokenizer.encode(user_input);

            std::cout << "\n[Misha GGUF Output]: " << std::flush;

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

            int generated_token_count = 0;
            double total_forward_ms = 0.0;
            auto gen_start = std::chrono::high_resolution_clock::now();

            std::vector<int> recent_tokens;

            for (int step = 0; step < rcfg.max_tokens; ++step) {
                int next_token = sample_token(logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);

                if (next_token == 2 && generated_token_count < rcfg.min_tokens) {
                    logits.data()[2] = -1e9f;
                    next_token = sample_token(logits.data(), config.vocab_size, rcfg.temperature, rcfg.top_p, rcfg.rep_penalty, recent_tokens, rng);
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
        std::cerr << "CRITICAL GGUF TENSOR ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}