#pragma once
#include "tensor.h"
#include "tensor_math_cpu.h"
#include "tensor_math_transformer.h"
#include <cmath>
#include <stdexcept>
#include <vector>

// Phase 5 — Multi-Head Attention & Attention Block (CPU Reference)
//
// Seob senised primitiivid (matmul, RoPE, Softmax) üheks täielikuks
// tähelepanu (Attention) arvutuskäiguks:
//   1. Projektsioonid Q, K, V (kasutades matmul'it)
//   2. RoPE (Rotary Position Embedding) rakendamine Q ja K peale
//   3. Attention Scores: Softmax( (Q * K^T) / sqrt(d_k) ) * V
namespace tensor_math_attention {

    // Scaled Dot-Product Attention (Single Head versioon selguse ja baasi jaoks)
    // Q, K, V on 2D maatriksid (kuju: [seq_len, head_dim])
    inline Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float scale_factor) {
        const auto& q_shape = Q.shape();
        const auto& k_shape = K.shape();
        const auto& v_shape = V.shape();

        if (q_shape.size() != 2 || k_shape.size() != 2 || v_shape.size() != 2) {
            throw std::runtime_error("attention: Q, K, V must be 2D matrices");
        }

        int seq_len_q = q_shape[0];
        int d_k = q_shape[1];
        int seq_len_k = k_shape[1]; // K on transponeeritud või õigetpidi? Vaatame K kuju: [seq_len_k, d_k]

        // 1. Arvutame Q * K^T
        // Kuna K on [seq_len_k, d_k], siis K transponeerituna on [d_k, seq_len_k]
        // Q on [seq_len_q, d_k]. Korrutis (Q) * (K^T) annab [seq_len_q, seq_len_k]
        Tensor K_t = tensor_math_cpu::transpose(K);
        Tensor scores = tensor_math_cpu::matmul(Q, K_t);

        // 2. Skaleerimine (divide by sqrt(d_k)) ja Softmax rea kaupa
        size_t rows = scores.shape()[0];
        size_t cols = scores.shape()[1];
        float* s_data = scores.data();

        for (size_t i = 0; i < rows; ++i) {
            // Võtame ühe rea (1D tensor Softmaxi jaoks)
            std::vector<int> row_shape = { static_cast<int>(cols) };
            Tensor row_tensor(row_shape, Device::CPU);
            for (size_t j = 0; j < cols; ++j) {
                row_tensor.data()[j] = (s_data[i * cols + j] / scale_factor);
            }

            // Rakendame Softmaxi
            Tensor softmax_row = tensor_math_transformer::softmax(row_tensor);
            for (size_t j = 0; j < cols; ++j) {
                s_data[i * cols + j] = softmax_row.data()[j];
            }
        }

        // 3. Scores * V (Tulemus: [seq_len_q, d_k])
        Tensor output = tensor_math_cpu::matmul(scores, V);
        return output;
    }

    // --- UUS: Multi-Head Attention (MHA) Orkestraator ---
    // Võtab sisse täismõõdus maatriksid [seq_len, d_model], tükeldab need
    // n_heads arvuks peadeks, laseb läbi ülemise funktsiooni ja kleebib kokku.
    inline Tensor multi_head_attention(const Tensor& Q_full, const Tensor& K_full, const Tensor& V_full,
                                       int n_heads, float scale_factor) {

        const auto& q_shape = Q_full.shape();
        const auto& k_shape = K_full.shape();
        const auto& v_shape = V_full.shape();

        if (q_shape.size() != 2 || k_shape.size() != 2 || v_shape.size() != 2) {
            throw std::runtime_error("multi_head_attention: Q, K, V must be 2D matrices [seq_len, d_model]");
        }

        int seq_len = q_shape[0];
        int d_model = q_shape[1];

        if (d_model % n_heads != 0) {
            throw std::runtime_error("multi_head_attention: d_model (" + std::to_string(d_model) +
                                     ") must be divisible by n_heads (" + std::to_string(n_heads) + ")");
        }

        int head_dim = d_model / n_heads;

        // Eraldame mälu lõpp-tulemusele, kuhu pead tagasi kleebitakse
        Tensor output_full({seq_len, d_model}, Device::CPU);
        float* out_data = output_full.data();

        // Tsükkel üle kõigi tähelepanupeade
        for (int h = 0; h < n_heads; ++h) {

            // 1. Loome mälu aktiivse pea lõikudele
            Tensor Q_head({seq_len, head_dim}, Device::CPU);
            Tensor K_head({seq_len, head_dim}, Device::CPU);
            Tensor V_head({seq_len, head_dim}, Device::CPU);

            float* qh_data = Q_head.data();
            float* kh_data = K_head.data();
            float* vh_data = V_head.data();

            const float* qf_data = Q_full.data();
            const float* kf_data = K_full.data();
            const float* vf_data = V_full.data();

            // 2. Lõikame maatriksist (slice) andmed ainult selle pea jaoks
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    int full_idx = s * d_model + h * head_dim + d;
                    int head_idx = s * head_dim + d;

                    qh_data[head_idx] = qf_data[full_idx];
                    kh_data[head_idx] = kf_data[full_idx];
                    vh_data[head_idx] = vf_data[full_idx];
                }
            }

            // 3. Rakendame baasarvutust sellele spetsiifilisele peale
            Tensor head_out = scaled_dot_product_attention(Q_head, K_head, V_head, scale_factor);
            const float* ho_data = head_out.data();

            // 4. Kleebime (concatenate) tulemuse tagasi suure maatriksi õigesse sektsiooni
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    int full_idx = s * d_model + h * head_dim + d;
                    int head_idx = s * head_dim + d;

                    out_data[full_idx] = ho_data[head_idx];
                }
            }
        }

        return output_full;
    }

} // namespace tensor_math_attention