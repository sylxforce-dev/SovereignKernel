#pragma once
#include "tensor.h"
#include "gguf_types.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tensor_math_quantized {

    inline float fp16_to_fp32(uint16_t h) {
        uint32_t w = (h & 0x7FFF) << 13;
        w += 0x38000000;
        w |= (h & 0x8000) << 16;
        union { uint32_t u; float f; } pun;
        pun.u = w;
        return pun.f;
    }

    inline void dequantize_q4_0_block(const block_q4_0& block, float* target) {
        float d = fp16_to_fp32(block.d);
        for (int i = 0; i < 16; ++i) {
            uint8_t byte_val = block.qs[i];
            int8_t v0 = (byte_val & 0x0F) - 8;
            int8_t v1 = (byte_val >> 4) - 8;
            target[i]      = static_cast<float>(v0) * d;
            target[i + 16] = static_cast<float>(v1) * d;
        }
    }

    inline Tensor matmul_q4_0(const void* quantized_weights, const Tensor& x, int M, int K) {
        // [Misha Fix]: Maatriks korrutatakse (M x K) * (K x N)
        // Kuna X on mälus [K, N], siis shape[0] on K ja shape[1] on N.
        int N = 1;
        int K_x = K;

        const auto& shape = x.shape();
        if (shape.size() == 1) {
            K_x = shape[0];
            N = 1;
        } else if (shape.size() == 2) {
            K_x = shape[0]; // Õige: K on esimene dimensioon (nt 2048)
            N = shape[1];   // Õige: N on teine dimensioon (nt 1)
        } else {
            throw std::runtime_error("matmul_q4_0: x peab olema 1D või 2D tensor");
        }

        if (K_x != K) {
            throw std::runtime_error("matmul_q4_0: dimension mismatch (K_x=" + std::to_string(K_x) + ", K=" + std::to_string(K) + ")");
        }

        // Tulemusmaatriks peab olema (M x N)
        std::vector<int> out_shape;
        if (shape.size() == 1) out_shape = {M};
        else out_shape = {M, N};

        Tensor result(out_shape, Device::CPU);
        float* C = result.data();
        const float* X = x.data();

        const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(quantized_weights);
        int blocks_per_row = K / 32;

        if (N == 1) {
            // ==========================================
            // FAST PATH (N=1) - AVX2
            // ==========================================
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                float sum = 0.0f;
                const block_q4_0* row_blocks = blocks + i * blocks_per_row;

#if defined(__AVX2__)
                __m256 vsum = _mm256_setzero_ps();
                __m128i low_mask  = _mm_set1_epi8(0x0F);
                __m128i offset    = _mm_set1_epi8(8);

                for (int b = 0; b < blocks_per_row; ++b) {
                    float d = fp16_to_fp32(row_blocks[b].d);
                    __m256 vd = _mm256_set1_ps(d);

                    const uint8_t* qs = row_blocks[b].qs;
                    const float* x_sub = X + b * 32;

                    __m128i raw_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                    __m128i bytes_low  = _mm_and_si128(raw_bytes, low_mask);
                    __m128i bytes_high = _mm_and_si128(_mm_srli_epi16(raw_bytes, 4), low_mask);

                    bytes_low  = _mm_sub_epi8(bytes_low, offset);
                    bytes_high = _mm_sub_epi8(bytes_high, offset);

                    __m256 v_low0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_low));
                    __m256 v_low1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes_low, 8)));
                    __m256 v_high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_high));
                    __m256 v_high1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes_high, 8)));

                    __m256 vx0 = _mm256_loadu_ps(x_sub + 0);
                    __m256 vx1 = _mm256_loadu_ps(x_sub + 8);
                    __m256 vx2 = _mm256_loadu_ps(x_sub + 16);
                    __m256 vx3 = _mm256_loadu_ps(x_sub + 24);

                    vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_low0, vd),  vx0, vsum);
                    vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_low1, vd),  vx1, vsum);
                    vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_high0, vd), vx2, vsum);
                    vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_high1, vd), vx3, vsum);
                }

                __m128 vlow  = _mm256_castps256_ps128(vsum);
                __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
                __m128 vrc   = _mm_add_ps(vlow, vhigh);
                vrc = _mm_hadd_ps(vrc, vrc);
                vrc = _mm_hadd_ps(vrc, vrc);

                float final_sum;
                _mm_store_ss(&final_sum, vrc);
                sum = final_sum;
#else
                for (int b = 0; b < blocks_per_row; ++b) {
                    float d = fp16_to_fp32(row_blocks[b].d);
                    const uint8_t* qs = row_blocks[b].qs;
                    const float* x_sub = X + b * 32;

                    for (int j = 0; j < 16; ++j) {
                        uint8_t byte_val = qs[j];
                        float v0 = static_cast<float>((byte_val & 0x0F) - 8) * d;
                        float v1 = static_cast<float>((byte_val >> 4) - 8) * d;
                        sum += v0 * x_sub[j] + v1 * x_sub[j + 16];
                    }
                }
#endif
                C[i] = sum;
            }
        } else {
            // ==========================================
            // SLOW PATH (N>1) - Prompt seedimine
            // ==========================================
            std::fill(C, C + M * N, 0.0f);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                const block_q4_0* row_blocks = blocks + i * blocks_per_row;
                float* rowC = C + i * N;

                for (int k = 0; k < K; k += 32) {
                    int b = k / 32;
                    float d = fp16_to_fp32(row_blocks[b].d);
                    const uint8_t* qs = row_blocks[b].qs;

                    for (int j = 0; j < 16; ++j) {
                        uint8_t byte_val = qs[j];
                        float v0 = (static_cast<float>(byte_val & 0x0F) - 8.0f) * d;
                        float v1 = (static_cast<float>(byte_val >> 4) - 8.0f) * d;

                        const float* rowX0 = X + (k + j) * N;
                        const float* rowX1 = X + (k + j + 16) * N;

                        for (int n = 0; n < N; ++n) {
                            rowC[n] += v0 * rowX0[n] + v1 * rowX1[n];
                        }
                    }
                }
            }
        }

        return result;
    }

    // ==========================================================================
    // INPLACE VERSIOON (N=1, generation-loop jaoks)
    // Sama AVX2 matemaatika mis matmul_q4_0 fast path, aga EI LOO uut Tensorit -
    // kirjutab tulemuse otse ette antud C puhvrisse. Kutsutakse 7x/kiht x 22 kihti
    // = 154x tokeni kohta, nii et allocation-kõrvaldamine siin annab suurima efekti
    // matmul-koguarvutusest. C peab olema vähemalt M float suurune, eelnevalt
    // allokeeritud (nt kihi-tsükli algul üks kord, mitte iga kutsega).
    inline void matmul_q4_0_inplace(const void* quantized_weights, const float* X, float* C, int M, int K) {
        const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(quantized_weights);
        int blocks_per_row = K / 32;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            float sum = 0.0f;
            const block_q4_0* row_blocks = blocks + i * blocks_per_row;

#if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            __m128i low_mask  = _mm_set1_epi8(0x0F);
            __m128i offset    = _mm_set1_epi8(8);

            for (int b = 0; b < blocks_per_row; ++b) {
                float d = fp16_to_fp32(row_blocks[b].d);
                __m256 vd = _mm256_set1_ps(d);

                const uint8_t* qs = row_blocks[b].qs;
                const float* x_sub = X + b * 32;

                __m128i raw_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i bytes_low  = _mm_and_si128(raw_bytes, low_mask);
                __m128i bytes_high = _mm_and_si128(_mm_srli_epi16(raw_bytes, 4), low_mask);

                bytes_low  = _mm_sub_epi8(bytes_low, offset);
                bytes_high = _mm_sub_epi8(bytes_high, offset);

                __m256 v_low0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_low));
                __m256 v_low1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes_low, 8)));
                __m256 v_high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_high));
                __m256 v_high1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes_high, 8)));

                __m256 vx0 = _mm256_loadu_ps(x_sub + 0);
                __m256 vx1 = _mm256_loadu_ps(x_sub + 8);
                __m256 vx2 = _mm256_loadu_ps(x_sub + 16);
                __m256 vx3 = _mm256_loadu_ps(x_sub + 24);

                vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_low0, vd),  vx0, vsum);
                vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_low1, vd),  vx1, vsum);
                vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_high0, vd), vx2, vsum);
                vsum = _mm256_fmadd_ps(_mm256_mul_ps(v_high1, vd), vx3, vsum);
            }

            __m128 vlow  = _mm256_castps256_ps128(vsum);
            __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
            __m128 vrc   = _mm_add_ps(vlow, vhigh);
            vrc = _mm_hadd_ps(vrc, vrc);
            vrc = _mm_hadd_ps(vrc, vrc);

            float final_sum;
            _mm_store_ss(&final_sum, vrc);
            sum = final_sum;
#else
            for (int b = 0; b < blocks_per_row; ++b) {
                float d = fp16_to_fp32(row_blocks[b].d);
                const uint8_t* qs = row_blocks[b].qs;
                const float* x_sub = X + b * 32;

                for (int j = 0; j < 16; ++j) {
                    uint8_t byte_val = qs[j];
                    float v0 = static_cast<float>((byte_val & 0x0F) - 8) * d;
                    float v1 = static_cast<float>((byte_val >> 4) - 8) * d;
                    sum += v0 * x_sub[j] + v1 * x_sub[j + 16];
                }
            }
#endif
            C[i] = sum;
        }
    }

} // namespace tensor_math_quantized