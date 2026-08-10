#pragma once
#include "tensor.h"
#include <stdexcept>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// AVX2 intrinsics toetus x86 riistvarale
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tensor_math_cpu {

    // --- Elementwise operations (Multithreaded with OpenMP) ---

    inline Tensor add(const Tensor& a, const Tensor& b) {
        if (a.num_elements() != b.num_elements()) {
            throw std::runtime_error("add: shape mismatch (element count differs)");
        }
        Tensor result(a.shape(), Device::CPU);
        size_t n = a.num_elements();
        const float* ad = a.data();
        const float* bd = b.data();
        float* rd = result.data();

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            rd[i] = ad[i] + bd[i];
        }
        return result;
    }

    inline Tensor subtract(const Tensor& a, const Tensor& b) {
        if (a.num_elements() != b.num_elements()) {
            throw std::runtime_error("subtract: shape mismatch (element count differs)");
        }
        Tensor result(a.shape(), Device::CPU);
        size_t n = a.num_elements();
        const float* ad = a.data();
        const float* bd = b.data();
        float* rd = result.data();

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            rd[i] = ad[i] - bd[i];
        }
        return result;
    }

    inline Tensor multiply(const Tensor& a, const Tensor& b) {
        if (a.num_elements() != b.num_elements()) {
            throw std::runtime_error("multiply: shape mismatch (element count differs)");
        }
        Tensor result(a.shape(), Device::CPU);
        size_t n = a.num_elements();
        const float* ad = a.data();
        const float* bd = b.data();
        float* rd = result.data();

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            rd[i] = ad[i] * bd[i];
        }
        return result;
    }

    inline Tensor divide(const Tensor& a, const Tensor& b) {
        if (a.num_elements() != b.num_elements()) {
            throw std::runtime_error("divide: shape mismatch (element count differs)");
        }
        size_t n = a.num_elements();
        for (size_t i = 0; i < n; ++i) {
            if (b.data()[i] == 0.0f) {
                throw std::runtime_error("divide: division by zero at element " + std::to_string(i));
            }
        }
        Tensor result(a.shape(), Device::CPU);
        const float* ad = a.data();
        const float* bd = b.data();
        float* rd = result.data();

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            rd[i] = ad[i] / bd[i];
        }
        return result;
    }

    // --- Matrix operations (2D tensors only) ---

    // Matrix multiplication: (M x K) * (K x N) = (M x N)
    inline Tensor matmul(const Tensor& a, const Tensor& b) {
        const auto& shape_a = a.shape();
        const auto& shape_b = b.shape();

        if (shape_a.size() != 2 || shape_b.size() != 2) {
            throw std::runtime_error("matmul: both tensors must be 2D");
        }

        int M = shape_a[0];
        int K = shape_a[1];
        int K2 = shape_b[0];
        int N = shape_b[1];

        if (K != K2) {
            throw std::runtime_error("matmul: inner dimensions must match (got "
                + std::to_string(K) + " vs " + std::to_string(K2) + ")");
        }

        Tensor result({M, N}, Device::CPU);
        const float* A = a.data();
        const float* B = b.data();
        float* C = result.data();

        // Nullime tulemuse algväärtused
        std::fill(C, C + M * N, 0.0f);

        // --- UUS: GEMV Fast-Path (N == 1) ---
        // LLM genereerimise ajal on sisendiks alati N=1 vektor. Vektoriseerime üle K dimensiooni.
        if (N == 1) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                const float* rowA = A + i * K;
                float sum = 0.0f;
                int k = 0;
#if defined(__AVX2__)
                __m256 v_sum = _mm256_setzero_ps();
                for (; k <= K - 8; k += 8) {
                    __m256 va = _mm256_loadu_ps(rowA + k);
                    __m256 vb = _mm256_loadu_ps(B + k);
                    v_sum = _mm256_fmadd_ps(va, vb, v_sum);
                }
                alignas(32) float buffer[8];
                _mm256_storeu_ps(buffer, v_sum);
                for (int b = 0; b < 8; ++b) {
                    sum += buffer[b];
                }
#endif
                for (; k < K; ++k) {
                    sum += rowA[k] * B[k];
                }
                C[i] = sum;
            }
            return result;
        }
        // --- GEMV Fast-Path Lõpp ---

        // Vana kood (GEMM) läheb edasi N > 1 jaoks (nt Prompt Processing / Batching)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            const float* rowA = A + i * K;
            float* rowC = C + i * N;

            for (int k = 0; k < K; ++k) {
                float a_ik = rowA[k];
                const float* rowB = B + k * N;
                int j = 0;

#if defined(__AVX2__)
                __m256 v_aik = _mm256_set1_ps(a_ik);
                for (; j <= N - 8; j += 8) {
                    __m256 v_b = _mm256_loadu_ps(rowB + j);
                    __m256 v_c = _mm256_loadu_ps(rowC + j);
                    v_c = _mm256_fmadd_ps(v_aik, v_b, v_c);
                    _mm256_storeu_ps(rowC + j, v_c);
                }
#endif
                for (; j < N; ++j) {
                    rowC[j] += a_ik * rowB[j];
                }
            }
        }

        return result;
    }

    // Transpose a 2D tensor: (M x N) -> (N x M)
    inline Tensor transpose(const Tensor& a) {
        const auto& shape = a.shape();
        if (shape.size() != 2) {
            throw std::runtime_error("transpose: tensor must be 2D");
        }

        int M = shape[0];
        int N = shape[1];

        Tensor result({N, M}, Device::CPU);
        const float* src = a.data();
        float* dst = result.data();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                dst[j * M + i] = src[i * N + j];
            }
        }

        return result;
    }

    // Dot product: two 1D tensors of equal length -> scalar
    inline float dot(const Tensor& a, const Tensor& b) {
        const auto& shape_a = a.shape();
        const auto& shape_b = b.shape();

        if (shape_a.size() != 1 || shape_b.size() != 1) {
            throw std::runtime_error("dot: both tensors must be 1D");
        }
        if (a.num_elements() != b.num_elements()) {
            throw std::runtime_error("dot: length mismatch");
        }

        size_t n = a.num_elements();
        const float* ad = a.data();
        const float* bd = b.data();
        float sum = 0.0f;

#if defined(__AVX2__)
        #pragma omp parallel reduction(+:sum)
        {
            __m256 v_sum = _mm256_setzero_ps();
            #pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n - (n % 8)); i += 8) {
                __m256 va = _mm256_loadu_ps(ad + i);
                __m256 vb = _mm256_loadu_ps(bd + i);
                v_sum = _mm256_fmadd_ps(va, vb, v_sum);
            }
            alignas(32) float buffer[8];
            _mm256_storeu_ps(buffer, v_sum);
            for (int k = 0; k < 8; ++k) {
                sum += buffer[k];
            }
        }

        for (size_t i = n - (n % 8); i < n; ++i) {
            sum += ad[i] * bd[i];
        }
#else
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            sum += ad[i] * bd[i];
        }
#endif

        return sum;
    }

    // ==========================================================================
    // INPLACE VERSIOON (N=1, generation-loop jaoks)
    // Sama AVX2 GEMV matemaatika mis matmul() fast path, aga EI LOO uut Tensorit -
    // kirjutab tulemuse otse ette antud C puhvrisse. B on lihtne float pointer
    // (mitte Tensor), kuna kutsuja pool on juba plain buffer (xb.data() jms).
    // C peab olema vähemalt M float suurune, eelnevalt allokeeritud.
    inline void matmul_inplace(const Tensor& a, const float* B, float* C, int M, int K) {
        const auto& shape_a = a.shape();
        if (shape_a.size() != 2) {
            throw std::runtime_error("matmul_inplace: weight tensor must be 2D");
        }
        if (shape_a[1] != K) {
            throw std::runtime_error("matmul_inplace: dimension mismatch");
        }

        const float* A = a.data();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            const float* rowA = A + static_cast<size_t>(i) * K;
            float sum = 0.0f;
            int k = 0;
#if defined(__AVX2__)
            __m256 v_sum = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 va = _mm256_loadu_ps(rowA + k);
                __m256 vb = _mm256_loadu_ps(B + k);
                v_sum = _mm256_fmadd_ps(va, vb, v_sum);
            }
            alignas(32) float buffer[8];
            _mm256_storeu_ps(buffer, v_sum);
            for (int b = 0; b < 8; ++b) {
                sum += buffer[b];
            }
#endif
            for (; k < K; ++k) {
                sum += rowA[k] * B[k];
            }
            C[i] = sum;
        }
    }

} // namespace tensor_math_cpu