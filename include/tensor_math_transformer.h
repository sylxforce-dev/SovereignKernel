#pragma once
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tensor_math_transformer {
    inline Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-5f) {
        const auto& shape = x.shape();
        if (shape.size() != 1) throw std::runtime_error("rmsnorm: input must be 1D");
        if (x.num_elements() != weight.num_elements()) throw std::runtime_error("rmsnorm: weight shape must match");

        size_t n = x.num_elements();
        const float* xd = x.data();
        const float* wd = weight.data();

        float sum_sq = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum_sq += xd[i] * xd[i];
        }
        float mean_sq = sum_sq / static_cast<float>(n);
        float rms = std::sqrt(mean_sq + eps);

        Tensor result(shape, Device::CPU);
        float* rd = result.data();
        for (size_t i = 0; i < n; ++i) {
            rd[i] = (xd[i] / rms) * wd[i];
        }
        return result;
    }

    inline Tensor softmax(const Tensor& x) {
        const auto& shape = x.shape();
        if (shape.size() != 1) throw std::runtime_error("softmax: input must be 1D");

        size_t n = x.num_elements();
        const float* xd = x.data();

        float max_val = xd[0];
        for (size_t i = 1; i < n; ++i) {
            max_val = std::max(max_val, xd[i]);
        }

        Tensor result(shape, Device::CPU);
        float* rd = result.data();

        float sum_exp = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            rd[i] = std::exp(xd[i] - max_val);
            sum_exp += rd[i];
        }
        for (size_t i = 0; i < n; ++i) {
            rd[i] /= sum_exp;
        }
        return result;
    }

    inline Tensor gelu(const Tensor& x) {
        const auto& shape = x.shape();
        size_t n = x.num_elements();
        const float* xd = x.data();
        Tensor result(shape, Device::CPU);
        float* rd = result.data();
        const float sqrt_2_over_pi = 0.7978845608028654f;

        for (size_t i = 0; i < n; ++i) {
            float val = xd[i];
            float inner = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
            rd[i] = 0.5f * val * (1.0f + std::tanh(inner));
        }
        return result;
    }

    // SUPPORTS MÜÜRISEIN: Dual-Mode RoPE Selector (HuggingFace split vs Karpathy adjacent)
    // Misha Runtime valib automaatselt sobivaima matemaatilise tee vastavalt mudeli struktuurile.
    inline Tensor rope(const Tensor& x, int position, float base = 10000.0f, bool use_hf_split = true) {
        const auto& shape = x.shape();
        if (shape.size() != 1) throw std::runtime_error("rope: input must be 1D");

        size_t n = x.num_elements();
        if (n % 2 != 0) throw std::runtime_error("rope: even element count required");

        const float* xd = x.data();
        Tensor result(shape, Device::CPU);
        float* rd = result.data();

        if (use_hf_split) {
            // Režiim 1: HuggingFace / .safetensors poolitatud paarid (Standard TinyLlama / Llama 2 jaoks)
            size_t half = n / 2;
            for (size_t i = 0; i < half; ++i) {
                float freq = 1.0f / std::pow(base, static_cast<float>(i * 2) / static_cast<float>(n));
                float val = static_cast<float>(position) * freq;

                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float x0 = xd[i];
                float x1 = xd[i + half];

                rd[i]        = x0 * cos_val - x1 * sin_val;
                rd[i + half] = x0 * sin_val + x1 * cos_val;
            }
        } else {
            // Režiim 2: Karpathy kõrvuti asetsevad paarid ((0,1), (2,3))
            for (size_t i = 0; i < n; i += 2) {
                float freq = 1.0f / std::pow(base, static_cast<float>(i) / static_cast<float>(n));
                float val = static_cast<float>(position) * freq;

                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float x0 = xd[i];
                float x1 = xd[i + 1];

                rd[i]     = x0 * cos_val - x1 * sin_val;
                rd[i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
        return result;
    }

    // ==========================================================================
    // INPLACE VERSIOON - Režiim 2 (Karpathy kõrvuti asetsevad paarid) ainult,
    // kuna real_runner2.cpp kutsub alati rope(..., false). Matemaatika on 1:1
    // identne rope() Režiim 2 haruga - lihtsalt loeb otse in-pointerist ja
    // kirjutab otse out-pointerisse, ilma Tensori loomiseta. Kasutatakse per-head
    // RoPE rakendamiseks (32 q-head + 4 k-head kihi kohta), kus varem loodi
    // 2 uut Tensorit iga head'i kohta (allocation-torm).
    // in ja out võivad olla sama pointer (in-place ülekirjutus lubatud).
    inline void rope_inplace_adjacent(const float* in, float* out, size_t n, int position, float base = 10000.0f) {
        for (size_t i = 0; i < n; i += 2) {
            float freq = 1.0f / std::pow(base, static_cast<float>(i) / static_cast<float>(n));
            float val = static_cast<float>(position) * freq;

            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            float x0 = in[i];
            float x1 = in[i + 1];

            out[i]     = x0 * cos_val - x1 * sin_val;
            out[i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    // ==========================================================================
    // INPLACE VERSIOON - Režiim 1 (HuggingFace split-paarid), kuna real_runner.cpp
    // kutsub rope(q_head, pos) vaikeväärtusega use_hf_split=true. Matemaatika 1:1
    // identne rope() Režiim 1 haruga. in ja out võivad olla sama pointer.
    inline void rope_inplace_hf_split(const float* in, float* out, size_t n, int position, float base = 10000.0f) {
        size_t half = n / 2;
        for (size_t i = 0; i < half; ++i) {
            float freq = 1.0f / std::pow(base, static_cast<float>(i * 2) / static_cast<float>(n));
            float val = static_cast<float>(position) * freq;

            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            float x0 = in[i];
            float x1 = in[i + half];

            out[i]        = x0 * cos_val - x1 * sin_val;
            out[i + half] = x0 * sin_val + x1 * cos_val;
        }
    }

    inline Tensor swiglu(const Tensor& x) {
        const auto& shape = x.shape();
        if (shape.size() != 1) throw std::runtime_error("swiglu: input must be 1D");

        size_t n = x.num_elements();
        if (n % 2 != 0) throw std::runtime_error("swiglu: even element count required");

        size_t half = n / 2;
        const float* xd = x.data();
        Tensor result({static_cast<int>(half)}, Device::CPU);
        float* rd = result.data();

        for (size_t i = 0; i < half; ++i) {
            float a = xd[i];          // w1_out (gate)
            float b = xd[half + i];   // w3_out (up)

            // LLaMA 2 SiLU aktiivsus gate peal
            float silu_a = a / (1.0f + std::exp(-a));
            rd[i] = silu_a * b;
        }
        return result;
    }
}