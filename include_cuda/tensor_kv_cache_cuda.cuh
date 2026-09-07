#pragma once
#include "tensor.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// Phase 6 — Tensor KV Cache Manager (CUDA Counterpart for testing)
// Praeguses "basic" faasis hoiame andmeid VRAM-is, aga lubame append()
// operatsioonis host->device koopiat, et testida loogikat. Lõplikus (VRAM-only)
// faasis on siin suured pre-allocated bufferid ja append on device-to-device.

namespace tensor_kv_cache_cuda {
    class TensorKVCacheCUDA {
    private:
        std::vector<Tensor> key_history_;
        std::vector<Tensor> value_history_;
        size_t max_seq_len_;

    public:
        explicit TensorKVCacheCUDA(size_t max_seq_len = 512) : max_seq_len_(max_seq_len) {
            key_history_.reserve(max_seq_len_);
            value_history_.reserve(max_seq_len_);
        }

        // Testimiseks: võtab sisse CPU tensorid, aga hoiab CUDA tensoritena
        void append(const Tensor& k_host, const Tensor& v_host) {
            if (key_history_.size() >= max_seq_len_) {
                throw std::runtime_error("TensorKVCacheCUDA: max sequence length exceeded");
            }

            if (k_host.device() != Device::CPU || v_host.device() != Device::CPU) {
                 throw std::runtime_error("TensorKVCacheCUDA::append: basic wrapper expects CPU tensors as input");
            }

            Tensor k_dev(k_host.shape(), Device::CUDA);
            Tensor v_dev(v_host.shape(), Device::CUDA);

            size_t bytes = k_host.num_elements() * sizeof(float);

            cudaError_t err = cudaMemcpy(k_dev.data(), k_host.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) throw std::runtime_error("KVCacheCUDA H2D copy failed for K");

            err = cudaMemcpy(v_dev.data(), v_host.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) throw std::runtime_error("KVCacheCUDA H2D copy failed for V");

            key_history_.push_back(std::move(k_dev));
            value_history_.push_back(std::move(v_dev));
        }

        size_t size() const { return key_history_.size(); }

        const std::vector<Tensor>& keys() const { return key_history_; }
        const std::vector<Tensor>& values() const { return value_history_; }

        void clear() {
            key_history_.clear();
            value_history_.clear();
        }
    };
}