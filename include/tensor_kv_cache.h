#pragma once
#include "tensor.h"
#include <vector>
#include <stdexcept>
#include <cstring>

// Phase 6 — Tensor KV Cache Manager
class TensorKVCache {
private:
    std::vector<Tensor> key_history_;
    std::vector<Tensor> value_history_;
    size_t max_seq_len_;

public:
    explicit TensorKVCache(size_t max_seq_len = 512) : max_seq_len_(max_seq_len) {
        // --- OPTIMIZATION: Eel-allokeerime vektorite mahu, et vältida mälulohinaid ---
        key_history_.reserve(max_seq_len_);
        value_history_.reserve(max_seq_len_);
    }

    void append(const Tensor& k_tensor, const Tensor& v_tensor) {
        if (key_history_.size() >= max_seq_len_) {
            throw std::runtime_error("TensorKVCache: max sequence length exceeded");
        }

        // CPU-only for now — CUDA-backed KV cache (shared memory controller,
        // device-to-device copy) lives in a separate CUDA file, added later.
        if (k_tensor.device() != Device::CPU || v_tensor.device() != Device::CPU) {
            throw std::runtime_error("TensorKVCache::append: CUDA tensors not yet supported here — see CUDA backend (later)");
        }

        Tensor k_copy(k_tensor.shape(), Device::CPU);
        Tensor v_copy(v_tensor.shape(), Device::CPU);

        size_t bytes = k_tensor.num_elements() * sizeof(float);
        std::memcpy(k_copy.data(), k_tensor.data(), bytes);
        std::memcpy(v_copy.data(), v_tensor.data(), bytes);

        key_history_.push_back(std::move(k_copy));
        value_history_.push_back(std::move(v_copy));
    }

    size_t size() const { return key_history_.size(); }

    const std::vector<Tensor>& keys() const { return key_history_; }
    const std::vector<Tensor>& values() const { return value_history_; }

    void clear() {
        key_history_.clear();
        value_history_.clear();
    }
};