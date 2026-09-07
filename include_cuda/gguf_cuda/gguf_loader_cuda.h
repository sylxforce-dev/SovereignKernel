#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

#include "gguf_reader_cuda.h"
#include "gguf_types_cuda.h"
#include "model_config.h"

namespace gguf_host_utils {
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

    #pragma pack(push, 1)
    struct block_q6_K {
        uint8_t ql[128];
        uint8_t qh[64];
        int8_t  scales[16];
        uint16_t d;
    };
    #pragma pack(pop)

    inline void dequantize_q6_k_block(const block_q6_K* x, float* y) {
        float d = fp16_to_fp32(x->d);
        const uint8_t* ql = x->ql;
        const uint8_t* qh = x->qh;
        const int8_t* sc = x->scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF)  | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 0] >> 4)   | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

struct CUDALayerWeightsQ4 {
    float* attention_norm = nullptr;
    void*  wq = nullptr;
    void*  wk = nullptr;
    void*  wv = nullptr;
    void*  wo = nullptr;
    float* ffn_norm = nullptr;
    void*  w1 = nullptr;
    void*  w2 = nullptr;
    void*  w3 = nullptr;
};

struct CUDATransformerModelQ4 {
    ModelConfig config;
    float* token_embedding_table = nullptr;
    std::vector<CUDALayerWeightsQ4> layers;
    float* final_norm = nullptr;
    float* output_weights = nullptr;
};

class GGUFLoaderCUDA {
private:
    static void* upload_raw_to_vram(const void* host_data, size_t bytes) {
        void* dev_ptr = nullptr;
        cudaError_t err = cudaMalloc(&dev_ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed (VRAM Out of Memory?): " + std::string(cudaGetErrorString(err)));
        }
        cudaMemcpy(dev_ptr, host_data, bytes, cudaMemcpyHostToDevice);
        return dev_ptr;
    }

    static float* upload_fp32_to_vram(const float* host_data, size_t num_elements) {
        return static_cast<float*>(upload_raw_to_vram(host_data, num_elements * sizeof(float)));
    }

public:
    static void load_weights_to_vram(const std::string& filepath, CUDATransformerModelQ4& cuda_model) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[GGUF VRAM Loader] Ei suutnud avada faili: " + filepath);
        }

        GGUFReader reader;
        reader.load(filepath);

        std::cout << "[GGUF VRAM Loader] Analüüsin ja laen kaale VRAM-i...\n";

        std::unordered_map<std::string, GGUFTensorInfo> tensor_map;
        for (const auto& t : reader.tensors) {
            tensor_map[t.name] = t;
            // Diagnostika: trükime välja tähtsamad tensorid, et näha tüüpe
            if (t.name == "token_embd.weight" || t.name == "output.weight" || t.name.find("blk.0.") != std::string::npos) {
                std::cout << "  [Tensor Audit] " << t.name << " | Tüüp: " << t.type << " | Dims: ";
                for (auto d : t.dimensions) std::cout << d << " ";
                std::cout << "\n";
            }
        }

        auto read_tensor_raw = [&](const std::string& name, size_t expected_bytes) -> std::vector<char> {
            auto it = tensor_map.find(name);
            if (it == tensor_map.end()) throw std::runtime_error("Tensorit ei leitud: " + name);

            std::vector<char> buffer(expected_bytes);
            uint64_t absolute_offset = reader.data_offset + it->second.offset;
            file.seekg(absolute_offset, std::ios::beg);
            file.read(buffer.data(), expected_bytes);
            return buffer;
        };

        const auto& cfg = cuda_model.config;
        cuda_model.layers.resize(cfg.num_layers);

        // 1. Token Embeddings
        auto it_embd = tensor_map.find("token_embd.weight");
        if (it_embd == tensor_map.end()) throw std::runtime_error("token_embd.weight puudub!");

        size_t embd_elements = cfg.vocab_size * cfg.dim;
        std::vector<float> embd_fp32(embd_elements);

        if (it_embd->second.type == GGML_TYPE_F32) {
            auto raw = read_tensor_raw("token_embd.weight", embd_elements * sizeof(float));
            std::memcpy(embd_fp32.data(), raw.data(), raw.size());
        } else if (it_embd->second.type == GGML_TYPE_Q4_0) {
            size_t bytes = (embd_elements / 32) * sizeof(block_q4_0);
            auto raw = read_tensor_raw("token_embd.weight", bytes);
            block_q4_0* blocks = reinterpret_cast<block_q4_0*>(raw.data());
            for (size_t b = 0; b < embd_elements / 32; ++b) {
                gguf_host_utils::dequantize_q4_0_block(blocks[b], embd_fp32.data() + b * 32);
            }
        } else {
            throw std::runtime_error("Tundmatu token_embd.weight tüüp!");
        }
        cuda_model.token_embedding_table = upload_fp32_to_vram(embd_fp32.data(), embd_elements);

        // 2. Hidden Layers
        size_t dim = cfg.dim;
        size_t hidden_dim = cfg.hidden_dim;
        size_t q_bytes = (dim * dim) / 32 * sizeof(block_q4_0);
        size_t kv_dim = (dim / cfg.num_heads) * cfg.num_kv_heads;
        size_t kv_bytes = (dim * kv_dim) / 32 * sizeof(block_q4_0);
        size_t up_down_bytes = (dim * hidden_dim) / 32 * sizeof(block_q4_0);
        size_t norm_bytes = dim * sizeof(float);

        for (int l = 0; l < cfg.num_layers; ++l) {
            auto& lw = cuda_model.layers[l];
            std::string p = "blk." + std::to_string(l) + ".";

            lw.wq = upload_raw_to_vram(read_tensor_raw(p + "attn_q.weight", q_bytes).data(), q_bytes);
            lw.wk = upload_raw_to_vram(read_tensor_raw(p + "attn_k.weight", kv_bytes).data(), kv_bytes);
            lw.wv = upload_raw_to_vram(read_tensor_raw(p + "attn_v.weight", kv_bytes).data(), kv_bytes);
            lw.wo = upload_raw_to_vram(read_tensor_raw(p + "attn_output.weight", q_bytes).data(), q_bytes);

            lw.w1 = upload_raw_to_vram(read_tensor_raw(p + "ffn_gate.weight", up_down_bytes).data(), up_down_bytes);
            lw.w2 = upload_raw_to_vram(read_tensor_raw(p + "ffn_down.weight", up_down_bytes).data(), up_down_bytes);
            lw.w3 = upload_raw_to_vram(read_tensor_raw(p + "ffn_up.weight", up_down_bytes).data(), up_down_bytes);

            lw.attention_norm = static_cast<float*>(upload_raw_to_vram(read_tensor_raw(p + "attn_norm.weight", norm_bytes).data(), norm_bytes));
            lw.ffn_norm = static_cast<float*>(upload_raw_to_vram(read_tensor_raw(p + "ffn_norm.weight", norm_bytes).data(), norm_bytes));
        }

        // 3. Final Norm ja Output Weights
        cuda_model.final_norm = static_cast<float*>(upload_raw_to_vram(read_tensor_raw("output_norm.weight", norm_bytes).data(), norm_bytes));

        auto it_out = tensor_map.find("output.weight");
        if (it_out != tensor_map.end()) {
            size_t out_elements = cfg.vocab_size * cfg.dim;
            std::vector<float> out_fp32(out_elements);
            uint32_t out_type = it_out->second.type;

            std::cout << "[GGUF VRAM Loader] output.weight leitud! Tüüp: " << out_type << "\n";

            if (out_type == 14) { // GGML_TYPE_Q6_K
                size_t bytes = (out_elements / 256) * sizeof(gguf_host_utils::block_q6_K);
                auto raw = read_tensor_raw("output.weight", bytes);
                gguf_host_utils::block_q6_K* blocks = reinterpret_cast<gguf_host_utils::block_q6_K*>(raw.data());
                for (size_t b = 0; b < out_elements / 256; ++b) {
                    gguf_host_utils::dequantize_q6_k_block(&blocks[b], out_fp32.data() + b * 256);
                }
            } else if (out_type == GGML_TYPE_Q4_0) {
                size_t bytes = (out_elements / 32) * sizeof(block_q4_0);
                auto raw = read_tensor_raw("output.weight", bytes);
                block_q4_0* blocks = reinterpret_cast<block_q4_0*>(raw.data());
                for (size_t b = 0; b < out_elements / 32; ++b) {
                    gguf_host_utils::dequantize_q4_0_block(blocks[b], out_fp32.data() + b * 32);
                }
            } else if (out_type == GGML_TYPE_F32) {
                auto raw = read_tensor_raw("output.weight", out_elements * sizeof(float));
                std::memcpy(out_fp32.data(), raw.data(), raw.size());
            } else {
                throw std::runtime_error("output.weight tüüpi ei toetata!");
            }
            cuda_model.output_weights = upload_fp32_to_vram(out_fp32.data(), out_elements);
        } else {
            std::cout << "[GGUF VRAM Loader] HOIATUS: output.weight puudub, seon output token_embedding_table'iga!\n";
            cuda_model.output_weights = cuda_model.token_embedding_table;
        }

        std::cout << "[GGUF VRAM Loader] Kõik kaalud edukalt VRAM-is ankurdatud!\n";
    }
};