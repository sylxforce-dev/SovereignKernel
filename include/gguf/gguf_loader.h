
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "gguf_reader.h"
#include "gguf_types.h"
#include "model.h"
#include "tensor_math_quantized.h" // SOVEREIGN KERNEL FIX: Lisatud dekvantiseerimiseks

class GGUFModelLoader {
public:
    static void load_weights(const std::string& filepath, TransformerModel& model) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[GGUF Loader] Ei suutnud avada faili lugemiseks: " + filepath);
        }

        GGUFReader reader;
        reader.load(filepath);

        std::cout << "[GGUF Loader] Kaardistan GGUF tensorid Sovereign Kerneli mahutitesse...\n";

        std::unordered_map<std::string, GGUFTensorInfo> tensor_map;
        for (const auto& t : reader.tensors) {
            tensor_map[t.name] = t;
        }

        auto read_tensor_data = [&](const std::string& name, void* dest, size_t expected_bytes) {
            auto it = tensor_map.find(name);
            if (it == tensor_map.end()) {
                throw std::runtime_error("[GGUF Loader] Tensori ei leitud failist: " + name);
            }
            const auto& t_info = it->second;

            uint64_t absolute_offset = reader.data_offset + t_info.offset;
            file.seekg(absolute_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(dest), expected_bytes);
        };

        const auto& cfg = model.config;

        // 1. Token Embedding (Dünaamiline Laadimine ja NaN Mürgi Kaitse)
        auto it_embd = tensor_map.find("token_embd.weight");
        if (it_embd == tensor_map.end()) {
            throw std::runtime_error("[GGUF Loader] Viga: token_embd.weight puudub failist!");
        }

        if (it_embd->second.type == GGML_TYPE_F32) {
            std::cout << "[GGUF Loader] Token Embedding on F32. Laen otse...\n";
            read_tensor_data("token_embd.weight", model.token_embedding_table.data(), model.token_embedding_table.num_elements() * sizeof(float));
        }
        else if (it_embd->second.type == GGML_TYPE_F16) {
            std::cout << "[GGUF Loader] Token Embedding on F16. Dekvantiseerin FP32-ks...\n";
            size_t bytes = model.token_embedding_table.num_elements() * sizeof(uint16_t);
            std::vector<uint16_t> buf(model.token_embedding_table.num_elements());
            read_tensor_data("token_embd.weight", buf.data(), bytes);

            float* dest = model.token_embedding_table.data();
            for (size_t i = 0; i < model.token_embedding_table.num_elements(); ++i) {
                dest[i] = tensor_math_quantized::fp16_to_fp32(buf[i]);
            }
        }
        else if (it_embd->second.type == GGML_TYPE_Q4_0) {
            std::cout << "[GGUF Loader] Token Embedding on Q4_0. Dekvantiseerin FP32-ks...\n";
            size_t bytes = (model.token_embedding_table.num_elements() / 32) * sizeof(block_q4_0);
            std::vector<char> buf(bytes);
            read_tensor_data("token_embd.weight", buf.data(), bytes);

            block_q4_0* blocks = reinterpret_cast<block_q4_0*>(buf.data());
            float* dest = model.token_embedding_table.data();
            for (size_t b = 0; b < model.token_embedding_table.num_elements() / 32; ++b) {
                tensor_math_quantized::dequantize_q4_0_block(blocks[b], dest + b * 32);
            }
        }
        else {
            throw std::runtime_error("[GGUF Loader] Tundmatu token_embd.weight tüüp!");
        }

        // 2. Layers (Attention, FFN, Normid)
        for (int l = 0; l < cfg.num_layers; ++l) {
            auto& lw = model.layers[l];
            std::string p = "blk." + std::to_string(l) + ".";

            read_tensor_data(p + "attn_q.weight", lw.wq.data(), lw.wq.num_elements() / 32 * sizeof(block_q4_0));
            read_tensor_data(p + "attn_k.weight", lw.wk.data(), lw.wk.num_elements() / 32 * sizeof(block_q4_0));
            read_tensor_data(p + "attn_v.weight", lw.wv.data(), lw.wv.num_elements() / 32 * sizeof(block_q4_0));
            read_tensor_data(p + "attn_output.weight", lw.wo.data(), lw.wo.num_elements() / 32 * sizeof(block_q4_0));

            read_tensor_data(p + "ffn_gate.weight", lw.w1.data(), lw.w1.num_elements() / 32 * sizeof(block_q4_0));
            read_tensor_data(p + "ffn_down.weight", lw.w2.data(), lw.w2.num_elements() / 32 * sizeof(block_q4_0));
            read_tensor_data(p + "ffn_up.weight", lw.w3.data(), lw.w3.num_elements() / 32 * sizeof(block_q4_0));

            read_tensor_data(p + "attn_norm.weight", lw.attention_norm.data(), lw.attention_norm.num_elements() * sizeof(float));
            read_tensor_data(p + "ffn_norm.weight", lw.ffn_norm.data(), lw.ffn_norm.num_elements() * sizeof(float));
        }

       // 3. Final Norm & Output Weights
        read_tensor_data("output_norm.weight", model.final_norm.data(), model.final_norm.num_elements() * sizeof(float));

        // --- DEEMONI LÕPLIK HUKKAMINE: Q6_K DEKVANTISEERIJA ---
        #pragma pack(push, 1)
        struct block_q6_K {
            uint8_t ql[128];
            uint8_t qh[64];
            int8_t  scales[16];
            uint16_t d; // fp16
        };
        #pragma pack(pop)

        auto dequantize_q6_k_block = [](const block_q6_K* x, float* y) {
            float d = tensor_math_quantized::fp16_to_fp32(x->d);
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
        };

        auto it_out = tensor_map.find("output.weight");
        if (it_out != tensor_map.end()) {
            uint32_t out_type = it_out->second.type;

            if (out_type == 14) { // GGML_TYPE_Q6_K
                std::cout << "[GGUF Loader] Laen output.weight Q6_K formaadis ja dekvantiseerin puhtaks FP32-ks...\n";
                size_t bytes = (model.output_weights.num_elements() / 256) * sizeof(block_q6_K);
                std::vector<char> buf(bytes);
                read_tensor_data("output.weight", buf.data(), bytes);

                block_q6_K* blocks = reinterpret_cast<block_q6_K*>(buf.data());
                float* dest = model.output_weights.data();
                for (size_t b = 0; b < model.output_weights.num_elements() / 256; ++b) {
                    dequantize_q6_k_block(&blocks[b], dest + b * 256);
                }
            }
            else if (out_type == GGML_TYPE_Q4_0) {
                std::cout << "[GGUF Loader] Laen output.weight Q4_0 formaadis ja dekvantiseerin puhtaks FP32-ks...\n";
                size_t bytes = (model.output_weights.num_elements() / 32) * sizeof(block_q4_0);
                std::vector<char> buf(bytes);
                read_tensor_data("output.weight", buf.data(), bytes);

                block_q4_0* blocks = reinterpret_cast<block_q4_0*>(buf.data());
                float* dest = model.output_weights.data();
                for (size_t b = 0; b < model.output_weights.num_elements() / 32; ++b) {
                    tensor_math_quantized::dequantize_q4_0_block(blocks[b], dest + b * 32);
                }
            } else {
                throw std::runtime_error("[GGUF Loader] KRIITILINE VIGA: Tundmatu output.weight formaat!");
            }
        } else {
            std::cout << "[GGUF Loader] HOIATUS: output.weight puudub failist!\n";
        }

        std::cout << "[GGUF Loader] Laaduri sild on üles seotud ja valmis andmete pumpamiseks!\n";
    }
};

inline void load_gguf_model(const std::string& filepath, TransformerModel& model) {
    GGUFModelLoader::load_weights(filepath, model);
}