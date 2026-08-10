#pragma once
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "model.h"
#include "model_config.h"

// Karpathy .bin faili päise struktuur
struct TransformerCheckpointHeader {
    int dim;          // transformer dimension
    int hidden_dim;   // for ffn layers
    int num_layers;   // number of layers
    int num_heads;    // number of query heads
    int num_kv_heads; // number of key/value heads (can be < num_heads)
    int vocab_size;   // vocab size (NEGATIVE = untied output weights)
    int seq_len;      // max sequence length
};

inline TransformerCheckpointHeader peek_model_header(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file for header peek: " + filepath);
    }
    TransformerCheckpointHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(TransformerCheckpointHeader));
    if (!file) {
        throw std::runtime_error("Failed to read checkpoint header (peek): " + filepath);
    }
    return header;
}

inline void load_model_weights(const std::string& filepath, TransformerModel& model) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + filepath);
    }

    TransformerCheckpointHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(TransformerCheckpointHeader));
    if (!file) {
        throw std::runtime_error("Failed to read checkpoint header from .bin file");
    }

    bool output_tied = (header.vocab_size >= 0);
    int actual_vocab_size = std::abs(header.vocab_size);

    std::cout << "[Model Loader] Read .bin Header -> dim: " << header.dim
              << ", hidden_dim: " << header.hidden_dim
              << ", layers: " << header.num_layers
              << ", num_heads: " << header.num_heads
              << ", num_kv_heads: " << header.num_kv_heads
              << ", vocab: " << actual_vocab_size
              << ", seq_len: " << header.seq_len
              << ", output_tied: " << (output_tied ? "true" : "false") << "\n";

    if (model.config.dim != header.dim || model.config.hidden_dim != header.hidden_dim ||
        model.config.num_layers != header.num_layers || model.config.num_heads != header.num_heads ||
        model.config.num_kv_heads != header.num_kv_heads || model.config.vocab_size != actual_vocab_size) {
        throw std::runtime_error(
            "load_model_weights: model.config ei klapi .bin päisega - kasuta peek_model_header() enne mudeli loomist.");
    }
    model.config.output_weights_tied = output_tied;

    int num_layers = header.num_layers;

    auto read_tensor_debug = [&](const std::string& name, Tensor& t) {
        size_t elements = t.num_elements();
        size_t bytes_to_read = elements * sizeof(float);
        std::streampos current_pos = file.tellg();

        std::cout << "[DEBUG] Reading tensor [" << name << "] elements: " << elements
                  << " (bytes: " << bytes_to_read << ") at file pos: " << current_pos << "\n";

        file.read(reinterpret_cast<char*>(t.data()), bytes_to_read);
        if (!file) {
            std::cerr << "[CRITICAL] Failed at tensor: " << name
                      << " | Expected bytes: " << bytes_to_read
                      << " | Current pos: " << file.tellg() << "\n";
            throw std::runtime_error("Error reading weights: unexpected end of file during tensor load: " + name);
        }
    };

    // 1. Token embeddingud
    read_tensor_debug("token_embedding_table", model.token_embedding_table);

    // 2. Kihtide kaalud (Karpathy formaadi standardne järjestus)
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_attention_norm", model.layers[i].attention_norm);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_wq", model.layers[i].wq);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_wk", model.layers[i].wk);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_wv", model.layers[i].wv);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_wo", model.layers[i].wo);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_ffn_norm", model.layers[i].ffn_norm);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_w1", model.layers[i].w1);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_w2", model.layers[i].w2);
    }
    for (int i = 0; i < num_layers; ++i) {
        read_tensor_debug("layer_" + std::to_string(i) + "_w3", model.layers[i].w3);
    }

    // 3. Final norm
    read_tensor_debug("final_norm", model.final_norm);

    // 4. Output weights (tied või untied)
    if (output_tied) {
        std::memcpy(model.output_weights.data(), model.token_embedding_table.data(),
                    model.token_embedding_table.num_elements() * sizeof(float));
        std::cout << "[Model Loader] Output weights tied with token embeddings successfully!\n";
    } else {
        read_tensor_debug("output_weights (untied)", model.output_weights);
        std::cout << "[Model Loader] Output weights loaded as separate (untied) tensor!\n";
    }

    file.close();
    std::cout << "[Model Loader] All .bin weights successfully mapped!\n";
}