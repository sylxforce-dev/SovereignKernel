#pragma once
#include "tensor.h"
#include "model_config.h"
#include <vector>

// --- Ühe kihi reaalsete kaalude struktuur ---
struct LayerWeights {
    Tensor attention_norm; // [dim]
    Tensor wq;             // [dim, dim]
    Tensor wk;             // [kv_dim, dim]
    Tensor wv;             // [kv_dim, dim]
    Tensor wo;             // [dim, dim]

    Tensor ffn_norm;       // [dim]
    Tensor w1;             // [hidden_dim, dim] (gate)
    Tensor w2;             // [dim, hidden_dim] (down)
    Tensor w3;             // [hidden_dim, dim] (up)

    LayerWeights(const ModelConfig& config)
        : attention_norm({config.dim}, Device::CPU),
          wq({config.dim, config.dim}, Device::CPU),
          wk({(config.dim / config.num_heads) * config.num_kv_heads, config.dim}, Device::CPU),
          wv({(config.dim / config.num_heads) * config.num_kv_heads, config.dim}, Device::CPU),
          wo({config.dim, config.dim}, Device::CPU),
          ffn_norm({config.dim}, Device::CPU),
          w1({config.hidden_dim, config.dim}, Device::CPU),
          w2({config.dim, config.hidden_dim}, Device::CPU),
          w3({config.hidden_dim, config.dim}, Device::CPU) {}
};

// --- Täielik Mudeli Konteiner ---
struct TransformerModel {
    ModelConfig config;

    Tensor token_embedding_table; // [vocab_size, dim]
    std::vector<LayerWeights> layers;
    Tensor final_norm;            // [dim]
    Tensor output_weights;        // [vocab_size, dim]

    TransformerModel(const ModelConfig& cfg)
        : config(cfg),
          token_embedding_table({cfg.vocab_size, cfg.dim}, Device::CPU),
          final_norm({cfg.dim}, Device::CPU),
          output_weights({cfg.vocab_size, cfg.dim}, Device::CPU)
    {
        for (int i = 0; i < cfg.num_layers; ++i) {
            layers.emplace_back(cfg);
        }
    }
};