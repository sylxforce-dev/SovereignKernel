#pragma once

struct ModelConfig {
    int dim = 768;          // Transformer dimension (110M)
    int hidden_dim = 2048;  // FFN hidden dimension (110M)
    int num_layers = 12;    // Number of transformer layers (110M)
    int num_heads = 12;     // Number of query heads (110M)
    int num_kv_heads = 12;  // Number of key/value heads (GQA support)
    int vocab_size = 32000; // Vocabulary size
    int seq_len = 256;      // Max sequence length / context
    float rope_theta = 10000.0f;
    // UUS: kas output projection kaal on tied token_embedding_table'iga (nagu
    // stories110M), või on failis eraldi output-tensor (nagu enamik suuremaid
    // mudeleid, nt open_llama_3b_v2/TinyLlama, kus HF config'is
    // tie_word_embeddings=false). Loader loeb selle .bin päisest, mitte ei
    // eelda vaikimisi.
    bool output_weights_tied = true;
};