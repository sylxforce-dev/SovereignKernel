#pragma once

#include <cstdint>
#include <cstddef>

// GGUF tensorite kvantiseerimise tüübid (GGML / GGUF standard)
enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

// Q4_0 ploki struktuur (32 kaalu plokis)
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;          // delta (scale) FP16 formaadis
    uint8_t  qs[16];     // 32 pakitud 4-bitist kaalu
};

// Q4_K ploki struktuur (Super-block)
struct block_q4_K {
    uint16_t d;          // super-block scale (FP16)
    uint16_t dmin;       // super-block min (FP16)
    uint8_t  scales[12]; // skaalad väiksematele plokkidele
    uint8_t  qs[128];    // 256 kaalu pakituna
};
#pragma pack(pop)

// Abifunktsioon ploki suuruse leidmiseks baitides vastavalt tüübile
inline size_t ggml_row_size(int type, int64_t ne) {
    switch (type) {
        case GGML_TYPE_F32:
            return ne * sizeof(float);
        case GGML_TYPE_F16:
            return ne * sizeof(uint16_t);
        case GGML_TYPE_Q4_0:
            return (ne / 32) * sizeof(block_q4_0);
        case GGML_TYPE_Q4_K:
            return (ne / 256) * sizeof(block_q4_K);
        default:
            return ne * sizeof(float);
    }
}