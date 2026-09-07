#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

namespace flash_attention_cuda {

    // Br = 32 (Päringuid bloki kohta), Bc = 64 (K/V klotsi suurus)
    // blockDim.x = 32 (Iga lõim arvutab ühe Q tokeni)
    template <int HEAD_DIM>
    __global__ void flash_attention_batched_causal_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* __restrict__ Out,
        int N, int num_heads, int num_kv_heads, float scale)
    {
        // Millist Q tokenit see lõim töötleb?
        int q_idx = blockIdx.y * blockDim.x + threadIdx.x;
        int head_idx = blockIdx.x;
        int kv_head_idx = head_idx / (num_heads / num_kv_heads); // Toetab MQA/GQA

        bool is_valid_q = (q_idx < N);

        // Dünaamiline SRAM: Siia laetakse K ja V klotsid VRAM-ist
        extern __shared__ float smem[];
        float* s_K = smem;                             // Bc x HEAD_DIM
        float* s_V = s_K + 64 * HEAD_DIM;              // Bc x HEAD_DIM

        // Registrid: Hoiame Q ja O otse protsessori tuumas! (Puhas kiirus)
        float q_reg[HEAD_DIM];
        float o_reg[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;

        // Laeme Q registritesse (kui lõim on aktiivne)
        if (is_valid_q) {
            const float* q_ptr = Q + (q_idx * num_heads + head_idx) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                q_reg[d] = q_ptr[d];
            }
        }

        // Online Softmaxi muutujad registrites (Mitte kuskil VRAM-is!)
        float m_i = -1e30f;
        float l_i = 0.0f;

        int num_tiles = (N + 63) / 64; // Bc = 64

        // Käime läbi kõik K/V klotsid
        for (int t = 0; t < num_tiles; ++t) {
            int k_start = t * 64;

            // IMPLICIT MASKING: Kui see klots on tulevikus terve bloki jaoks, katkestame kohe.
            int block_max_q = blockIdx.y * 32 + 31;
            if (k_start > block_max_q) break;

            // KOLLABORATIIVNE LAADIMINE: 32 lõime laevad koos 64 tokeni jagu K ja V andmeid SRAM-i
            int total_elements = 64 * HEAD_DIM;
            for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
                int k_idx = k_start + (i / HEAD_DIM);
                int d_idx = i % HEAD_DIM;

                // Mälupaigutus eeldab: [N, num_heads, head_dim]
                if (k_idx < N) {
                    s_K[i] = K[(k_idx * num_kv_heads + kv_head_idx) * HEAD_DIM + d_idx];
                    s_V[i] = V[(k_idx * num_kv_heads + kv_head_idx) * HEAD_DIM + d_idx];
                } else {
                    s_K[i] = 0.0f;
                    s_V[i] = 0.0f;
                }
            }
            __syncthreads(); // Ootame, kuni kõik andmed on SRAM-is turvaliselt olemas

            if (is_valid_q) {
                // Arvutame Attention skoorid K/V klotsi vastu
                for (int k = 0; k < 64; ++k) {
                    int actual_k_idx = k_start + k;

                    // IMPLICIT MASKING: Kontroll lõime tasemel (tulevikku ei näe)
                    if (actual_k_idx > q_idx || actual_k_idx >= N) continue;

                    // Skalaarkorrutis (Dot Product)
                    float s = 0.0f;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        s += q_reg[d] * s_K[k * HEAD_DIM + d];
                    }
                    s *= scale;

                    // THE NEW MATH (Online Softmax)
                    float m_new = max(m_i, s);
                    float exp_corr = expf(m_i - m_new); // Parandustegur eelmisele summale
                    float p = expf(s - m_new);

                    l_i = l_i * exp_corr + p;

                    for (int d = 0; d < HEAD_DIM; ++d) {
                        o_reg[d] = o_reg[d] * exp_corr + p * s_V[k * HEAD_DIM + d];
                    }
                    m_i = m_new;
                }
            }
            __syncthreads(); // Sünkroniseerime enne järgmise klotsi SRAM-i tõmbamist
        }

        // Finaal: Normaliseerime ja kirjutame tulemuse VRAM-i (Ainult ÜKS KORD!)
        if (is_valid_q) {
            float inv_l = 1.0f / l_i;
            float* out_ptr = Out + (q_idx * num_heads + head_idx) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                out_ptr[d] = o_reg[d] * inv_l;
            }
        }
    }

    // Launch funktsioon (Arhitektuuriline ankur)
    inline void launch_flash_attention_batched_causal(
        const float* Q, const float* K, const float* V, float* Out,
        int N, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream = 0)
    {
        int Bc = 64; // Klotsi suurus
        size_t smem_size = (Bc * head_dim + Bc * head_dim) * sizeof(float); // K ja V maht SRAMis

        dim3 grid(num_heads, (N + 31) / 32); // 1 Grid blokk 32 tokeni kohta iga pea jaoks
        dim3 block(32); // 1 Lõim = 1 Token (Optimaalne registrite kasutus)

        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // TinyLlama (ja enamus mudeleid) kasutab head_dim = 64
        if (head_dim == 64) {
            flash_attention_batched_causal_kernel<64><<<grid, block, smem_size, stream>>>(Q, K, V, Out, N, num_heads, num_kv_heads, scale);
        } else if (head_dim == 128) { // Tuleviku valmidus
            flash_attention_batched_causal_kernel<128><<<grid, block, smem_size, stream>>>(Q, K, V, Out, N, num_heads, num_kv_heads, scale);
        } else {
            // Turvalukk
            printf("CRITICAL ERROR: Unsupported head_dim %d for FlashAttention!\n", head_dim);
        }
    }
} // namespace flash_attention_cuda