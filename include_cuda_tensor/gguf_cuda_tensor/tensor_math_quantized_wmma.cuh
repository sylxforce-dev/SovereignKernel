#pragma once
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor.h"
#include "gguf_types_cuda.h"

namespace tensor_math_quantized_wmma {

using namespace nvcuda;

// ---------------------------------------------------------------------
// TENSOR SOVEREIGN: BATCHED GEMM (M x K) x (K x N)
// Klassikaline WMMA kernel ühe maatriksi jaoks.
// ---------------------------------------------------------------------
__global__ void gemm_q4_0_wmma_kernel(
    const void* __restrict__ W_q,     // M x K (Row-major, Q4_0 blockid)
    const half* __restrict__ X_fp16,  // N x K (Row-major)
    float* __restrict__ out,          // N x M (Row-major)
    int M, int N, int K)
{
    int m_tile = blockIdx.x * 16;
    int n_tile = blockIdx.y * 16;

    if (m_tile >= M || n_tile >= N) return;

    int k_tiles = K / 16;
    int blocks_per_row = K / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half a_tile[16 * 16];
    __shared__ half b_tile[16 * 16];

    int lane = threadIdx.x % 32;

    for (int kt = 0; kt < k_tiles; ++kt) {

        // 1. Loeme B_tile (X transpose lennult)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = lane + i * 32;
            int k_local = linear_idx / 16;
            int n_local = linear_idx % 16;

            int global_n = n_tile + n_local;
            int global_k = kt * 16 + k_local;

            if (global_n < N) {
                b_tile[k_local * 16 + n_local] = X_fp16[global_n * K + global_k];
            } else {
                b_tile[k_local * 16 + n_local] = __float2half(0.0f);
            }
        }

        // 2. Loeme A_tile (W Q4_0 dequant)
        if (lane < 16) {
            int local_row = lane;
            int global_row = m_tile + local_row;
            if (global_row < M) {
                int global_col_start = kt * 16;
                int block_idx = global_col_start / 32;
                int in_block_offset = global_col_start % 32;

                const uint8_t* blk_ptr = static_cast<const uint8_t*>(W_q)
                    + (static_cast<size_t>(global_row) * blocks_per_row + block_idx) * 18;

                uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(blk_ptr);
                float d = __half2float(*reinterpret_cast<const __half*>(&d_fp16));

                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    int nibble_idx = in_block_offset + c;
                    uint8_t q = blk_ptr[2 + (nibble_idx % 16)];
                    float w = (nibble_idx < 16)
                        ? static_cast<float>((q & 0x0F) - 8) * d
                        : static_cast<float>((q >> 4) - 8) * d;
                    a_tile[local_row * 16 + c] = __float2half(w);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 16; ++c) a_tile[local_row * 16 + c] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // 3. WMMA Süda
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_tile, 16);
        wmma::load_matrix_sync(b_frag, b_tile, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    __shared__ float c_tile[16 * 16];
    wmma::store_matrix_sync(c_tile, acc_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // 4. Salvestame out (N x M)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int linear_idx = lane + i * 32;
        int m_local = linear_idx / 16;
        int n_local = linear_idx % 16;

        int global_m = m_tile + m_local;
        int global_n = n_tile + n_local;

        if (global_m < M && global_n < N) {
            out[global_n * M + global_m] = c_tile[m_local * 16 + n_local];
        }
    }
}

// ---------------------------------------------------------------------
// [LEVEL 4] THE BEAST PREFILL: FUSED QKV WMMA
// Purustab mälumüüri: Loeb X_fp16 ühe korra ja jagab GPU gridi dünaamiliselt Q, K, V vahel.
// ---------------------------------------------------------------------
__global__ void gemm_qkv_fused_q4_0_wmma_kernel(
    const void* __restrict__ W_q,
    const void* __restrict__ W_k,
    const void* __restrict__ W_v,
    const half* __restrict__ X_fp16,
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    int dim, int kv_dim, int N, int K)
{
    int m_tile = blockIdx.x * 16;
    int n_tile = blockIdx.y * 16;

    // Kokku M = dim + kv_dim + kv_dim
    int M_total = dim + kv_dim * 2;
    if (m_tile >= M_total || n_tile >= N) return;

    // Ruumiline ruutimine (Spacial Routing)
    const void* current_W;
    float* current_out;
    int local_m_tile;
    int current_M;

    if (m_tile < dim) {
        current_W = W_q;
        current_out = Q_out;
        local_m_tile = m_tile;
        current_M = dim;
    } else if (m_tile < dim + kv_dim) {
        current_W = W_k;
        current_out = K_out;
        local_m_tile = m_tile - dim;
        current_M = kv_dim;
    } else {
        current_W = W_v;
        current_out = V_out;
        local_m_tile = m_tile - dim - kv_dim;
        current_M = kv_dim;
    }

    int k_tiles = K / 16;
    int blocks_per_row = K / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half a_tile[16 * 16];
    __shared__ half b_tile[16 * 16];

    int lane = threadIdx.x % 32;

    for (int kt = 0; kt < k_tiles; ++kt) {
        // 1. Loeme B_tile (X transpose lennult) - Püsib kuumana L2-s!
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = lane + i * 32;
            int k_local = linear_idx / 16;
            int n_local = linear_idx % 16;

            int global_n = n_tile + n_local;
            int global_k = kt * 16 + k_local;

            if (global_n < N) {
                b_tile[k_local * 16 + n_local] = X_fp16[global_n * K + global_k];
            } else {
                b_tile[k_local * 16 + n_local] = __float2half(0.0f);
            }
        }

        // 2. Loeme A_tile (Ruumiline dequant)
        if (lane < 16) {
            int local_row = lane;
            int global_row = local_m_tile + local_row;
            if (global_row < current_M) {
                int global_col_start = kt * 16;
                int block_idx = global_col_start / 32;
                int in_block_offset = global_col_start % 32;

                const uint8_t* blk_ptr = static_cast<const uint8_t*>(current_W)
                    + (static_cast<size_t>(global_row) * blocks_per_row + block_idx) * 18;

                uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(blk_ptr);
                float d = __half2float(*reinterpret_cast<const __half*>(&d_fp16));

                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    int nibble_idx = in_block_offset + c;
                    uint8_t q = blk_ptr[2 + (nibble_idx % 16)];
                    float w = (nibble_idx < 16)
                        ? static_cast<float>((q & 0x0F) - 8) * d
                        : static_cast<float>((q >> 4) - 8) * d;
                    a_tile[local_row * 16 + c] = __float2half(w);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 16; ++c) a_tile[local_row * 16 + c] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // 3. WMMA Süda
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_tile, 16);
        wmma::load_matrix_sync(b_frag, b_tile, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    __shared__ float c_tile[16 * 16];
    wmma::store_matrix_sync(c_tile, acc_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // 4. Salvestame Dünaamilisse Out puhvrisse
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int linear_idx = lane + i * 32;
        int m_local = linear_idx / 16;
        int n_local = linear_idx % 16;

        int global_m = local_m_tile + m_local;
        int global_n = n_tile + n_local;

        if (global_m < current_M && global_n < N) {
            current_out[global_n * current_M + global_m] = c_tile[m_local * 16 + n_local];
        }
    }
}

// Lanch wrapperid
inline void launch_gemm_q4_0_wmma(const void* W_q, const half* X_fp16, float* out, int M, int N, int K, cudaStream_t stream = 0) {
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dim3 block(32);
    gemm_q4_0_wmma_kernel<<<grid, block, 0, stream>>>(W_q, X_fp16, out, M, N, K);
}

inline void launch_gemm_qkv_fused_wmma(
    const void* W_q, const void* W_k, const void* W_v,
    const half* X_fp16,
    float* Q_out, float* K_out, float* V_out,
    int dim, int kv_dim, int N, int K, cudaStream_t stream = 0)
{
    int M_total = dim + 2 * kv_dim;
    dim3 grid((M_total + 15) / 16, (N + 15) / 16);
    dim3 block(32);
    gemm_qkv_fused_q4_0_wmma_kernel<<<grid, block, 0, stream>>>(
        W_q, W_k, W_v, X_fp16, Q_out, K_out, V_out, dim, kv_dim, N, K
    );
}

} // namespace tensor_math_quantized_wmma