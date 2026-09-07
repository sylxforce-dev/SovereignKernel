#pragma once
#include "tensor.h"
#include "gguf_types_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace tensor_math_quantized_cuda {

    __device__ inline float fp16_to_fp32_cuda(uint16_t h) {
        __half_raw hr; hr.x = h; return __half2float(hr);
    }

    // ====================================================================
    // 🚀 ZERO-SYNC Q4_0 GEMV
    // ====================================================================
    __global__ void gemv_q4_0_kernel_warp_vectorized(
        const void* __restrict__ W_q,
        const float* __restrict__ x,
        float* __restrict__ out,
        int M, int K)
    {
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int lane_id = threadIdx.x % 32;
        int row = warp_id;
        if (row >= M) return;

        int blocks_per_row = K / 32;
        const uint8_t* row_ptr = static_cast<const uint8_t*>(W_q) + row * blocks_per_row * 18;
        float local_sum = 0.0f;

        int qs_idx = lane_id % 16;
        int shift = (lane_id / 16) * 4;

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk_ptr = row_ptr + b * 18;
            uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(blk_ptr);
            float d = __half2float(*reinterpret_cast<const __half*>(&d_fp16));
            uint8_t q = blk_ptr[2 + qs_idx];
            float weight = static_cast<float>(((q >> shift) & 0x0F) - 8) * d;
            local_sum += weight * x[b * 32 + lane_id];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (lane_id == 0) out[row] = local_sum;
    }

    // ====================================================================
    // 🚀 ZERO-SYNC FUSED QKV (Reads 'x' once, computes Q, K, V)
    // ====================================================================
    __global__ void gemv_q4_0_qkv_fused_warp_vectorized(
        const void* __restrict__ W_q,
        const void* __restrict__ W_k,
        const void* __restrict__ W_v,
        const float* __restrict__ x,
        float* __restrict__ out_q,
        float* __restrict__ out_k,
        float* __restrict__ out_v,
        int dim, int kv_dim)
    {
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int lane_id = threadIdx.x % 32;
        int row = warp_id;
        if (row >= dim) return;

        int blocks_per_row = dim / 32;
        const uint8_t* row_ptr_q = static_cast<const uint8_t*>(W_q) + row * blocks_per_row * 18;

        float local_sum_q = 0.0f; float local_sum_k = 0.0f; float local_sum_v = 0.0f;

        bool calc_kv = (row < kv_dim);
        const uint8_t* row_ptr_k = nullptr;
        const uint8_t* row_ptr_v = nullptr;
        if (calc_kv) {
            row_ptr_k = static_cast<const uint8_t*>(W_k) + row * blocks_per_row * 18;
            row_ptr_v = static_cast<const uint8_t*>(W_v) + row * blocks_per_row * 18;
        }

        int qs_idx = lane_id % 16;
        int shift = (lane_id / 16) * 4;

        for (int b = 0; b < blocks_per_row; ++b) {
            float x_val = x[b * 32 + lane_id];

            const uint8_t* blk_q = row_ptr_q + b * 18;
            uint16_t d_fp16_q = *reinterpret_cast<const uint16_t*>(blk_q);
            float d_q = __half2float(*reinterpret_cast<const __half*>(&d_fp16_q));
            uint8_t q_val = blk_q[2 + qs_idx];
            float w_q = static_cast<float>(((q_val >> shift) & 0x0F) - 8) * d_q;
            local_sum_q += w_q * x_val;

            if (calc_kv) {
                const uint8_t* blk_k = row_ptr_k + b * 18;
                const uint8_t* blk_v = row_ptr_v + b * 18;
                uint16_t d_fp16_k = *reinterpret_cast<const uint16_t*>(blk_k);
                uint16_t d_fp16_v = *reinterpret_cast<const uint16_t*>(blk_v);
                float d_k = __half2float(*reinterpret_cast<const __half*>(&d_fp16_k));
                float d_v = __half2float(*reinterpret_cast<const __half*>(&d_fp16_v));
                uint8_t q_k = blk_k[2 + qs_idx];
                uint8_t q_v = blk_v[2 + qs_idx];
                float w_k = static_cast<float>(((q_k >> shift) & 0x0F) - 8) * d_k;
                float w_v = static_cast<float>(((q_v >> shift) & 0x0F) - 8) * d_v;

                local_sum_k += w_k * x_val;
                local_sum_v += w_v * x_val;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum_q += __shfl_down_sync(0xffffffff, local_sum_q, offset);
            if (calc_kv) {
                local_sum_k += __shfl_down_sync(0xffffffff, local_sum_k, offset);
                local_sum_v += __shfl_down_sync(0xffffffff, local_sum_v, offset);
            }
        }

        if (lane_id == 0) {
            out_q[row] = local_sum_q;
            if (calc_kv) {
                out_k[row] = local_sum_k;
                out_v[row] = local_sum_v;
            }
        }
    }

    // ====================================================================
    // 🚀 ZERO-SYNC FUSED W1 + W3 + SWIGLU
    // ====================================================================
    __global__ void gemv_q4_0_w1_w3_swiglu_fused_warp_vectorized(
        const void* __restrict__ W1_q,
        const void* __restrict__ W3_q,
        const float* __restrict__ x,
        float* __restrict__ swiglu_out,
        int M, int K)
    {
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int lane_id = threadIdx.x % 32;
        int row = warp_id;
        if (row >= M) return;

        int blocks_per_row = K / 32;
        const uint8_t* row_ptr_1 = static_cast<const uint8_t*>(W1_q) + row * blocks_per_row * 18;
        const uint8_t* row_ptr_3 = static_cast<const uint8_t*>(W3_q) + row * blocks_per_row * 18;

        float local_sum1 = 0.0f; float local_sum3 = 0.0f;
        int qs_idx = lane_id % 16; int shift = (lane_id / 16) * 4;

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk1 = row_ptr_1 + b * 18;
            const uint8_t* blk3 = row_ptr_3 + b * 18;
            uint16_t d_fp16_1 = *reinterpret_cast<const uint16_t*>(blk1);
            uint16_t d_fp16_3 = *reinterpret_cast<const uint16_t*>(blk3);
            float d1 = __half2float(*reinterpret_cast<const __half*>(&d_fp16_1));
            float d3 = __half2float(*reinterpret_cast<const __half*>(&d_fp16_3));
            uint8_t q1 = blk1[2 + qs_idx]; uint8_t q3 = blk3[2 + qs_idx];
            float w1 = static_cast<float>(((q1 >> shift) & 0x0F) - 8) * d1;
            float w3 = static_cast<float>(((q3 >> shift) & 0x0F) - 8) * d3;
            float x_val = x[b * 32 + lane_id];
            local_sum1 += w1 * x_val; local_sum3 += w3 * x_val;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum1 += __shfl_down_sync(0xffffffff, local_sum1, offset);
            local_sum3 += __shfl_down_sync(0xffffffff, local_sum3, offset);
        }
        if (lane_id == 0) {
            swiglu_out[row] = (local_sum1 / (1.0f + expf(-local_sum1))) * local_sum3;
        }
    }

    // ====================================================================
    // 🚀 ZERO-SYNC FUSED W2 + RESIDUAL ADD (Smashing the VRAM wall)
    // ====================================================================
    __global__ void gemv_q4_0_fused_residual_warp_vectorized(
        const void* __restrict__ W_q,
        const float* __restrict__ x,
        float* __restrict__ out_residual,
        int M, int K)
    {
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int lane_id = threadIdx.x % 32;
        int row = warp_id;
        if (row >= M) return;

        int blocks_per_row = K / 32;
        const uint8_t* row_ptr = static_cast<const uint8_t*>(W_q) + row * blocks_per_row * 18;
        float local_sum = 0.0f;
        int qs_idx = lane_id % 16; int shift = (lane_id / 16) * 4;

        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk_ptr = row_ptr + b * 18;
            uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(blk_ptr);
            float d = __half2float(*reinterpret_cast<const __half*>(&d_fp16));
            uint8_t q = blk_ptr[2 + qs_idx];
            float weight = static_cast<float>(((q >> shift) & 0x0F) - 8) * d;
            local_sum += weight * x[b * 32 + lane_id];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) out_residual[row] += local_sum;
    }

    inline void launch_gemv_q4_0(const void* W_q, const float* x, float* out, int M, int K, cudaStream_t stream = 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / 32;
        int blocks = (M + warps_per_block - 1) / warps_per_block;
        gemv_q4_0_kernel_warp_vectorized<<<blocks, threads_per_block, 0, stream>>>(W_q, x, out, M, K);
    }

    inline void launch_gemv_qkv_fused(const void* W_q, const void* W_k, const void* W_v, const float* x, float* out_q, float* out_k, float* out_v, int dim, int kv_dim, cudaStream_t stream = 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / 32;
        int blocks = (dim + warps_per_block - 1) / warps_per_block;
        gemv_q4_0_qkv_fused_warp_vectorized<<<blocks, threads_per_block, 0, stream>>>(W_q, W_k, W_v, x, out_q, out_k, out_v, dim, kv_dim);
    }

    inline void launch_gemv_q4_0_w1_w3_swiglu(const void* W1_q, const void* W3_q, const float* x, float* swiglu_out, int M, int K, cudaStream_t stream = 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / 32;
        int blocks = (M + warps_per_block - 1) / warps_per_block;
        gemv_q4_0_w1_w3_swiglu_fused_warp_vectorized<<<blocks, threads_per_block, 0, stream>>>(W1_q, W3_q, x, swiglu_out, M, K);
    }

    inline void launch_gemv_q4_0_fused_residual(const void* W_q, const float* x, float* out_residual, int M, int K, cudaStream_t stream = 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / 32;
        int blocks = (M + warps_per_block - 1) / warps_per_block;
        gemv_q4_0_fused_residual_warp_vectorized<<<blocks, threads_per_block, 0, stream>>>(W_q, x, out_residual, M, K);
    }

} // namespace tensor_math_quantized_cuda