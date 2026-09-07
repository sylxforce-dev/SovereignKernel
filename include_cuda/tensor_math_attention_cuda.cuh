#pragma once
#include "tensor.h"
#include "tensor_math_cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

// Phase 5 — CUDA Multi-Head Attention (Native VRAM Pipeline)
// Kõik operatsioonid ja andmeliikumised toimuvad rangelt Device::CUDA mälus.

namespace tensor_math_attention_cuda {

    // --- CUDA KERNELID ---

    // 1. Softmax + Scale
    __global__ void softmax_scale_rows_kernel(float* scores, int rows, int cols, float scale_factor) {
        extern __shared__ float sdata[];
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int block_size = blockDim.x;
        float* row_ptr = scores + row * cols;

        float local_max = -1e30f;
        for (int j = tid; j < cols; j += block_size) {
            float v = row_ptr[j] / scale_factor;
            row_ptr[j] = v;
            local_max = fmaxf(local_max, v);
        }
        sdata[tid] = local_max;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
            __syncthreads();
        }
        float row_max = sdata[0];
        __syncthreads();

        float local_sum = 0.0f;
        for (int j = tid; j < cols; j += block_size) {
            float e = expf(row_ptr[j] - row_max);
            row_ptr[j] = e;
            local_sum += e;
        }
        sdata[tid] = local_sum;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        float row_sum = sdata[0];
        __syncthreads();

        for (int j = tid; j < cols; j += block_size) {
            row_ptr[j] = row_ptr[j] / row_sum;
        }
    }

    // 2. Maatriksi Transponeerimine (asendab CPU transpose)
    __global__ void transpose_2d_kernel(const float* in, float* out, int rows, int cols) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < rows && c < cols) {
            out[c * rows + r] = in[r * cols + c];
        }
    }

    // 3. Andmete lahtilõikamine peadeks (Slice)
    __global__ void extract_head_kernel(const float* full, float* head, int seq_len, int d_model, int head_dim, int h) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < seq_len * head_dim) {
            int s = idx / head_dim;
            int d = idx % head_dim;
            head[idx] = full[s * d_model + h * head_dim + d];
        }
    }

    // 4. Andmete tagasikleepimine (Concat)
    __global__ void concat_head_kernel(float* full, const float* head, int seq_len, int d_model, int head_dim, int h) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < seq_len * head_dim) {
            int s = idx / head_dim;
            int d = idx % head_dim;
            full[s * d_model + h * head_dim + d] = head[idx];
        }
    }


    // --- HOST WRAPPERID (Ootavad ja tagastavad ainult Device::CUDA tensoreid) ---

    inline Tensor transpose_cuda(const Tensor& input) {
        if (input.device() != Device::CUDA || input.shape().size() != 2) {
            throw std::runtime_error("transpose_cuda: input must be 2D CUDA tensor");
        }
        int rows = input.shape()[0];
        int cols = input.shape()[1];
        Tensor out({cols, rows}, Device::CUDA);

        dim3 block(16, 16);
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        transpose_2d_kernel<<<grid, block>>>(input.data(), out.data(), rows, cols);
        cudaDeviceSynchronize();
        return out;
    }

    inline Tensor softmax_scale_rows(const Tensor& scores_dev, float scale_factor) {
        if (scores_dev.device() != Device::CUDA || scores_dev.shape().size() != 2) {
            throw std::runtime_error("softmax_scale_rows_cuda: input must be 2D CUDA tensor");
        }
        int rows = scores_dev.shape()[0];
        int cols = scores_dev.shape()[1];

        // Loome GPU peale uue tensori ja kopeerime andmed (D2D), et inputi mitte in-place muuta
        Tensor out_dev(scores_dev.shape(), Device::CUDA);
        size_t bytes = scores_dev.num_elements() * sizeof(float);
        cudaMemcpy(out_dev.data(), scores_dev.data(), bytes, cudaMemcpyDeviceToDevice);

        int block_size = 1;
        while (block_size < cols && block_size < 256) block_size <<= 1;
        if (block_size < 1) block_size = 1;
        size_t shared_bytes = block_size * sizeof(float);

        softmax_scale_rows_kernel<<<rows, block_size, shared_bytes>>>(out_dev.data(), rows, cols, scale_factor);
        cudaDeviceSynchronize();

        return out_dev;
    }

    inline Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float scale_factor) {
        if (Q.device() != Device::CUDA || K.device() != Device::CUDA || V.device() != Device::CUDA) {
            throw std::runtime_error("attention_cuda: Q, K, V must be CUDA tensors");
        }

        Tensor K_t = transpose_cuda(K);
        Tensor scores = tensor_math_cuda::matmul(Q, K_t);
        Tensor scores_norm = softmax_scale_rows(scores, scale_factor);
        Tensor output = tensor_math_cuda::matmul(scores_norm, V);

        return output;
    }

    inline Tensor multi_head_attention(const Tensor& Q_full, const Tensor& K_full, const Tensor& V_full,
                                        int n_heads, float scale_factor) {
        if (Q_full.device() != Device::CUDA || K_full.device() != Device::CUDA || V_full.device() != Device::CUDA) {
            throw std::runtime_error("multi_head_attention_cuda: Inputs must be CUDA tensors");
        }

        int seq_len = Q_full.shape()[0];
        int d_model = Q_full.shape()[1];
        int head_dim = d_model / n_heads;

        Tensor output_full({seq_len, d_model}, Device::CUDA);

        int total_head_elements = seq_len * head_dim;
        int block_size = 256;
        int grid_size = (total_head_elements + block_size - 1) / block_size;

        for (int h = 0; h < n_heads; ++h) {
            Tensor Q_head({seq_len, head_dim}, Device::CUDA);
            Tensor K_head({seq_len, head_dim}, Device::CUDA);
            Tensor V_head({seq_len, head_dim}, Device::CUDA);

            // Slicing (andmed liiguvad VRAM-ist VRAM-i)
            extract_head_kernel<<<grid_size, block_size>>>(Q_full.data(), Q_head.data(), seq_len, d_model, head_dim, h);
            extract_head_kernel<<<grid_size, block_size>>>(K_full.data(), K_head.data(), seq_len, d_model, head_dim, h);
            extract_head_kernel<<<grid_size, block_size>>>(V_full.data(), V_head.data(), seq_len, d_model, head_dim, h);
            cudaDeviceSynchronize();

            Tensor head_out = scaled_dot_product_attention(Q_head, K_head, V_head, scale_factor);

            // Concatenation (andmed liiguvad tagasi suurde maatriksisse VRAM-is)
            concat_head_kernel<<<grid_size, block_size>>>(output_full.data(), head_out.data(), seq_len, d_model, head_dim, h);
            cudaDeviceSynchronize();
        }

        return output_full;
    }

} // namespace tensor_math_attention_cuda