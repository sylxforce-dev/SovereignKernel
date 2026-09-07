#pragma once
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Phase 5 — CUDA Transformer Math (Native VRAM Pipeline)
// Kõik operatsioonid ja andmeliikumised toimuvad rangelt Device::CUDA mälus.
// Host-wrapperid ei tee enam cudaMemcpy H2D ega D2H koopiaid.

namespace tensor_math_transformer_cuda {

    // ==========================================================================
    // 1. RMSNorm
    // ==========================================================================
    __global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int block_size = blockDim.x;

        float partial = 0.0f;
        for (int i = tid; i < n; i += block_size) {
            float v = x[i];
            partial += v * v;
        }
        sdata[tid] = partial;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }

        float rms;
        if (tid == 0) {
            float mean_sq = sdata[0] / static_cast<float>(n);
            sdata[0] = sqrtf(mean_sq + eps);
        }
        __syncthreads();
        rms = sdata[0];

        for (int i = tid; i < n; i += block_size) {
            out[i] = (x[i] / rms) * weight[i];
        }
    }

    inline Tensor rmsnorm(const Tensor& x_dev, const Tensor& weight_dev, float eps = 1e-5f) {
        if (x_dev.device() != Device::CUDA || weight_dev.device() != Device::CUDA) {
            throw std::runtime_error("rmsnorm_cuda: Inputs must be CUDA tensors");
        }
        int n = static_cast<int>(x_dev.num_elements());
        Tensor out_dev(x_dev.shape(), Device::CUDA);

        int block_size = 1;
        while (block_size < n && block_size < 1024) block_size <<= 1;
        if (block_size < 1) block_size = 1;
        size_t shared_bytes = block_size * sizeof(float);

        rmsnorm_kernel<<<1, block_size, shared_bytes>>>(x_dev.data(), weight_dev.data(), out_dev.data(), n, eps);
        cudaDeviceSynchronize();

        return out_dev;
    }

    // ==========================================================================
    // 2. RoPE (In-place adjacent)
    // ==========================================================================
    __global__ void rope_inplace_adjacent_kernel(const float* in, float* out, int n, int position, float base) {
        int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int i = pair_idx * 2;

        if (i < n) {
            float freq = 1.0f / powf(base, static_cast<float>(i) / static_cast<float>(n));
            float val = static_cast<float>(position) * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            float x0 = in[i];
            float x1 = in[i + 1];

            out[i]     = x0 * cos_val - x1 * sin_val;
            out[i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    inline void rope_inplace_adjacent(Tensor& x_dev, int position, float base = 10000.0f) {
        if (x_dev.device() != Device::CUDA) {
            throw std::runtime_error("rope_inplace_adjacent_cuda: input must be CUDA tensor");
        }
        int n = static_cast<int>(x_dev.num_elements());

        int num_pairs = n / 2;
        int block_size = num_pairs < 256 ? num_pairs : 256;
        if (block_size < 1) block_size = 1;
        int grid_size = (num_pairs + block_size - 1) / block_size;

        // In-place operatsioon otse VRAM-is
        rope_inplace_adjacent_kernel<<<grid_size, block_size>>>(x_dev.data(), x_dev.data(), n, position, base);
        cudaDeviceSynchronize();
    }

    // ==========================================================================
    // 3. SwiGLU
    // ==========================================================================
    __global__ void swiglu_kernel(const float* in, float* out, int half_n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < half_n) {
            float a = in[tid];
            float b = in[half_n + tid];
            float silu_a = a / (1.0f + expf(-a));
            out[tid] = silu_a * b;
        }
    }

    inline Tensor swiglu(const Tensor& x_dev) {
        if (x_dev.device() != Device::CUDA) {
             throw std::runtime_error("swiglu_cuda: input must be CUDA tensor");
        }
        int n = static_cast<int>(x_dev.num_elements());
        int half_n = n / 2;
        Tensor out_dev({half_n}, Device::CUDA);

        int block_size = 256;
        int grid_size = (half_n + block_size - 1) / block_size;
        swiglu_kernel<<<grid_size, block_size>>>(x_dev.data(), out_dev.data(), half_n);
        cudaDeviceSynchronize();

        return out_dev;
    }

    // ==========================================================================
    // 4. GELU
    // ==========================================================================
    __global__ void gelu_kernel(const float* in, float* out, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            float val = in[tid];
            const float sqrt_2_over_pi = 0.7978845608028654f;
            float inner = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
            out[tid] = 0.5f * val * (1.0f + tanhf(inner));
        }
    }

    inline Tensor gelu(const Tensor& x_dev) {
         if (x_dev.device() != Device::CUDA) {
             throw std::runtime_error("gelu_cuda: input must be CUDA tensor");
        }
        int n = static_cast<int>(x_dev.num_elements());
        Tensor out_dev(x_dev.shape(), Device::CUDA);

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        gelu_kernel<<<grid_size, block_size>>>(x_dev.data(), out_dev.data(), n);
        cudaDeviceSynchronize();

        return out_dev;
    }

    // ==========================================================================
    // 5. 1D Softmax
    // ==========================================================================
    __global__ void softmax_1d_kernel(const float* in, float* out, int n) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int block_size = blockDim.x;

        float local_max = -1e30f;
        for (int i = tid; i < n; i += block_size) {
            local_max = fmaxf(local_max, in[i]);
        }
        sdata[tid] = local_max;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
            __syncthreads();
        }
        float max_val = sdata[0];
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < n; i += block_size) {
            float e = expf(in[i] - max_val);
            out[i] = e;
            local_sum += e;
        }
        sdata[tid] = local_sum;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        float sum_val = sdata[0];
        __syncthreads();

        for (int i = tid; i < n; i += block_size) {
            out[i] /= sum_val;
        }
    }

    inline Tensor softmax(const Tensor& x_dev) {
        if (x_dev.device() != Device::CUDA) {
             throw std::runtime_error("softmax_cuda: input must be CUDA tensor");
        }
        int n = static_cast<int>(x_dev.num_elements());
        Tensor out_dev(x_dev.shape(), Device::CUDA);

        int block_size = 1;
        while (block_size < n && block_size < 1024) block_size <<= 1;
        if (block_size < 1) block_size = 1;
        size_t shared_bytes = block_size * sizeof(float);

        softmax_1d_kernel<<<1, block_size, shared_bytes>>>(x_dev.data(), out_dev.data(), n);
        cudaDeviceSynchronize();

        return out_dev;
    }

} // namespace tensor_math_transformer_cuda