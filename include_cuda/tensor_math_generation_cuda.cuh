#pragma once
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

// Phase 7 — Generation & Sampling (CUDA Basic Wrapper for testing)

namespace tensor_math_generation_cuda {

    // --- 1. Temperature Skaleerimine GPU-s ---
    __global__ void apply_temperature_kernel(float* data, int size, float temp) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            data[tid] /= temp;
        }
    }

    // Võtab sisse CPU logits tensori, skaleerib GPU-s ja annab CPU-sse tagasi (testimiseks)
    inline void apply_temperature(Tensor& logits_host, float temperature) {
        if (temperature <= 0.0f) throw std::runtime_error("Temperature must be > 0");
        if (temperature == 1.0f) return;

        // PARANDUS: Viskasime abifunktsiooni välja, kasutame puhast num_elements()
        size_t size = logits_host.num_elements();
        Tensor logits_dev(logits_host.shape(), Device::CUDA);
        size_t bytes = size * sizeof(float);

        cudaError_t err = cudaMemcpy(logits_dev.data(), logits_host.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error("apply_temperature_cuda: H2D failed");

        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        apply_temperature_kernel<<<grid_size, block_size>>>(logits_dev.data(), size, temperature);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) throw std::runtime_error("apply_temperature_cuda: kernel failed");

        // Kirjutame testides tagasi CPU mällu
        err = cudaMemcpy(logits_host.data(), logits_dev.data(), bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("apply_temperature_cuda: D2H failed");
    }

    // --- 2. Greedy Sample (Argmax) GPU-s ---
    __global__ void argmax_kernel(const float* data, int* out_idx, float* out_val, int size) {
        extern __shared__ int shared_idx[];
        float* shared_val = (float*)&shared_idx[blockDim.x];

        int tid = threadIdx.x;
        int global_id = blockIdx.x * blockDim.x + threadIdx.x;

        float max_v = -1e30f; // Kindel miinus-lõpmatus
        int max_i = 0;

        for (int i = global_id; i < size; i += blockDim.x * gridDim.x) {
            if (data[i] > max_v) {
                max_v = data[i];
                max_i = i;
            }
        }

        shared_val[tid] = max_v;
        shared_idx[tid] = max_i;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_val[tid + s] > shared_val[tid]) {
                    shared_val[tid] = shared_val[tid + s];
                    shared_idx[tid] = shared_idx[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            out_val[blockIdx.x] = shared_val[0];
            out_idx[blockIdx.x] = shared_idx[0];
        }
    }

    inline int greedy_sample(const Tensor& logits_host) {
        // PARANDUS: Viskasime abifunktsiooni välja, kasutame puhast num_elements()
        size_t size = logits_host.num_elements();
        if (size == 0) throw std::runtime_error("greedy_sample_cuda: empty tensor");

        Tensor logits_dev(logits_host.shape(), Device::CUDA);
        size_t bytes = size * sizeof(float);
        cudaError_t err = cudaMemcpy(logits_dev.data(), logits_host.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error("greedy_sample_cuda: H2D failed");

        int block_size = 256;
        int grid_size = 256;

        int* d_block_indices;
        float* d_block_maxes;
        cudaMalloc(&d_block_indices, grid_size * sizeof(int));
        cudaMalloc(&d_block_maxes, grid_size * sizeof(float));

        size_t shared_mem_size = block_size * (sizeof(int) + sizeof(float));

        argmax_kernel<<<grid_size, block_size, shared_mem_size>>>(logits_dev.data(), d_block_indices, d_block_maxes, size);
        cudaDeviceSynchronize();

        std::vector<int> h_block_indices(grid_size);
        std::vector<float> h_block_maxes(grid_size);
        cudaMemcpy(h_block_indices.data(), d_block_indices, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_block_maxes.data(), d_block_maxes, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_block_indices);
        cudaFree(d_block_maxes);

        float best_val = -1e30f;
        int best_idx = 0;
        for (int i = 0; i < grid_size; ++i) {
            if (h_block_maxes[i] > best_val) {
                best_val = h_block_maxes[i];
                best_idx = h_block_indices[i];
            }
        }

        return best_idx;
    }
}