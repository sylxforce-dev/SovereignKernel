#pragma once
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Phase 8 — CUDA Matrix Multiplication (Native VRAM Pipeline)
// Removed CPU host<->device copies. Expects and returns Device::CUDA tensors.

namespace tensor_math_cuda {

    // One thread computes one output element C[row][col].
    __global__ void matmul_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    inline Tensor matmul(const Tensor& a_dev, const Tensor& b_dev) {
        const auto& shape_a = a_dev.shape();
        const auto& shape_b = b_dev.shape();

        if (shape_a.size() != 2 || shape_b.size() != 2) {
            throw std::runtime_error("matmul_cuda: both tensors must be 2D");
        }

        // NÕUAME PUHAST CUDA MÄLU (CPU tensorid pole enam lubatud)
        if (a_dev.device() != Device::CUDA || b_dev.device() != Device::CUDA) {
            throw std::runtime_error("matmul_cuda: both tensors must be CUDA tensors");
        }

        int M = shape_a[0];
        int K = shape_a[1];
        int K2 = shape_b[0];
        int N = shape_b[1];

        if (K != K2) {
            throw std::runtime_error("matmul_cuda: inner dimensions must match (got "
                + std::to_string(K) + " vs " + std::to_string(K2) + ")");
        }

        // Loome vastuse otse VRAM-i, mingeid H2D koopiaid ei toimu
        Tensor c_dev({M, N}, Device::CUDA);

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        matmul_kernel<<<grid, block>>>(a_dev.data(), b_dev.data(), c_dev.data(), M, K, N);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("matmul_cuda: kernel launch failed: ") + cudaGetErrorString(err));
        }

        // Tagastame puhta VRAM tensori, D2H koopiat CPU-sse enam ei tehta!
        return c_dev;
    }

} // namespace tensor_math_cuda