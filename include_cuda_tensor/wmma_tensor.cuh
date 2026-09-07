#pragma once
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor.h"
#include "gguf_types_cuda.h"

// Faas 2 — WMMA Tensor Core matmul + Q4_0 dequant FP16-ks.
namespace tensor_math_cuda_tensor {

using namespace nvcuda;

// ---------------------------------------------------------------------
// step0: puhas WMMA matmul (valideeritud, PASS, max error ~0.0012)
// ---------------------------------------------------------------------
__global__ void wmma_matmul_16x16_kernel(const half* a, const half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

class TensorFP16 {
private:
    half* data_;
    std::vector<int> shape_;
    size_t num_elements_;
    bool owns_data_;

public:
    explicit TensorFP16(const std::vector<int>& shape)
        : data_(nullptr), shape_(shape), owns_data_(true) {
        num_elements_ = 1;
        for (int dim : shape) num_elements_ *= dim;

        cudaError_t err = cudaMalloc(&data_, num_elements_ * sizeof(half));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("TensorFP16 allocation failed: ") + cudaGetErrorString(err));
        }
    }

    ~TensorFP16() {
        if (owns_data_ && data_) cudaFree(data_);
    }

    TensorFP16(const TensorFP16&) = delete;
    TensorFP16& operator=(const TensorFP16&) = delete;

    TensorFP16(TensorFP16&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)),
          num_elements_(other.num_elements_), owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.owns_data_ = false;
    }

    half* data() const { return data_; }
    size_t num_elements() const { return num_elements_; }
    const std::vector<int>& shape() const { return shape_; }

    static TensorFP16 from_host_half(const std::vector<half>& host_data, const std::vector<int>& shape) {
        TensorFP16 dst(shape);
        cudaMemcpy(dst.data(), host_data.data(), host_data.size() * sizeof(half), cudaMemcpyHostToDevice);
        return dst;
    }
};

inline Tensor matmul_wmma_16x16(const TensorFP16& a, const TensorFP16& b) {
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();

    if (shape_a.size() != 2 || shape_b.size() != 2 ||
        shape_a[0] != 16 || shape_a[1] != 16 ||
        shape_b[0] != 16 || shape_b[1] != 16) {
        throw std::runtime_error(
            "matmul_wmma_16x16: step0 nouab tapselt 16x16 x 16x16 sisendit (tiling/padding tuleb hiljem)");
    }

    Tensor c(shape_a, Device::CUDA);
    wmma_matmul_16x16_kernel<<<1, 32>>>(a.data(), b.data(), c.data());

    return c;
}

// ---------------------------------------------------------------------
// step1: Q4_0 -> FP16 dequant, otse, ilma FP32 vahesammuta.
// ---------------------------------------------------------------------
__global__ void dequantize_q4_0_to_fp16_kernel(
    const void* __restrict__ W_q,
    half* __restrict__ out,
    int num_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* blk_ptr = static_cast<const uint8_t*>(W_q) + block_idx * 18;

    uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(blk_ptr);
    float d = __half2float(*reinterpret_cast<const __half*>(&d_fp16));

    half* out_block = out + block_idx * 32;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t q = blk_ptr[2 + i];
        float low  = static_cast<float>((q & 0x0F) - 8) * d;
        float high = static_cast<float>((q >> 4)   - 8) * d;
        out_block[i]      = __float2half(low);
        out_block[16 + i] = __float2half(high);
    }
}

inline TensorFP16 dequantize_q4_0(const void* W_q_device, int num_blocks) {
    TensorFP16 out({num_blocks * 32});
    int threads = 128;
    int blocks = (num_blocks + threads - 1) / threads;
    dequantize_q4_0_to_fp16_kernel<<<blocks, threads>>>(W_q_device, out.data(), num_blocks);
    return out;
}

// ---------------------------------------------------------------------
// step2: Q4_0 GEMV läbi WMMA — (M x K) x (K x 1), batched 1 kaupa Decode'iks
// ---------------------------------------------------------------------
__global__ void gemv_q4_0_wmma_kernel(
    const void* __restrict__ W_q,
    const half* __restrict__ x_fp16,
    float* __restrict__ out,
    int M, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int m_tile = warp_id * 16;
    if (m_tile >= M) return;

    int k_tiles = K / 16;
    int blocks_per_row = K / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half a_tile[16 * 16];
    __shared__ half b_tile[16 * 16];

    int lane = threadIdx.x % 32;

    for (int kt = 0; kt < k_tiles; ++kt) {
        if (lane < 16) {
            b_tile[lane * 16 + 0] = x_fp16[kt * 16 + lane];
            #pragma unroll
            for (int c = 1; c < 16; ++c) b_tile[lane * 16 + c] = __float2half(0.0f);
        }

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

    if (lane < 16) {
        int global_row = m_tile + lane;
        if (global_row < M) out[global_row] = c_tile[lane * 16 + 0];
    }
}

inline void launch_gemv_q4_0_wmma(const void* W_q, const half* x_fp16, float* out, int M, int K, cudaStream_t stream = 0) {
    int warps_needed = (M + 15) / 16;
    int threads_per_block = 32;
    int warps_per_block = threads_per_block / 32;
    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
    gemv_q4_0_wmma_kernel<<<blocks, threads_per_block, 0, stream>>>(W_q, x_fp16, out, M, K);
}

// ---------------------------------------------------------------------
// step3 (UUS TENSOR SOVEREIGN): BATCHED GEMM (M x K) x (K x N)
// Loob massiivse kiiruse pikkade promptide (Prefill) seedimisel.
// ---------------------------------------------------------------------
__global__ void gemm_q4_0_wmma_kernel(
    const void* __restrict__ W_q,     // M x K (Row-major, Q4_0 blockid)
    const half* __restrict__ X_fp16,  // N x K (Row-major)
    float* __restrict__ out,          // N x M (Row-major)
    int M, int N, int K)
{
    // Grid x = M blokid, Grid y = N blokid
    int m_tile = blockIdx.x * 16;
    int n_tile = blockIdx.y * 16;

    if (m_tile >= M || n_tile >= N) return;

    int k_tiles = K / 16;
    int blocks_per_row = K / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half a_tile[16 * 16];
    __shared__ half b_tile[16 * 16];

    int lane = threadIdx.x % 32; // 1 warp = 32 threads, haldab tervet 16x16 tile'i

    for (int kt = 0; kt < k_tiles; ++kt) {

        // --- 1. Loeme B_tile (X transpose) ---
        // Iga lõim laeb 8 elementi, kattes 32 lõimega kokku 256 elementi (16x16 tile)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = lane + i * 32;
            int k_local = linear_idx / 16;
            int n_local = linear_idx % 16;

            int global_n = n_tile + n_local;
            int global_k = kt * 16 + k_local;

            if (global_n < N) {
                // WMMA ootab A*B, kus B on (KxN). X on mälus (NxK). Seega teeme lennult transponeerimise B-tile'i.
                b_tile[k_local * 16 + n_local] = X_fp16[global_n * K + global_k];
            } else {
                b_tile[k_local * 16 + n_local] = __float2half(0.0f);
            }
        }

        // --- 2. Loeme A_tile (W dequant) ---
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

        // --- 3. WMMA Süda ---
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

    // --- 4. Salvestame out (N x M, kuna tokenid on read ja embeddingud veerud) ---
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

inline void launch_gemm_q4_0_wmma(const void* W_q, const half* X_fp16, float* out, int M, int N, int K, cudaStream_t stream = 0) {
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dim3 block(32); // Täpselt 1 warp per 16x16 tile, ohutu shared memory jagamine
    gemm_q4_0_wmma_kernel<<<grid, block, 0, stream>>>(W_q, X_fp16, out, M, N, K);
}

} // namespace tensor_math_cuda_tensor