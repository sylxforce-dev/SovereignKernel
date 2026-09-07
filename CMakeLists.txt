cmake_minimum_required(VERSION 3.18)

project(SovereignKernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

# =========================================================================
# [KÜBERNEETILINE ANKUR: CUDA GRAPHS PTDS LUKK]
# See sunnib KÕIK kernelid (ka staatilistes teekides) kasutama per-thread striimi.
# Ilma selleta graafi salvestamine ebaõnnestub!
# =========================================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

# OpenMP tugi kõigi CPU tuumade rakendamiseks maatrikskorrutistes
find_package(OpenMP REQUIRED)

# =========================================================================
# [SOVEREIGN KERNEL STATIC LIBRARY TARGET]
# Pakib kogu mäluhalduse, matemaatika, KV-cache'i, generatsiooni ja API teegiks.
# =========================================================================
add_library(sovereign_kernel STATIC
        src/sovereign_kernel.cu
        src/sovereign_api.cu
        include/sovereign_api.h
        include/tensor.h
        include/tensor_kv_cache.h
        include/tensor_math_attention.h
        include/tensor_math_cpu.h
        include/tensor_math_generation.h
        include/tensor_math_transformer.h
)

set_target_properties(sovereign_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Lisame KÕIK vajalikud include kaustad (ka uue gguf_cuda) globaalselt teegile:
target_include_directories(sovereign_kernel PUBLIC
        src
        include
        include_cuda
        include_cuda/gguf_cuda
        model_data
        include/gguf
)
target_link_libraries(sovereign_kernel PRIVATE CUDA::cudart)

# Kui OpenMP on leitud, lingime selle kernelile
if(OpenMP_CXX_FOUND)
    target_link_libraries(sovereign_kernel PRIVATE OpenMP::OpenMP_CXX)
endif()

# =========================================================================
# [DIAGNOSTIKA JA TESTID - CPU SÜSTEEMID]
# =========================================================================

# Real Model Runner test (CPU)
add_executable(model_runner src/real_runner.cpp)
target_link_libraries(model_runner PRIVATE sovereign_kernel CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(model_runner PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(model_runner PRIVATE /arch:AVX2)
else()
    target_compile_options(model_runner PRIVATE -mavx2 -mfma)
endif()

# Real Model Runner 2 test (CPU GGUF)
add_executable(model_runner2 src/real_runner2.cpp
        include/gguf/gguf_reader.h
        include/gguf/gguf_types.h
        include/gguf/gguf_loader.h
        include/gguf/tensor_math_quantized.h
)
target_link_libraries(model_runner2 PRIVATE sovereign_kernel CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(model_runner2 PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(model_runner2 PRIVATE /arch:AVX2)
else()
    target_compile_options(model_runner2 PRIVATE -mavx2 -mfma)
endif()

# =========================================================================
# [CUDA MOOTORID JA TESTID]
# =========================================================================

# Smoketest
add_executable(cuda_smoketest src_cuda/step0_smoketest.cu
        include_cuda/tensor_math_cuda.cuh
        include_cuda/tensor_math_transformer_cuda.cuh
)
target_link_libraries(cuda_smoketest PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_smoketest PRIVATE include include_cuda)
set_target_properties(cuda_smoketest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transform test
add_executable(cuda_transform_test src_cuda/cuda_transform_test.cu
        include_cuda/tensor_math_transformer_cuda.cuh
)
target_link_libraries(cuda_transform_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_transform_test PRIVATE include include_cuda)
set_target_properties(cuda_transform_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Attention test
add_executable(cuda_attention_test src_cuda/cuda_attention_test.cu
        include_cuda/tensor_math_attention_cuda.cuh
        include_cuda/tensor_kv_cache_cuda.cuh
        include_cuda/tensor_math_generation_cuda.cuh
)
target_link_libraries(cuda_attention_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_attention_test PRIVATE include include_cuda)
set_target_properties(cuda_attention_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# FFN Test
add_executable(cuda_ffn_test src_cuda/cuda_ffn_test.cu
        include_cuda/tensor_math_transformer_cuda.cuh
)
target_link_libraries(cuda_ffn_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_ffn_test PRIVATE include include_cuda)
set_target_properties(cuda_ffn_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Generation Test
add_executable(cuda_generation_test src_cuda/cuda_generation_test.cu
        include_cuda/tensor_math_generation_cuda.cuh
)
target_link_libraries(cuda_generation_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_generation_test PRIVATE include include_cuda)
set_target_properties(cuda_generation_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# KV Cache Test
add_executable(cuda_kv_cache_test src_cuda/cuda_kv_cache_test.cu
        include_cuda/tensor_kv_cache_cuda.cuh
)
target_link_libraries(cuda_kv_cache_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_kv_cache_test PRIVATE include include_cuda)
set_target_properties(cuda_kv_cache_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Toy Transformer Test
add_executable(cuda_toy_transformer_test src_cuda/cuda_toy_transformer_test.cu
        include_cuda/tensor_math_cuda.cuh
        include_cuda/tensor_math_transformer_cuda.cuh
        include_cuda/tensor_math_attention_cuda.cuh
        include_cuda/tensor_math_generation_cuda.cuh
)
target_link_libraries(cuda_toy_transformer_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_toy_transformer_test PRIVATE include include_cuda)
set_target_properties(cuda_toy_transformer_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Real Runner CUDA
add_executable(real_runner_cuda src_cuda/real_runner_cuda.cu)
target_link_libraries(real_runner_cuda PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(real_runner_cuda PRIVATE include include_cuda model_data)
set_target_properties(real_runner_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =========================================================================
# [REAL RUNNER GGUF CUDA - Pure VRAM Quantized Inference (Llama 3 8B)]
# =========================================================================
add_executable(real_runner_gguf_cuda
        src_cuda/real_runner_gguf_cuda.cu
        include_cuda/gguf_cuda/gguf_types_cuda.h
        include_cuda/gguf_cuda/gguf_reader_cuda.h
        include_cuda/gguf_cuda/gguf_loader_cuda.h
        include_cuda/gguf_cuda/tensor_math_quantized_cuda.cuh
)
target_link_libraries(real_runner_gguf_cuda PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(real_runner_gguf_cuda PRIVATE
        include
        include_cuda
        include_cuda/gguf_cuda
        model_data
)
set_target_properties(real_runner_gguf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =========================================================================
# [CUDA TENSOR CORE (WMMA) FAAS 2]
# =========================================================================

# WMMA Smoketest
add_executable(cuda_tensor_smoketest src_cuda_tensor/step0_wmma_smoketest.cu
        include_cuda_tensor/wmma_tensor.cuh
)
target_link_libraries(cuda_tensor_smoketest PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_tensor_smoketest PRIVATE include include_cuda include_cuda_tensor)
set_target_properties(cuda_tensor_smoketest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Q4_0 dequant test
add_executable(cuda_tensor_q4_dequant_test src_cuda_tensor/step1_q4_dequant_test.cu
        include_cuda_tensor/wmma_tensor.cuh
)
target_link_libraries(cuda_tensor_q4_dequant_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_tensor_q4_dequant_test PRIVATE include include_cuda include_cuda_tensor)
set_target_properties(cuda_tensor_q4_dequant_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# WMMA Q4_0 GEMV test
add_executable(cuda_tensor_gemv_wmma_test src_cuda_tensor/step2_gemv_wmma_test.cu
        include_cuda_tensor/wmma_tensor.cuh
)
target_link_libraries(cuda_tensor_gemv_wmma_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_tensor_gemv_wmma_test PRIVATE include include_cuda include_cuda_tensor)
set_target_properties(cuda_tensor_gemv_wmma_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =========================================================================
# [REAL RUNNER GGUF WMMA - TENSOR SOVEREIGN ENGINE]
# =========================================================================
add_executable(real_runner_gguf_wmma
        src_cuda_tensor/real_runner_gguf_wmma.cu
        include_cuda/gguf_cuda/gguf_types_cuda.h
        include_cuda/gguf_cuda/gguf_reader_cuda.h
        include_cuda/gguf_cuda/gguf_loader_cuda.h
        include_cuda/gguf_cuda/tensor_math_quantized_cuda.cuh
        include_cuda_tensor/wmma_tensor.cuh
        include_cuda_tensor/gguf_cuda_tensor/tensor_math_quantized_wmma.cuh
        include_cuda_tensor/flash_attention_batched.cuh
)
target_link_libraries(real_runner_gguf_wmma PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(real_runner_gguf_wmma PRIVATE
        include
        include_cuda
        include_cuda/gguf_cuda
        include_cuda_tensor
        model_data
)
set_target_properties(real_runner_gguf_wmma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cuda_tensor_gemm_wmma_test src_cuda_tensor/step3_gemm_wmma_batched_test.cu
        include_cuda_tensor/gguf_cuda_tensor/tensor_math_quantized_wmma.cuh
)
target_link_libraries(cuda_tensor_gemm_wmma_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_tensor_gemm_wmma_test PRIVATE include include_cuda include_cuda_tensor include_cuda_tensor/gguf_cuda_tensor)
set_target_properties(cuda_tensor_gemm_wmma_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cuda_tensor_flash_attn_test src_cuda_tensor/step4_flash_attention_test.cu
        include_cuda_tensor/flash_attention_batched.cuh
)
target_link_libraries(cuda_tensor_flash_attn_test PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(cuda_tensor_flash_attn_test PRIVATE include include_cuda include_cuda_tensor include_cuda_tensor/gguf_cuda_tensor)
set_target_properties(cuda_tensor_flash_attn_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(real_runner_gguf_wmma_v2 src_cuda_tensor/real_runner_gguf_wmma_v2.cu)
target_link_libraries(real_runner_gguf_wmma_v2 PRIVATE sovereign_kernel CUDA::cudart)
target_include_directories(real_runner_gguf_wmma_v2 PRIVATE include include_cuda include_cuda_tensor include_cuda_tensor/gguf_cuda_tensor model_data)
set_target_properties(real_runner_gguf_wmma_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)