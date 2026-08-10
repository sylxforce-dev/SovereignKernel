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

# Lisame include kataloogide hulka ka model_data ja include, et GGUF failid oleksid leitavad
target_include_directories(sovereign_kernel PUBLIC src include model_data include/gguf)
target_link_libraries(sovereign_kernel PRIVATE CUDA::cudart)

# Kui OpenMP on leitud, lingime selle kernelile, et CPU matemaatika kasutaks kõiki tuumi
if(OpenMP_CXX_FOUND)
    target_link_libraries(sovereign_kernel PRIVATE OpenMP::OpenMP_CXX)
endif()

# =========================================================================
# [DIAGNOSTIKA JA TESTID - AINULT LÕPUSÜSTEEM]
# =========================================================================

# Real Model Runner test (vana)
add_executable(model_runner src/real_runner.cpp)
target_link_libraries(model_runner PRIVATE sovereign_kernel CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(model_runner PRIVATE OpenMP::OpenMP_CXX)
endif()

# Sunduslik AVX2 lipp - sama mis model_runner2 (varem PUUDUS siit, tähendas
# et tensor_math_cpu.h AVX2 harud ei aktiveerunud kunagi, kood jooksis skalaar-fallback'is)
if(MSVC)
    target_compile_options(model_runner PRIVATE /arch:AVX2)
else()
    target_compile_options(model_runner PRIVATE -mavx2 -mfma)
endif()

# Real Model Runner 2 test: loads GGUF weights and runs quantized inference pipeline (AVX2 Enabled)
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

# Sunduslik AVX2 riistvaraline lipp MSVC kompilaatorile Q4_0 dekvantiseerimise kiirendamiseks
if(MSVC)
    target_compile_options(model_runner2 PRIVATE /arch:AVX2)
else()
    target_compile_options(model_runner2 PRIVATE -mavx2 -mfma)
endif()