#include <iostream>
#include <cuda_runtime.h>
#include "sovereign_kernel.h"

extern "C" void sovereign_kernel_engine_pulse() {
    std::cout << "[SovereignKernel Runtime] Pulss tuvastatud. Tuum on aktiivne." << std::endl;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess) {
        std::cout << "[SovereignKernel CUDA] Leitud CUDA seadmeid: " << deviceCount << std::endl;
    } else {
        std::cout << "[SovereignKernel CUDA] Viga seadme tuvastamisel: " << cudaGetErrorString(err) << std::endl;
    }

    Tensor t({2, 2}, Device::CPU);
    std::cout << "[SovereignKernel Tensor] Test-tensor mõõtmetega [2, 2] edukalt loodud mällu." << std::endl;
}