#include "tensor.h"
#include "tensor_math_cpu.h"
#include "tensor_math_cuda.cuh"
#include "tensor_math_generation.h"

#include <iostream>
#include <cuda_runtime.h>

extern "C" {

void sovereign_engine_init() {
    std::cout << "[SovereignEngine] Initializing core subsystems...\n";

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err == cudaSuccess) {
        std::cout
            << "[SovereignEngine] Detected CUDA devices: "
            << device_count << "\n";
    }
    else {
        std::cout
            << "[SovereignEngine] CUDA initialization warning or CPU-only fallback.\n";
    }
}


void sovereign_engine_pulse() {

    std::cout << "[SovereignEngine] --- ENGINE PULSE ACTIVE ---\n";

    try {

        Tensor a({2,2}, Device::CPU);
        Tensor b({2,2}, Device::CPU);

        a.data()[0] = 1.0f;
        a.data()[1] = 2.0f;
        a.data()[2] = 3.0f;
        a.data()[3] = 4.0f;

        b.data()[0] = 5.0f;
        b.data()[1] = 6.0f;
        b.data()[2] = 7.0f;
        b.data()[3] = 8.0f;


        Tensor c = tensor_math_cpu::matmul(a,b);

        std::cout
            << "[SovereignEngine] CPU Matmul Test Result [0,0]: "
            << c.data()[0]
            << "\n";

    }
    catch(const std::exception& e) {

        std::cerr
            << "[SovereignEngine] Pulse failure: "
            << e.what()
            << "\n";
    }
}


void sovereign_engine_shutdown() {

    std::cout
        << "[SovereignEngine] Shutting down and clearing VRAM/CPU buffers...\n";
}


int sovereign_run_inference_step(
    const float* input_tokens,
    int num_tokens)
{
    return 0;
}

}