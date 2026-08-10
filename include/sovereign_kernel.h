#pragma once

// SovereignKernel - Core Public API
#include "tensor.h"
#include "tensor_math_cpu.h"
#include "tensor_math_transformer.h"
#include "tensor_math_attention.h"
#include "tensor_kv_cache.h"
#include "tensor_math_generation.h"

#ifdef __cplusplus
extern "C" {
#endif

void sovereign_kernel_engine_pulse();

#ifdef __cplusplus
}
#endif