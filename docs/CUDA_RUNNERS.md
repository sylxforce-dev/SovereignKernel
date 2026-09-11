# CUDA Runners — What Each One Actually Does

SovereignKernel has four GPU inference entry points. They are not four versions of the same thing — each represents a different point in the optimization path, and they use meaningfully different execution strategies. This document describes what each one does, concretely, based on the code.

> **For the full narrative** — every measured phase, every rejected experiment (async streams, float4 loads, N-gram speculative decoding, WMMA decode, the cuBLAS branch, Top-K sampling), and the current profiling work — see the [Runtime Engineering Diary](SovereignKernel_Runtime_Engineering_Diary.md). This document stays focused on the file-level structural differences between the four runners; the diary is the up-to-date source for throughput numbers and ongoing investigation.

---

## 1. `real_runner_gguf_cuda.cu` — Pure CUDA baseline

**Location:** `src_cuda/real_runner_gguf_cuda.cu`
**Layer:** Pure CUDA (no Tensor Cores, no CUDA Graphs)

This is the original GPU inference implementation and the baseline for the CUDA execution path: GGUF/Q4_0 weights are loaded directly into VRAM and inference is performed using custom CUDA kernels, with one token processed at a time.

The original implementation was built **without CUDA Graphs**. CUDA Graphs were integrated later as a separate optimization stage on top of the working Pure CUDA execution path. This baseline therefore represents the direct host-driven CUDA implementation before graph-based launch optimization.

**What's in it:**

- **FLOAT4-vectorized RMSNorm** (`rmsnorm_kernel_float4`) and a fused **Add+RMSNorm** kernel (`add_and_rmsnorm_kernel_float4`) — both operate on `float4` (16-byte) chunks rather than scalar floats, reducing the number of element-level memory operations in the norm passes.
- **FlashAttention-lite with online softmax** (`mha_kv_cache_kernel`) — one CUDA block is assigned to each attention head. Threads process portions of the KV cache while maintaining running maximum and normalization values, avoiding a separate full score-normalization pass. Grouped-query attention is handled directly through `heads_per_kv_group`.
- **RoPE + KV-cache write** (`rope_and_cache_k_kernel`) — rotates the key vectors and writes the resulting values directly into the appropriate per-layer KV-cache position.
- **Q4_0 GEMV** for the linear layers through `tensor_math_quantized_cuda::launch_gemv_q4_0`, including the fused **W1+W3+SwiGLU** FFN path through `launch_gemv_q4_0_w1_w3_swiglu`.
- **No CUDA Graphs in the original baseline.** `forward_pass_gguf_cuda()` is invoked directly from the host generation loop for every token, and the required CUDA kernel sequence is launched again for each step.
- **No Tensor Cores.** This runner uses the custom CUDA/Q4 execution path rather than the later WMMA/Tensor Core prefill path.
- **Runtime config reload every turn** — `load_runtime_config()` and `load_system_prompt()` are called for each user input, allowing runtime parameters and the system prompt to be changed without restarting the process.

**Role:** original correctness/performance baseline for the GPU inference path. Every later runner builds on the concepts established here.

---

## 2. `real_runner_gguf_wmma.cu` — "V1 Gold Standard": CUDA Graphs + Tensor Cores

**Location:** `src_cuda_tensor/real_runner_gguf_wmma.cu`
**Layer:** CUDA Tensor Core (WMMA), built on top of the pure-CUDA GGUF loading code

Two structural changes from the baseline runner:

**a) CUDA Graphs for N=1 decode.** The entire per-token decode forward pass is captured once into a CUDA Graph on the first generated token, then replayed on every subsequent token via `cudaGraphLaunch`. This collapses dozens of individual kernel-launch host round-trips into a single graph-launch call per token.

**b) PTX-level L2 cache pinning on RMSNorm.** Inline PTX (`st.cg.global.v4.f32`) tells the hardware to cache the write at the L2 level rather than evict it prematurely, since the next kernel in the pipeline reads it straight back.

**c) WMMA tensor-core GEMM for the fused QKV projection**, computing Q/K/V from a single fused tensor-core GEMM call.

**d) Batched prefill, separate from decode** — `forward_pass_batched_prefill()` handles the initial prompt (N > 1 tokens) as one batched pass, distinct from the N=1 decode path.

**e) Stop-sequence detection** with a rolling buffer against `<|user|>`.

**Role:** the "known good" reference implementation for the tensor-core path at this stage of the project.

---

## 3. `real_runner_gguf_wmma_v2.cu` — "T4 Speculative Decoder" (retired)

**Location:** `src_cuda_tensor/real_runner_gguf_wmma_v2.cu`
**Layer:** CUDA Tensor Core (WMMA), same foundation as `_wmma.cu`

A parallel experiment layered on the WMMA foundation, adding **self-speculative decoding**: n-gram draft matching against the model's own generation history, verified via a batched WMMA forward pass instead of sequential N=1 steps.

**Status: retired.** This was ultimately superseded by the decode-path work in `_wmma_v4.cu` (see below) — FP16 KV cache, read-only cache-path reads, and CPU/GPU pipelining gave larger, more reliable gains than self-speculative decoding did on this workload, and the speculative approach was not carried forward. Kept in the repo as part of the engineering record; see the Runtime Engineering Diary for the full negative-result writeup.

---

## 4. `real_runner_gguf_wmma_v4.cu` — Current production decode path

**Location:** `src_cuda_tensor/real_runner_gguf_wmma_v4.cu`
**Layer:** CUDA Tensor Core (WMMA) for prefill, heavily optimized custom scalar CUDA for decode
**Status: current, actively maintained.** This is the runner behind every recent throughput number in the Runtime Engineering Diary.

This file is the accumulation point for a series of decode-path optimizations, each validated independently and kept only when it measured a real improvement (see the diary for the full accept/reject history):

- **FP16 KV cache** with fused RoPE + cache-write, and read-only-cache-path (`__ldg`) reads on every K/V access inside attention.
- **CUDA Graphs + CPU/GPU async pipelining** — host-side token bookkeeping (detokenization, stop-sequence check, buffered output) runs concurrently with GPU decode instead of blocking on it.
- **Zero-copy mapped host memory** for logits, avoiding an explicit device-to-host copy.
- **FP16, then Q4_0, output-projection weights** — `output.weight` was first halved (FP32→FP16) then quantized further in VRAM (FP16→Q4_0, ~36.8MB), cutting the final-layer GEMV's memory traffic well below the original FP32 footprint.
- **2-warp split-K GEMV decomposition** for the N=1 linear layers.
- **Deferred-scale Q4_0 dequantization** — the block scale factor `d` is applied once per 32-weight block after accumulation, instead of once per element, saving redundant floating-point multiplies without changing the result (see `tensor_math_quantized_cuda.cuh`).
- **Prefix-sum-localized sampling** — the CPU sampler reuses the per-thread partial sums already computed during the parallel softmax to localize the roulette-selection scan to one thread's chunk instead of the full 32,000-entry vocabulary. (An earlier Top-K=40 filter was tried for the same purpose and regressed performance; it was reverted — see the diary.)
- **WO-projection fused with its residual add**, following the same fused-residual pattern already used for the FFN down-projection.

WMMA/Tensor Cores remain confined to **batched prefill** in this runner, same as `_wmma.cu` — decode-path WMMA was tested independently (twice) and consistently regressed throughput (~90–100 tok/s vs. several hundred tok/s for the scalar decode path), confirming that N=1 decode on this hardware is memory-bandwidth-bound, not compute-bound.

**Current measured throughput:** ~324 tok/s average decode, ~340 tok/s best observed window (see the Runtime Engineering Diary for the full version-by-version breakdown and the currently open question about sequence-length-dependent GPU cost).

---

## Summary comparison

| | `real_runner_gguf_cuda` | `real_runner_gguf_wmma` | `real_runner_gguf_wmma_v2` | `real_runner_gguf_wmma_v4` |
|---|---|---|---|---|
| Layer | Pure CUDA | WMMA tensor cores | WMMA tensor cores | WMMA (prefill only) + custom scalar CUDA (decode) |
| Decode strategy | Per-token host launches | CUDA Graphs (N=1) | Batched speculative + CUDA Graphs fallback | CUDA Graphs (N=1) + CPU/GPU async pipeline |
| KV cache | FP32 | FP32 | FP32 | FP16, read-only cache-path reads |
| Output projection | FP32 | FP32 | FP32 | Q4_0 (via FP16 intermediate) |
| QKV projection | Separate GEMVs | Fused WMMA GEMM | Separate WMMA GEMVs | Fused, 2-warp split-K |
| Speculative decoding | No | No | Yes (self-speculative, n-gram draft) | No |
| Sampling | Full-vocab CPU softmax | Full-vocab CPU softmax | Full-vocab CPU softmax | Prefix-sum-localized CPU softmax |
| Status | Baseline, stable | Stable reference point | Retired | **Current, actively maintained** |

Throughput figures for each runner reflect the stage of the project at which that runner was the active one — see the Runtime Engineering Diary for the authoritative current numbers, since decode performance has moved substantially since the earlier runners were last the primary target of optimization.
