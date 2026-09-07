# SovereignKernel

A custom C++/CUDA LLM inference runtime, built from scratch as a learning project.

This is **not** an attempt to compete with llama.cpp. It exists to answer one question: *what does it actually take to run a transformer, end-to-end, on your own hardware, with code you wrote and understand line by line?*

## What this is

SovereignKernel implements the full inference path for transformer language models — tokenization, embeddings, attention, RoPE, feed-forward, KV caching, and quantized weight loading — without relying on an existing inference library. The goal is a minimal but *correct* Tensor Engine with both CPU and CUDA backends.

Target hardware: RTX 5060 Ti (8GB VRAM), Ryzen 7 7700, 16GB RAM (single-channel).

Two models are used for development and testing:
- **TinyLlama-1.1B-Chat** (dim=2048, hidden=5632, 22 layers, 32 heads / 4 KV heads, vocab=32000) — the real target model, loaded from GGUF/Q4_0.
- **Andrej Karpathy's `stories110M`** — a much smaller model used as a fast iteration/sanity-check target, loaded from the original Karpathy `.bin` checkpoint format.

## Current status

**CPU backend: functional and optimized.** Full forward passes run correctly on both models, on both weight formats, producing coherent output — and after a dedicated optimization pass, CPU inference speed is now in the same range as llama.cpp's own CPU backend on identical hardware and identical GGUF weights.

**CUDA backend: functional.** Full GGUF/Q4_0 forward passes run end-to-end on GPU, using CUDA Graphs for the N=1 decode step and a fused batched path for prefill.

**CUDA Tensor Core (WMMA) backend: functional.** A second GPU path built on top of the pure-CUDA layer, using WMMA tensor-core GEMM for the Q4_0 matmuls and a batched flash-attention kernel. An experimental variant adds self-speculative decoding (n-gram lookup against the generated sequence, verified via batched WMMA GEMM) — this piece is still being iterated on.

## Portability — this is currently tuned for one specific machine

This is a personal learning project, not a general-purpose library, and it currently reflects that directly in the code. If you clone this and try to run it on different hardware, expect to have to change a few things by hand:

- **Thread count is hardcoded.** `omp_set_num_threads(8)` in both `main()` entry points assumes 8 physical cores (this machine's Ryzen 7 7700). On a CPU with a different core count, change this to your own physical core count — not your logical/SMT thread count, which is usually double the physical count and will make things slower, not faster (see the "Performance" section below for why).
- **AVX2 is a hard compile-time requirement, not a runtime-checked fallback.** `CMakeLists.txt` passes `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang) unconditionally to both CPU executable targets. Most x86 CPUs from the last decade support this, but older CPUs and non-x86 hardware (ARM, including Apple Silicon) do not — the build will either fail to compile or crash at runtime with an illegal-instruction fault on unsupported hardware. If you're on such hardware, you'd need to either drop those flags (falls back to the scalar path already present in `tensor_math_cpu.h` / `tensor_math_quantized.h` via `#if defined(__AVX2__)`) or add proper runtime CPU-feature detection.
- **`CMAKE_CUDA_ARCHITECTURES native` targets only the GPU in the machine that compiles it** (currently an RTX 5060 Ti). The resulting binary won't run on a different GPU generation — recompiling on the target machine, or setting an explicit architecture list, is required. The WMMA tensor-core path additionally requires a GPU generation that supports tensor cores (Volta/Turing or newer).
- **Model and tokenizer file paths are hardcoded** as absolute Windows paths inside `main()` in the runner entry points (e.g. `C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/...`). You'll need to point these at wherever you've placed the `.bin`/GGUF weights and tokenizer file on your own machine.
- **CUDA Toolkit is pinned to 12.8** specifically — see "Design decisions" below for why. Other CUDA versions may or may not work; this hasn't been tested.

None of this is hard to fix for your own machine — it's just not abstracted away yet, because there's been no second machine to abstract it for.

### What's actually implemented (CPU path)

- **Tokenizer**: real byte-level BPE (character-level init + iterative adjacent-pair merging by vocab score) — not a greedy longest-match approximation.
- **Attention**: real scaled dot-product attention per head, causal masking, softmax, with a per-layer KV cache (`vector<TensorKVCache>`, one per layer, so K/V from different layers don't get mixed together). The per-head loop is parallelized across CPU cores (OpenMP), not run serially.
- **RoPE**: applied per-head, in two different rotation conventions depending on the checkpoint's origin — adjacent-pair rotation `(0,1),(2,3)...` for the Karpathy `.bin` path, and split-half rotation (HuggingFace/Llama-2 convention) for the GGUF/TinyLlama path. Both are implemented as allocation-free in-place functions that write directly into pre-allocated buffers.
- **FFN**: SwiGLU, `FFN(x) = (SiLU(x·W1) ⊙ (x·W3)) · W2` — matches the LLaMA-style formula, including which branch actually gets the SiLU activation.
- **Weight loading**: reads Karpathy's `.bin` checkpoint format directly, weight-type-major across all layers (matching the actual on-disk layout, not a naive per-layer loop); and reads GGUF files with Q4_0 dequantization + AVX2-accelerated matmul.
- **GGUF support**: loads GGUF files and performs Q4_0 dequantization + matmul on CPU, with AVX2 SIMD intrinsics for both the Q4_0 path and the FP32 `.bin` path. Includes a lightweight "Sovereign Telemetry" sanity check after key stages of the first forward pass (min/max range per tensor, flags NaN/Inf) — this is a basic debug aid, not a security mechanism.

Every one of the above went through at least one real, non-trivial bug during development (wrong FFN weight shapes, attention that computed Q/K/V+RoPE but never actually used them, SiLU applied to the wrong branch, a tokenizer that "worked" but wasn't real BPE, an AVX2 compile flag silently missing from one of the two build targets so that path ran in scalar fallback). They were caught by comparing actual output against expected behavior, not just checking that the code compiled and ran.

### What's actually implemented (CUDA path)

- **Pure CUDA (`src_cuda/`, `include_cuda/`)**: GGUF/Q4_0 weight loading directly to VRAM, N=1 decode via CUDA Graphs (captured once, replayed per token to cut launch overhead), a fused batched prefill path for the initial prompt, and PTX-level L2 cache pinning (`st.cg`) on the RMSNorm kernels to avoid needlessly evicting hot data.
- **CUDA Tensor Core / WMMA (`src_cuda_tensor/`, `include_cuda_tensor/`)**: builds on top of the pure-CUDA layer — WMMA-based Q4_0 GEMM/GEMV for the linear layers, a batched flash-attention kernel, and a fused QKV projection. An experimental second entry point adds self-speculative decoding: it looks up n-gram matches in the already-generated sequence, drafts candidate continuation tokens from that match, and verifies them in a single batched WMMA forward pass instead of one sequential N=1 step per token.

Both CUDA layers share the same CPU-side core library (`sovereign_kernel`, built from `include/`) for tensor definitions, KV cache structure, and config/tokenizer handling — they're not a from-scratch reimplementation, they sit on top of the same foundation as the CPU path.

## Design decisions

**Toolchain is locked to CUDA 12.8**, not the latest release. The reason: the rest of my local-AI stack (llama.cpp builds, GGUF tooling, other projects) is compiled and tested against 12.8, and upgrading system-wide risks breaking that stack through PATH resolution order. Since CUDA 12.8 doesn't support newer Visual Studio toolsets, the project uses VS2022 Build Tools specifically (not the newest VS release) — this is an officially supported CUDA+VS combination, chosen deliberately rather than fought around.

**Kernels — CPU and CUDA — are built and validated one at a time**, smallest-to-largest. Each new piece of math (matmul, RMSNorm, RoPE, attention) is:
1. Tested on a tiny example (2×2 / 3×3) before scaling up.
2. Checked with `cudaGetLastError()` / `cudaDeviceSynchronize()` and `compute-sanitizer` (for the CUDA phase specifically).
3. Compared numerically against a known-correct reference (tolerance ~1e-4 to 1e-3 for FP32) before moving to the next piece.

This is slower than writing everything at once, but it means a bug shows up next to the one component that could have caused it, instead of somewhere in a pile of interacting kernels. The same discipline applied during the CPU optimization pass below: every step was checked against the "Sovereign Telemetry" output to confirm numerical output stayed identical before and after each change.

**No heap allocation in the hot path.** The single biggest lesson from the CPU optimization pass: allocating a new buffer (`Tensor`) inside a loop that runs hundreds or thousands of times per generated token is far more expensive than the actual math. Every per-layer and per-head buffer (Q/K/V projections, attention scores, FFN intermediates) is now allocated once and reused across the whole generation loop via `thread_local` scratch buffers, with all core math functions (`matmul`, RoPE) provided in both an allocating (`Tensor`-returning) form and an allocation-free in-place form that writes directly into a caller-supplied pointer. The CUDA decode path applies the same principle at the GPU level: buffers are allocated once and the whole per-token decode graph is captured via CUDA Graphs rather than re-issuing kernel launches from the host every step.

**Single repo, layered dependency structure, not split into separate projects.** The CPU core (`include/`, `src/`), pure-CUDA layer, and CUDA Tensor/WMMA layer live in one repository rather than three. The dependency chain runs one direction — pure CUDA depends on the CPU core, and the Tensor/WMMA layer depends on both the CPU core and the pure-CUDA GGUF-loading code — so there's no clean way to build the Tensor layer in isolation anyway. Keeping it as one repo with organized folders (`include/` / `include_cuda/` / `include_cuda_tensor/`, mirrored under `src/`) avoids cross-repo submodule management for something that's a single dependency stack, not three independent siblings.

## Performance

CPU numbers, measured on the same machine (Ryzen 7 7700, single-channel 16GB RAM), Release build:

| Model | Format | Speed |
|---|---|---|
| TinyLlama-1.1B | GGUF (Q4_0) | ~33–40 tok/s |
| stories110M | `.bin` (FP32) | ~62 tok/s |

For reference, llama.cpp's CPU-only backend achieves ~40–41 tok/s on the identical TinyLlama GGUF/Q4_0 weights on this same machine — SovereignKernel's CPU path is now in that same range, not an order of magnitude behind it.

**Getting here took a focused optimization pass**, in order of actual impact:
1. **Compiler flag bug (biggest single jump)**: the build system specified `-O3` for Release builds, which is GCC/Clang syntax that MSVC's `cl.exe` silently ignores (with a `D9002` warning) rather than erroring on — meaning Release builds were never actually optimized. Fixed with an `if(MSVC) /O2 else -O3` branch.
2. **Eliminating heap allocations from the hot path**: matmul, RoPE, and per-head attention scoring all originally allocated a new `Tensor` on every call — for a 22-layer, 32-head model, this meant tens of thousands of allocations per generated token. Replaced with allocation-free in-place variants writing into pre-allocated `thread_local` buffers.
3. **Parallelizing the attention head loop**: attention across heads was running serially on one core while the rest sat idle; wrapped in `#pragma omp parallel for`.
4. **Thread count and affinity tuning**: pinning to 8 threads (physical core count, not the 16 logical SMT threads) via `omp_set_num_threads(8)` helped. Notably, `OMP_WAIT_POLICY=active` (busy-spin instead of sleeping between parallel regions) made things *worse* on this hardware — the spin-wait traffic competes with the already-scarce single-channel memory bandwidth that the actual matmul needs.
5. **A missing AVX2 compile flag** on one of the two executable targets, silently leaving that whole code path running in scalar fallback instead of SIMD.

Known remaining inefficiencies (not yet addressed, lower priority at current speed): a handful of smaller per-layer `Tensor` allocations remain (RMSNorm output, FFN intermediate buffers, attention concat buffer) — roughly 8 per layer, much smaller in impact than the matmul/RoPE allocations that were already removed. Matmul cache blocking/tiling (splitting the weight matrix into cache-sized tiles instead of streaming full rows) also remains untried.

The CPU path's role going forward is as a **correctness reference and performance floor** for the CUDA backend — the target for CUDA is meaningfully higher than what CPU can reach on this hardware, not just "however fast CPU happens to be." CUDA-side tok/s numbers aren't published here yet pending a proper measurement pass across both the pure-CUDA and WMMA paths.

## Roadmap

- [x] Tensor primitives, CPU matmul/RMSNorm/RoPE/attention/FFN
- [x] BPE tokenizer
- [x] `.bin` weight loading + full forward pass
- [x] GGUF loading + Q4_0 dequantization on CPU
- [x] CPU performance pass — allocation-free hot path, AVX2 confirmed active on both formats, CPU speed brought in line with llama.cpp's CPU backend
- [x] CUDA matmul kernel
- [x] CUDA RMSNorm kernel
- [x] CUDA RoPE kernel
- [x] CUDA attention kernel
- [x] End-to-end CUDA forward pass (N=1 decode via CUDA Graphs + batched prefill)
- [x] CUDA Tensor Core (WMMA) GEMM/GEMV path
- [x] Batched flash-attention kernel (WMMA path)
- [ ] Self-speculative decoding (n-gram draft + batched WMMA verify) — experimental, still being iterated on
- [ ] CUDA-side performance measurement pass and published tok/s numbers
- [ ] Runtime CPU-feature detection (replace hardcoded AVX2 requirement)

**Performance target**: at least ~50 tok/s on TinyLlama-1.1B on the RTX 5060 Ti — comparable to what llama.cpp's CUDA backend typically achieves on hardware in this class (~50–60 tok/s). This is treated as a pass/fail signal, not a stretch goal: if the CUDA kernels land far below this once complete, it means something structural is wrong (memory access pattern, kernel launch overhead, a synchronization issue) rather than "needs more tuning." An initial working version in the 25–30 tok/s range would already be a strong signal that the architecture is sound, even if reaching 50 takes further optimization.

## Why

I wanted to actually understand — not just use — the mechanics behind LLM inference: how weights get quantized and dequantized, how attention and KV caching really work under the hood, what makes memory bandwidth and allocation overhead the bottlenecks they are, and what it takes to make all of that run fast on both CPU and GPU. This project is that process, done in the open.
