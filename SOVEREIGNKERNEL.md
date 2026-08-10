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

**CUDA backend: toolchain wired up, kernel logic not yet written.** The build already compiles and links against the CUDA Toolkit (`sovereign_kernel` is built from `.cu` files, `CUDA_SEPARABLE_COMPILATION` is on, `find_package(CUDAToolkit REQUIRED)` is required to configure at all) — but the actual GPU kernels (matmul, RMSNorm, RoPE, attention) haven't been written yet. That's the next phase of work, following the same one-piece-at-a-time validation approach used throughout the CPU work (see "Design decisions" below).

**Important build note:** `find_package(CUDAToolkit REQUIRED)` is currently unconditional — this means an NVIDIA GPU + CUDA Toolkit 12.8 is required to configure and build this project **at all**, even if all you want is the CPU inference path. There's no `ENABLE_CUDA=OFF` escape hatch yet. This will likely become optional once the CUDA kernels are actually doing work and there's a real CPU-vs-GPU tradeoff to make — right now it's just unused scaffolding that happens to gate the whole build.

## If you want to actually build and run this yourself

This is a personal learning project on one specific machine, not a packaged release, so getting it running elsewhere takes real work, not just a `git clone` + `cmake --build`. Expect to hit and fix all of the following, roughly in this order:

1. **Install the exact toolchain**: CUDA Toolkit 12.8 specifically (not newer), MSVC via VS2022 Build Tools (not VS2026), CMake, and an OpenMP-capable compiler. Mismatched CUDA/VS versions are a common source of `cmake configure` failures that have nothing to do with this project's code — see "Design decisions" below for why 12.8 is pinned.
2. **You need an NVIDIA GPU**, even for CPU-only inference, per the build note above. AMD/Intel GPU or CPU-only machines cannot currently configure this project.
3. **Get your own model weights.** None are included in the repo (`.gitignore` deliberately excludes `.bin`/`.gguf`/`.safetensors`/`.pt`). You'll need to source `stories110M.bin` (Karpathy's original release) and a TinyLlama-1.1B-Chat GGUF Q4_0 quantization yourself, plus matching tokenizer files.
4. **Fix the hardcoded absolute paths.** `real_runner.cpp` and `real_runner2.cpp` both have Windows paths baked into `main()` (e.g. `C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/...`) pointing at my machine. Edit these to point at wherever you put your own weight/tokenizer files.
5. **Match or change the thread count.** `omp_set_num_threads(8)` is hardcoded to this machine's physical core count (Ryzen 7 7700). Set it to your own physical (not logical/SMT) core count, or performance will be worse than it should be.
6. **Confirm AVX2 support.** `CMakeLists.txt` passes `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang) unconditionally to both CPU executable targets. Most x86 CPUs from the last ~decade have this, but older CPUs and non-x86 hardware (ARM/Apple Silicon) don't — the build will fail to compile or crash at runtime with an illegal-instruction fault. If that's you, drop those flags to fall back to the scalar path already present via `#if defined(__AVX2__)` guards, or add real runtime CPU-feature detection.
7. **Recompile for your own GPU.** `CMAKE_CUDA_ARCHITECTURES native` targets only the GPU present at compile time (currently an RTX 5060 Ti). A binary built on my machine won't run on a different GPU generation — you need to build on your own target machine, or set an explicit architecture list.

None of this is conceptually hard — it's just genuinely not abstracted away yet, because there's been no second machine to test any of it against. Treat this as source you compile for your own setup, not a binary you run as-is.

### Tuning for your own CPU (exact locations)

The CPU performance numbers below are specific to a Ryzen 7 7700 (8 physical cores, AVX2, single-channel DDR5). If you're running on different hardware, these are the exact places to change:

- **Thread count** — `omp_set_num_threads(8);` appears in `main()` in both `src/real_runner.cpp` (line ~247) and `src/real_runner2.cpp` (line ~339). Change `8` to your CPU's **physical** core count (check with Task Manager → Performance → CPU → "Cores", not "Logical processors" — using the logical/SMT count is usually slower here, not faster, per the `OMP_WAIT_POLICY` finding below).
- **AVX2 compile flags** — `CMakeLists.txt`, in the `model_runner` and `model_runner2` target blocks: `target_compile_options(... /arch:AVX2)` (MSVC) or `-mavx2 -mfma` (GCC/Clang). If your CPU is AVX2-capable (basically anything from the last decade, Intel Haswell/AMD Excavator or newer), leave these as-is — this is what's giving you the SIMD speedup. If you hit an illegal-instruction crash, your CPU doesn't support AVX2 and you'll need to remove these flags (falls back to the scalar path already present in `tensor_math_cpu.h` / `tensor_math_quantized.h`).
- **Don't set `OMP_WAIT_POLICY=active`** as an environment variable — it was tested on this hardware and made things measurably *worse* (busy-spin threads compete with the matmul for scarce memory bandwidth on single-channel RAM). If you have dual/quad-channel RAM this tradeoff may differ, but it wasn't re-tested there.
- There's no runtime CPU-feature auto-detection anywhere in the codebase — every one of the above is a manual, compile-time decision you make for your own machine.

### What's actually implemented (CPU path)

- **Tokenizer**: real byte-level BPE (character-level init + iterative adjacent-pair merging by vocab score) — not a greedy longest-match approximation.
- **Attention**: real scaled dot-product attention per head, causal masking, softmax, with a per-layer KV cache (`vector<TensorKVCache>`, one per layer, so K/V from different layers don't get mixed together). The per-head loop is parallelized across CPU cores (OpenMP), not run serially.
- **RoPE**: applied per-head, in two different rotation conventions depending on the checkpoint's origin — adjacent-pair rotation `(0,1),(2,3)...` for the Karpathy `.bin` path, and split-half rotation (HuggingFace/Llama-2 convention) for the GGUF/TinyLlama path. Both are implemented as allocation-free in-place functions that write directly into pre-allocated buffers.
- **FFN**: SwiGLU, `FFN(x) = (SiLU(x·W1) ⊙ (x·W3)) · W2` — matches the LLaMA-style formula, including which branch actually gets the SiLU activation.
- **Weight loading**: reads Karpathy's `.bin` checkpoint format directly, weight-type-major across all layers (matching the actual on-disk layout, not a naive per-layer loop); and reads GGUF files with Q4_0 dequantization + AVX2-accelerated matmul.
- **GGUF support**: loads GGUF files and performs Q4_0 dequantization + matmul on CPU, with AVX2 SIMD intrinsics for both the Q4_0 path and the FP32 `.bin` path. Includes a lightweight "Sovereign Telemetry" sanity check after key stages of the first forward pass (min/max range per tensor, flags NaN/Inf) — this is a basic debug aid, not a security mechanism.

Every one of the above went through at least one real, non-trivial bug during development (wrong FFN weight shapes, attention that computed Q/K/V+RoPE but never actually used them, SiLU applied to the wrong branch, a tokenizer that "worked" but wasn't real BPE, an AVX2 compile flag silently missing from one of the two build targets so that path ran in scalar fallback). They were caught by comparing actual output against expected behavior, not just checking that the code compiled and ran.

## Design decisions

**Toolchain is locked to CUDA 12.8**, not the latest release. The reason: the rest of my local-AI stack (llama.cpp builds, GGUF tooling, other projects) is compiled and tested against 12.8, and upgrading system-wide risks breaking that stack through PATH resolution order. Since CUDA 12.8 doesn't support newer Visual Studio toolsets, the project uses VS2022 Build Tools specifically (not the newest VS release) — this is an officially supported CUDA+VS combination, chosen deliberately rather than fought around.

**Kernels — CPU and, next, CUDA — are built and validated one at a time**, smallest-to-largest. Each new piece of math (matmul, RMSNorm, RoPE, attention) is:
1. Tested on a tiny example (2×2 / 3×3) before scaling up.
2. Checked with `cudaGetLastError()` / `cudaDeviceSynchronize()` and `compute-sanitizer` (for the CUDA phase specifically).
3. Compared numerically against a known-correct reference (tolerance ~1e-4 to 1e-3 for FP32) before moving to the next piece.

This is slower than writing everything at once, but it means a bug shows up next to the one component that could have caused it, instead of somewhere in a pile of interacting kernels. The same discipline applied during the CPU optimization pass below: every step was checked against the "Sovereign Telemetry" output to confirm numerical output stayed identical before and after each change.

**No heap allocation in the hot path.** The single biggest lesson from the CPU optimization pass: allocating a new buffer (`Tensor`) inside a loop that runs hundreds or thousands of times per generated token is far more expensive than the actual math. Every per-layer and per-head buffer (Q/K/V projections, attention scores, FFN intermediates) is now allocated once and reused across the whole generation loop via `thread_local` scratch buffers, with all core math functions (`matmul`, RoPE) provided in both an allocating (`Tensor`-returning) form and an allocation-free in-place form that writes directly into a caller-supplied pointer.

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

The CPU path's role going forward is as a **correctness reference and performance floor** for the CUDA backend — the target for CUDA is meaningfully higher than what CPU can reach on this hardware, not just "however fast CPU happens to be."

## Roadmap

- [x] Tensor primitives, CPU matmul/RMSNorm/RoPE/attention/FFN
- [x] BPE tokenizer
- [x] `.bin` weight loading + full forward pass
- [x] GGUF loading + Q4_0 dequantization on CPU
- [x] CPU performance pass — allocation-free hot path, AVX2 confirmed active on both formats, CPU speed brought in line with llama.cpp's CPU backend
- [ ] CUDA matmul kernel (first, since everything else depends on it)
- [ ] CUDA RMSNorm kernel
- [ ] CUDA RoPE kernel
- [ ] CUDA attention kernel
- [ ] End-to-end CUDA forward pass

**Performance target**: at least ~50 tok/s on TinyLlama-1.1B on the RTX 5060 Ti — comparable to what llama.cpp's CUDA backend typically achieves on hardware in this class (~50–60 tok/s). This is treated as a pass/fail signal, not a stretch goal: if the CUDA kernels land far below this once complete, it means something structural is wrong (memory access pattern, kernel launch overhead, a synchronization issue) rather than "needs more tuning." An initial working version in the 25–30 tok/s range would already be a strong signal that the architecture is sound, even if reaching 50 takes further optimization.

## Why

I wanted to actually understand — not just use — the mechanics behind LLM inference: how weights get quantized and dequantized, how attention and KV caching really work under the hood, what makes memory bandwidth and allocation overhead the bottlenecks they are, and what it takes to make all of that run fast on both CPU and GPU. This project is that process, done in the open.
