 SovereignKernel
 
A custom C++/CUDA LLM inference runtime, built from scratch as a learning project.
 
**In plain terms:** most people run language models through frameworks like PyTorch or llama.cpp, which handle all the underlying math and memory management for you. This project does that work by hand instead — tokenizing text, running it through the transformer's layers, and generating output tokens, using code written and understood line by line, without relying on an existing inference library.
 
This is **not** an attempt to compete with llama.cpp — it exists to answer one question: *what does it actually take to run a transformer, end-to-end, on your own hardware?*
 
## Status at a glance
 
- ✅ **CPU inference: working and optimized.** Runs two real models (TinyLlama-1.1B and Karpathy's stories110M) end-to-end, at speeds matching llama.cpp's own CPU backend on the same hardware.
- ✅ **CUDA (GPU) backend: Phase 1 complete and locked.** Full GGUF Q4_0 end-to-end inference running entirely GPU-resident (weights loaded directly into VRAM, no CPU streaming during generation). Stabilized at **151.48 tok/s** on an RTX 5060 Ti — the practical ceiling for standard CUDA cores (no Tensor Cores yet) on this GPU/model combination. Full optimization journey (8 measured phases, including two documented dead ends) is in the [CUDA optimization diary](./docs/SovereignKernel_CUDA_Optimization_Diary.md).
- 🚧 **Phase 2 — Tensor Cores (WMMA):** next up. Target is to push past the scalar-core ceiling toward 200+ tok/s by moving from SIMT to Tensor Core matrix execution.
- ⚠️ **Not a plug-and-play tool.** Building and running this yourself requires the CUDA Toolkit, a matching GPU, and your own model weight files — see below.
## Want the technical deep-dive?
 
Architecture, design decisions, performance numbers, exact build requirements, and the roadmap all live in **[SOVEREIGNKERNEL.md](./SOVEREIGNKERNEL.md)**.
 
For the full CUDA optimization story — every kernel rewrite, every benchmark, and the two experiments that didn't work — see **[docs/SovereignKernel_CUDA_Optimization_Diary.md](./docs/SovereignKernel_CUDA_Optimization_Diary.md)**.
 
## Why
 
I wanted to actually understand — not just use — the mechanics behind LLM inference: how weights get quantized and dequantized, how attention and KV caching really work under the hood, what makes memory bandwidth and allocation overhead the bottlenecks they are, and what it takes to make all of that run fast on both CPU and GPU. This project is that process, done in the open.
 
