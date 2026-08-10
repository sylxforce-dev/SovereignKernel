# SovereignKernel

A custom C++/CUDA LLM inference runtime, built from scratch as a learning project.

**In plain terms:** most people run language models through frameworks like PyTorch or llama.cpp, which handle all the underlying math and memory management for you. This project does that work by hand instead — tokenizing text, running it through the transformer's layers, and generating output tokens, using code written and understood line by line, without relying on an existing inference library.

This is **not** an attempt to compete with llama.cpp — it exists to answer one question: *what does it actually take to run a transformer, end-to-end, on your own hardware?*

## Status at a glance

- ✅ **CPU inference: working and optimized.** Runs two real models (TinyLlama-1.1B and Karpathy's stories110M) end-to-end, at speeds matching llama.cpp's own CPU backend on the same hardware.
- 🚧 **CUDA (GPU) backend: build toolchain is wired up, kernels not written yet.** This is the current phase of work.
- ⚠️ **Not a plug-and-play tool.** Building and running this yourself requires the CUDA Toolkit, a matching GPU, and your own model weight files — see below.

## Want the technical deep-dive?

Architecture, design decisions, performance numbers, exact build requirements, and the roadmap all live in **[SOVEREIGNKERNEL.md](./SOVEREIGNKERNEL.md)**.

## Why

I wanted to actually understand — not just use — the mechanics behind LLM inference: how weights get quantized and dequantized, how attention and KV caching really work under the hood, what makes memory bandwidth and allocation overhead the bottlenecks they are, and what it takes to make all of that run fast on both CPU and GPU. This project is that process, done in the open.
