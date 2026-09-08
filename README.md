# SovereignKernel

A custom C++/CUDA LLM inference runtime, built from scratch as a learning project.

This is **not** an attempt to compete with llama.cpp. It exists to answer one question: *what does it actually take to run a transformer, end-to-end, on your own hardware, with code you wrote and understand line by line?*

## Why

I wanted to actually understand — not just use — the mechanics behind LLM inference: how weights get quantized and dequantized, how attention and KV caching really work under the hood, what makes memory bandwidth and allocation overhead the bottlenecks they are, and what it takes to make all of that run fast on both CPU and GPU.

This project is that process, done in the open.

## What this is

SovereignKernel implements the full inference path for transformer language models — tokenization, embeddings, attention, RoPE, feed-forward, KV caching, and quantized weight loading — without relying on an existing inference library.

The goal is a minimal but correct Tensor Engine with both CPU and CUDA backends.

**Reference hardware:** RTX 5060 Ti (8GB VRAM), Ryzen 7 7700, 16GB RAM (single-channel).

The reference system is used for development and benchmarking. The CUDA backend is architecture-targeted at build time rather than being hardcoded to the RTX 5060 Ti.

## ⚠️ Important Execution Warning

If you are planning to compile, configure, or run this repository, you must read this first:
👉 **[MUST READ BEFORE ATTEMPTING](MUST_READ_BEFORE_ATTEMPTING.md)**
*(Covers hardcoded absolute paths, strict model support for TinyLlama/Karpathy, and AVX2/OpenMP core constraints).*

## Models

Two models are used for development and testing:

- **TinyLlama-1.1B-Chat**
  - dim=2048
  - hidden=5632
  - 22 layers
  - 32 heads / 4 KV heads
  - vocab=32000
  - GGUF / Q4_0
  - Primary target model

- **Andrej Karpathy's `stories110M`**
  - Original Karpathy `.bin` checkpoint format
  - Smaller model used for fast iteration and sanity checks

## Architecture & Documentation Hub

The state, metrics, and evolution of the engine are maintained in dedicated runtime diaries. **Read them in this order** — each builds on context from the one before it:

### 1. [CUDA Optimization Diary](docs/SovereignKernel_CUDA_Optimization_Diary.md)
Historical record of Phase 1: the step-by-step optimization of the pure scalar-CUDA implementation, from an initial 53 tok/s up to a **151.48 tok/s** milestone that was, at the time, treated as the practical ceiling for standard CUDA cores. This is a point-in-time log of that phase — the 151 tok/s figure was later exceeded (see below) once Tensor Core work began; it's kept here unedited as the historical record of how Phase 1 actually happened, negative results included.

### 2. [Runtime Diary & Architecture Audit v1.7](docs/Sovereign_Kernel_Runtime_Diary_v1_7.md)
The definitive record of the engine's **current** capabilities: pure CUDA execution has since exceeded 200 tok/s, the Tensor/Hybrid Decode path reached a validated 226.969 tok/s milestone and now sits in a stable **240–260 tok/s** steady-state class, and fused WMMA prefill has reached ~75.7ms on a representative prompt. This document supersedes the throughput ceiling implied by the Optimization Diary above — read it for where the engine actually stands today.

### 3. [CUDA Runners Comparison](docs/CUDA_RUNNERS.md)
A structural, code-level breakdown of the three active GPU inference entry points (Pure CUDA baseline, WMMA V1 "Gold Standard", and the actively-iterated speculative-decoding variant), with current measured throughput for each. Reading this last makes the most sense — the per-runner comparison here draws on the history and current state established in the two diaries above.

---
*Status: Phase 2 (Runtime Baseline) SEALED. Phase 3 (Agentic Orchestration & Retrieval) OPEN.*
