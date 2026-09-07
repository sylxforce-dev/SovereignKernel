# CUDA Runners — What Each One Actually Does

SovereignKernel has three GPU inference entry points. They are not three versions of the same thing — each represents a different point in the optimization path, and they use meaningfully different execution strategies. This document describes what each one does, concretely, based on the code.

---

## 1. `real_runner_gguf_cuda.cu` — Pure CUDA baseline

**Location:** `src_cuda/real_runner_gguf_cuda.cu`
**Layer:** Pure CUDA (no tensor cores)

This is the straightforward GPU forward pass: GGUF/Q4_0 weights loaded directly into VRAM, standard scalar/vectorized CUDA kernels, no CUDA Graphs, no speculative decoding. Each generated token issues its own sequence of kernel launches from the host, one token at a time.

**What's in it:**

- **FLOAT4-vectorized RMSNorm** (`rmsnorm_kernel_float4`) and a fused **Add+RMSNorm** kernel (`add_and_rmsnorm_kernel_float4`) — both read/write in `float4` (16-byte) chunks instead of scalar floats, cutting memory transaction count by 4x for the norm passes.
- **FlashAttention-lite with online softmax** (`mha_kv_cache_kernel`) — one block per attention head, each thread scans a slice of the KV cache and maintains a running max/sum (the online-softmax trick from FlashAttention) instead of materializing the full score row before normalizing. Grouped-query attention is handled directly (`heads_per_kv_group` maps each query head to its shared KV head).
- **RoPE + KV cache write fused into one kernel** (`rope_and_cache_k_kernel`, referenced in the truncated middle section) — rotates K in-place and writes it straight into the per-layer KV cache slot in the same launch.
- **Q4_0 GEMV** for all linear layers (`tensor_math_quantized_cuda::launch_gemv_q4_0`), including a fused **W1+W3+SwiGLU** kernel for the FFN up-projection (`launch_gemv_q4_0_w1_w3_swiglu`) — computes both FFN branches and applies SiLU-gating in one launch instead of three separate kernel calls.
- **No CUDA Graphs.** `forward_pass_gguf_cuda()` is called directly from the host generation loop on every step — the kernel launch sequence is re-issued from scratch each token, with the usual host-side launch overhead this implies.
- **Runtime config reload every turn** — `load_runtime_config()` and `load_system_prompt()` are called fresh on every user input, so temperature/top-p/system prompt can be edited between messages without restarting.

**Sampling:** temperature + top-p (nucleus) with repetition penalty, same `sample_token()` logic reused across all three runners.

**Role:** this is the correctness/performance baseline for the CUDA path — the thing the CUDA Graphs and WMMA runners are measured against.

---

## 2. `real_runner_gguf_wmma.cu` — "V1 Gold Standard": CUDA Graphs + Tensor Cores

**Location:** `src_cuda_tensor/real_runner_gguf_wmma.cu`
**Layer:** CUDA Tensor Core (WMMA), built on top of the pure-CUDA GGUF loading code

This is the stable, hardened decode path. Two structural changes from the baseline runner:

**a) CUDA Graphs for N=1 decode.** The entire per-token decode forward pass (`forward_pass_decode_cuda`) is captured once into a CUDA Graph on the first generated token (`cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaGraphInstantiate`), then replayed on every subsequent token via `cudaGraphLaunch`. This collapses dozens of individual kernel-launch host round-trips into a single graph-launch call per token, which is where most of the host-side overhead in the baseline runner disappears.

**b) PTX-level L2 cache pinning on RMSNorm.** `rmsnorm_kernel_ptx_l2_locked` and `add_and_rmsnorm_kernel_ptx_l2_locked` use inline PTX (`st.cg.global.v4.f32`) instead of a normal store — `.cg` ("cache global") tells the hardware to cache the write at the L2 level and not evict it prematurely, since the very next kernel in the pipeline reads it straight back. This is a deliberate hand-tuned choice, not something the compiler does automatically at `-O3`.

**c) WMMA tensor-core GEMM for the fused QKV projection.** `tensor_math_quantized_wmma::launch_gemm_qkv_fused_wmma` computes Q, K, and V projections from a single fused tensor-core GEMM call rather than three separate GEMVs.

**d) Batched prefill, separate from decode.** `forward_pass_batched_prefill()` handles the initial prompt (N > 1 tokens) as one batched pass through the network, distinct from the N=1 CUDA-Graph decode path used for every token generated afterward.

**e) Stop-sequence detection with a rolling buffer** — checks the last 64 output characters against `<|user|>` to catch the model trying to hallucinate a new turn, and trims/withholds partial output before it prints.

**Role:** this is the "known good" reference implementation for the tensor-core path — no experimental logic, just the CUDA-Graphs + WMMA + PTX-cache-pinning combination validated and left alone.

---

## 3. `real_runner_gguf_wmma_v2.cu` — "T4 Speculative Decoder" (experimental)

**Location:** `src_cuda_tensor/real_runner_gguf_wmma_v2.cu`
**Layer:** CUDA Tensor Core (WMMA), same foundation as `_wmma.cu` — actively diverging from it

This is not a replacement for the file above — it's a parallel experiment layered on the same WMMA foundation, adding **self-speculative decoding**. Diffing the two files shows the actual changes:

**a) N-gram draft matching.** `find_ngram_match()` scans the already-generated token sequence backwards for the most recent occurrence of the last 3 tokens, and if found, proposes the tokens that followed that match last time as a draft continuation (up to 4 candidate tokens). This is *self*-speculative — there's no separate small draft model, the draft comes from the model's own generation history.

**b) Batched verification instead of N=1 steps.** When a draft exists, `forward_pass_batched_speculative()` runs the drafted tokens through the network in a single batched WMMA forward pass (instead of one sequential CUDA-Graph decode per token), then walks the returned logits token-by-token: each draft token is accepted (`spec_hits++`) if it matches what the model would have actually sampled at that position, and generation falls back to the real sampled token at the first mismatch. This is the same accept/reject principle as standard speculative decoding, just with a free self-generated draft instead of a separate draft model.

**c) `pos_offset` plumbed through RoPE and KV-cache-write kernels.** `rope_q_batched_kernel`, `rope_k_inplace_batched_kernel`, and `float2half_copy_batched_kernel` all gained a `pos_offset` parameter so a batch of N speculative tokens gets written to the *correct* absolute KV-cache positions, not positions 0..N-1 — necessary because speculative batches don't start at position 0 the way prefill does.

**d) RMSNorm reverted to plain FLOAT4, not PTX-locked.** `rmsnorm_kernel_float4` / `add_and_rmsnorm_kernel_float4` here are the same simple form as the pure-CUDA baseline runner, not the `st.cg`-pinned version from `_wmma.cu` — this file is mid-iteration, not yet carrying forward every optimization from the "gold standard" runner.

**e) Multi-logit output buffer.** `host_logits_multi` sized `seq_len * vocab_size` — needed because a batched speculative pass returns logits for every position in the batch, not just the last one (each position's logits are needed to check whether the draft token matches what would actually have been sampled there).

**f) Telemetry reports `spec_hits`** — the final performance line includes a "Speculative Free Tokens" count, i.e. how many tokens were generated "for free" via an accepted draft match rather than a full sequential decode step.

**Status: work in progress.** Code comments in this file are marked `// [T4 EXPERIMENT]` and `// --- TAASTATUD: ... ---` ("RESTORED: ..."), indicating pieces of functionality (prefill-latency timing, the 20-token telemetry window, the final report) were rebuilt after being lost or broken at some point — this file has not settled the way `_wmma.cu` has. Whether n-gram speculative decoding gives a real net speedup depends heavily on how repetitive the output is — code and structured text will hit drafts far more often than free-form conversational text, so the win is workload-dependent and hasn't been benchmarked yet.

---

## Summary comparison

| | `real_runner_gguf_cuda` | `real_runner_gguf_wmma` | `real_runner_gguf_wmma_v2` |
|---|---|---|---|
| Layer | Pure CUDA | WMMA tensor cores | WMMA tensor cores |
| Decode strategy | Per-token host launches | CUDA Graphs (N=1) | Batched speculative + CUDA Graphs fallback |
| RMSNorm | FLOAT4 | PTX `st.cg` L2-locked | FLOAT4 (not yet ported to PTX-locked) |
| QKV projection | Separate GEMVs | Fused WMMA GEMM | Separate WMMA GEMVs |
| Speculative decoding | No | No | Yes (self-speculative, n-gram draft) |
| Status | Baseline, stable | Stable ("gold standard") | Experimental, actively changing |
