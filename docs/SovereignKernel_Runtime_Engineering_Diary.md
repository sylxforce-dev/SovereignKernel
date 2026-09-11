# SovereignKernel

A custom C++/CUDA LLM inference runtime built from scratch as a learning project.

The goal was simple:

> **Understand what actually happens inside LLM inference instead of treating the runtime as a black box.**

SovereignKernel started as a hobby experiment and gradually turned into a complete local inference runtime with its own tokenizer, GGUF loader, CPU path, CUDA kernels, KV cache, attention implementation and GPU execution pipeline.

This document records the engineering process rather than pretending the final implementation appeared fully formed.

---

# Runtime Engineering Diary

## Hardware

- NVIDIA RTX 5060 Ti 8GB
- AMD Ryzen 7 7700
- 16GB DDR5 RAM
- Windows 11

## Test Model

- TinyLlama-1.1B-Chat
- GGUF
- Q4_0
- 22 transformer layers
- 2048 hidden dimension
- 5632 FFN dimension
- GQA attention

---

# 1. Starting Point — CPU Runtime

The first goal was simply to make inference work without relying on an existing runtime.

Implemented from scratch:

- GGUF loading
- Q4_0 weight handling
- byte-level BPE tokenizer
- embeddings
- RMSNorm
- RoPE
- attention
- KV cache
- FFN
- logits generation
- CPU sampling
- OpenMP execution

The CPU implementation became fully functional.

Typical TinyLlama performance:

```text
~33–40 tok/s
```

This was already close to the llama.cpp CPU backend on the same machine.

The important milestone was not the exact number.

It was:

> **The complete inference path worked without using llama.cpp internals.**

---

# 2. CUDA — First Real GPU Runtime

The next step was moving the runtime to CUDA.

The initial CUDA implementation avoided:

- cuBLAS
- CUTLASS
- Tensor Cores
- PyTorch
- existing inference kernels

Instead, the important operations were implemented directly:

- Q4_0 CUDA GEMV
- RMSNorm
- RoPE
- attention
- FFN
- residual operations
- logits projection

The first CUDA versions were considerably slower than the final runtime.

---

# 3. CUDA Optimization Phase

Optimization became an iterative process:

```text
Baseline
   ↓
Measure
   ↓
Find bottleneck
   ↓
Change one thing
   ↓
Benchmark
   ↓
Keep / Revert
```

Several optimization phases gradually pushed performance from roughly:

```text
~53 tok/s
```

to:

```text
151.48 tok/s
```

The important lesson appeared very early:

> An optimization that sounds good is not necessarily an optimization.

---

# 4. Failed Experiments

Not every experiment improved performance.

Some were explicitly tested and rejected.

## CUDA Streams

Asynchronous CUDA streams were tested as a way to overlap operations.

Result:

```text
No useful improvement.
```

The experiment was reverted.

---

## float4 Loads

Vectorized `float4` memory access was tested in the attention path.

The expected theory was:

```text
larger loads
→ fewer memory instructions
→ higher throughput
```

Reality did not justify the expected gain.

The experiment was rejected.

This became an important rule for the project:

> **Do not confuse fewer source-level memory operations with fewer actual GPU memory transactions.**

---

## N-gram Speculative Decoding

A speculative decoding experiment was also tested.

The idea looked attractive:

```text
Generate several likely tokens
→ verify them
→ reduce expensive inference passes
```

In practice, the CPU-side orchestration overhead outweighed the benefit on this workload.

The result was a regression.

It was removed rather than forced into the runtime.

---

# 5. Phase 2 — CUDA Runtime Becomes Stable

After several iterations the runtime reached a much more stable architecture.

Typical performance:

```text
~240–260 tok/s
```

with hot windows reaching approximately:

```text
~267 tok/s
```

The runtime now had:

- custom Q4 GEMV
- CUDA Graphs
- custom GQA attention
- FP16 KV cache
- fused CUDA operations
- GPU-resident inference
- custom RMSNorm
- RoPE
- FFN fusion
- CPU sampling

At this point the focus changed.

The problem was no longer:

> "Can we make CUDA inference work?"

It became:

> **"Where are the remaining milliseconds going?"**

---

# 6. Correctness Before Speed

One of the most important debugging episodes involved WMMA.

A single-vector WMMA implementation produced a huge numerical error:

```text
max error:
7.93
```

That was obviously unacceptable.

The problem was traced to a shared-memory / multi-warp race.

The implementation was changed from multiple warps writing shared arrays to a single warp / 32-thread arrangement.

After the fix:

```text
max error:
0.00142
```

This was a good example of why performance numbers without correctness checks are dangerous.

A fast kernel producing wrong logits is not an optimization.

It's a bug.

---

# 7. Batched WMMA

WMMA was then moved into the places where it actually made sense.

The important distinction became:

### Decode

```text
N = 1
```

Extremely latency-sensitive.

### Prefill

```text
N > 1
```

Much better suited to matrix multiplication and Tensor Core execution.

WMMA decode experiments were not competitive and were not forced into the final decode path.

WMMA remained useful for prefill.

Correctness tests included matrices such as:

```text
M/K/N = 128/128/16
M/K/N = 128/128/5
M/K/N = 256/256/11
M/K/N = 2048/2048/7
```

Errors remained small enough for the intended inference workload.

---

# 8. Flash Attention / GQA

The attention implementation was also tested independently.

Flash-attention style online softmax was introduced to avoid unnecessary intermediate materialization.

The runtime supports TinyLlama's GQA configuration:

```text
32 query heads
4 KV heads
head dimension = 64
```

Attention correctness tests passed.

This allowed the attention path to remain custom rather than replacing it with a vendor library.

---

# 9. CUDA Graphs

CUDA Graphs were introduced to reduce repeated launch overhead.

The decode path contains many small operations, so launching every kernel independently can become expensive.

The graph captures the repeated execution structure and replays it for subsequent tokens.

This was especially relevant for:

```text
N = 1 decode
```

where launch latency matters much more than in large matrix workloads.

---

# 10. The V6 Runtime

After several rounds of optimization, the runtime reached the V6 generation.

The decode pipeline became approximately:

```text
Embedding
    ↓
Attention RMSNorm
    ↓
Fused QKV Q4 GEMV
    ↓
RoPE + KV Cache
    ↓
Custom GQA Attention
    ↓
WO GEMV
    ↓
Attention Residual
    ↓
FFN RMSNorm
    ↓
W1 + W3
    ↓
SwiGLU
    ↓
W2 + Residual
    ↓
Final RMSNorm
    ↓
Output Logits
```

The important thing is that the pipeline is specialized around the actual workload.

It is not a generic dense GEMM pipeline.

---

# 11. V6.3 — Current Known-Good Baseline

V6.3 became the stable control version.

Typical runs:

```text
~285–290 tok/s
```

with individual hot windows exceeding:

```text
300 tok/s
```

Example:

```text
Decode Speed:       288.887 tok/s
Avg Decode Pass:      3.43522 ms/token
Prefill latency:     58.2312 ms
```

Another run:

```text
Decode Speed:       289.643 tok/s
Avg Decode Pass:      3.42498 ms/token
Prefill latency:    177.668 ms
```

Hot windows have reached approximately:

```text
~309 tok/s
```

The important benchmark target became:

```text
3.33 ms/token
≈ 300 tok/s
```

rather than chasing isolated spikes.

---

# 12. The Time Machine Regression

At one point an experimental mutation caused performance to fall to approximately:

```text
222.147 tok/s
4.476 ms/token
```

The output was also broken.

This became another useful lesson:

> **Never optimize blindly on top of a moving baseline.**

The known-good runtime was kept as the control version.

Experimental changes belong in the lab.

The stable version stays untouched.

---

# 13. The 441 tok/s Trap

One benchmark produced:

```text
441.785 tok/s
2.20996 ms/token
```

It looked spectacular.

It was also wrong.

Investigation revealed that the decode path had been accidentally modified so that part of the transformer computation was missing while another section had been duplicated.

The giveaway was:

```text
hidden_dim
```

being unused.

The intended FFN operations were missing:

```text
W1
W3
SwiGLU
W2
```

while other operations were duplicated.

The result:

```text
Fast ≠ Correct
```

The benchmark was discarded.

This became one of the strongest rules in the project:

> **A benchmark only counts if the full model computation is still present and numerically valid.**

---

# 14. Sampling and Coherence

GPU math correctness and model output quality turned out to be separate problems.

The runtime's GPU kernels can be correct while TinyLlama still produces strange text.

Sampling currently uses:

```text
Temperature:       0.7
Repetition penalty: 1.18
Top-P:             0.9 configured
```

An explicit Top-K=40 filter was later tested as a way to shorten the serial roulette-selection scan (see section 26b), but it regressed decode speed. The sampler instead runs the full-vocabulary softmax with a prefix-sum-localized roulette selection (section 26c).

CPU sampling was retained because GPU sampling experiments produced poor output quality and did not justify the additional complexity.

The runtime itself can therefore be fast and numerically correct while TinyLlama produces things like recursive "facts about cats" or strange continuation-style text.

That is primarily a model / prompting / sampling issue, not evidence that CUDA inference is broken.

---

# 15. Prompt Template Investigation

TinyLlama sometimes behaves more like a text continuation engine than an instruction-following assistant.

For example, short prompts can produce long recursive continuations.

This led to another important debugging rule:

> **Do not fix a model-behavior problem by randomly changing GPU kernels.**

The better diagnostic is to compare the exact prompt formatting and token sequence against a known-good reference implementation.

The runtime should first establish:

```text
same model
same tokenizer
same prompt template
same tokens
same sampling configuration
```

before blaming inference math.

---

# 16. llama.cpp Comparison

llama.cpp was used as a performance reference.

Example results on the same hardware:

```text
~351 tok/s
~368 tok/s
~351 tok/s
```

depending on prompt and generated length.

The comparison is useful because it establishes a practical target.

But the architectures are different.

llama.cpp can use vendor-optimized CUDA libraries and Tensor Core paths.

SovereignKernel deliberately explores the problem with custom kernels and a custom runtime.

Therefore the interesting question is not:

> "Can a custom runtime magically beat every vendor library?"

It is:

> **"How much performance can a specialized runtime extract when we control the entire execution path?"**

---

# 17. The cuBLAS Experiment

After reaching V6.3 stability, a separate experimental branch was created to answer a specific question:

> **How much does cuBLAS actually help this workload?**

The experiment was intentionally isolated from the known-good V6.3 runtime.

This is important.

V6.3 remains the control group.

V5 is the laboratory.

---

# 18. V5 — Q4 Dequant + cuBLAS

The first cuBLAS implementation used:

```text
Q4_0 weights
    ↓
Dequantization
    ↓
FP16 scratch buffer
    ↓
cuBLAS GEMM
```

This looked reasonable on paper.

The benchmark destroyed that assumption.

Result:

```text
Decode Speed:       23.6995 tok/s
Avg Decode Pass:     42.1666 ms/token
```

Compared with V6.3:

```text
V5:    ~23.7 tok/s
V6.3: ~289 tok/s
```

The result was not evidence that cuBLAS itself is slow.

It was evidence that:

> **Per-token dequantization + materialization + cuBLAS is a terrible architecture for this particular N=1 Q4 decode path.**

There were approximately:

```text
22 layers × 7 dequant/GEMM operations
≈ 154 operations
```

per generated token.

The huge overhead became obvious.

---

# 19. V5 — Persistent FP16 Experiment

The next experiment removed per-token dequantization.

Instead of:

```text
Q4
 ↓
dequant every token
 ↓
GEMM
```

the model was pre-baked once:

```text
Q4 model
   ↓
FP16 conversion at startup
   ↓
FP16 model permanently resident in VRAM
```

Approximately:

```text
~2.2 GB
```

of FP16 model weights were kept in VRAM.

This produced a dramatic improvement.

Example:

```text
Decode Speed:       164.037 tok/s
Avg Decode Pass:      6.07313 ms/token
```

Another run:

```text
~158 tok/s
~6.30 ms/token
```

The important result was not that cuBLAS "lost."

The important result was:

```text
23.7 tok/s
    ↓
164 tok/s
```

Removing the per-token dequantization/materialization path produced a massive improvement.

That localized a major bottleneck.

---

# 20. What the cuBLAS Experiment Actually Taught

Three architectures gave three very different results:

| Runtime | Weight path | Decode |
|---|---|---:|
| V5 | Q4 → dequant → cuBLAS | ~23.7 tok/s |
| V5 | persistent FP16 → cuBLAS | ~164 tok/s |
| V6.3 | custom Q4 decode | ~289 tok/s |

This is much more interesting than simply saying:

> "cuBLAS is slow."

The actual lesson is:

> **The execution architecture matters more than the library name.**

For N=1 decode, SovereignKernel's custom Q4 path avoids large FP16 materialization and specializes the computation around the actual compressed representation.

---

# 21. The "Physical Ceiling" Hypothesis

An attempt was made to estimate the theoretical bandwidth cost of the FP16 model.

The rough calculation was:

```text
~2.2 GB / token
```

against approximately:

```text
448 GB/s theoretical memory bandwidth
```

giving a rough upper-bound-style estimate.

However, this is not a physical tok/s ceiling.

It ignores things such as:

- cache behavior
- actual kernel efficiency
- reuse
- instruction overhead
- launch overhead
- arithmetic
- memory access pattern
- GEMM implementation
- synchronization

Therefore:

> **164 tok/s is an observed V5 result, not a proven hardware limit for cuBLAS.**

The experiment tells us where this implementation spends time.

It does not define the maximum possible performance of the GPU.

---

# 22. Row-Major / Column-Major Investigation

Another hypothesis suggested that cuBLAS was fundamentally reading the FP16 weights using the wrong memory coordinates because of row-major versus column-major layout.

That claim was not accepted without verification.

cuBLAS can represent row-major logical matrices through appropriate transposition and leading-dimension configuration.

Therefore:

> **Wrong output quality alone is not proof of a row-major/column-major bug.**

The actual matrix dimensions, transpose flags, leading dimensions and stored layout have to be checked directly.

Again:

```text
Claim
 ↓
Inspect code
 ↓
Measure
 ↓
Verify
```

Not:

```text
Sounds plausible
 ↓
Declare physics
```

---

# 23. Zero-Copy Experiments

Mapped host memory was also investigated.

For example:

```cpp
cudaHostAlloc(
    (void**)&host_logits,
    vocab_size * sizeof(float),
    cudaHostAllocMapped
);

cudaHostGetDevicePointer(
    &d_logits,
    host_logits,
    0
);
```

The goal was to experiment with GPU writes directly into mapped host memory.

The important discovery was that:

> **Zero-copy does not mean zero overhead.**

The GPU still communicates over PCIe and synchronization still matters.

These experiments therefore remained isolated rather than being blindly inserted into the production decode path.

---

# 24. Current Engineering Strategy

The runtime is now treated as two separate problems.

## Decode

```text
N = 1
```

Focus:

- latency
- kernel launches
- memory movement
- Q4 GEMV
- cache behavior
- CUDA Graphs
- synchronization
- register pressure

Target:

```text
≤ 3.33 ms/token
≈ 300 tok/s
```

---

## Prefill

```text
N > 1
```

Focus:

- batched GEMM
- WMMA
- Tensor Cores
- fused QKV
- Flash Attention
- memory throughput

Decode and prefill are therefore not forced through the same optimization strategy.

---

# 25. Current Baseline

The known-good V6.3 baseline at this point in the investigation was approximately:

```text
~285–290 tok/s sustained
~3.4 ms/token
300+ tok/s hot windows
```

(This was later superseded by V6.4 and V6.5 — see sections 26f–26j.)

The next meaningful performance gap is relatively small:

```text
~289 tok/s
      ↓
~300 tok/s
```

Approximately:

```text
3.46 ms/token
      ↓
3.33 ms/token
```

The remaining gap is therefore roughly:

```text
~0.13 ms/token
```

At this point, guessing becomes increasingly useless.

The next step is profiling.

---

# 26. Profiling Philosophy

The optimization process is deliberately conservative.

The known-good runtime is preserved.

Experiments are isolated.

Every change should answer a concrete question.

For example:

```text
Question:
Where is the missing ~0.13 ms?

Measure:
Profile the decode path.

Hypothesis:
One specific kernel / synchronization / memory path dominates.

Change:
Modify only that component.

Benchmark:
Compare against V6.3.

Decision:
Keep or revert.
```

This prevents another "Time Machine" regression.

---

# 26b. The Top-K Sampling Detour

A hypothesis suggested that the serial roulette-selection loop over the full 32,000-token vocabulary was a hidden CPU-side bottleneck, especially once CPU work started running concurrently with GPU decode via the async pipeline (see the zero-copy / async work above).

The fix seemed obvious: limit the roulette to the top 40 candidates using `std::nth_element`.

Result:

```text
~270 tok/s → ~251–253 tok/s
```

A regression, not an improvement.

The `nth_element` partial-sort and the per-token reset of the candidate-index array added more serial CPU cost than the shortened roulette scan saved.

The change was reverted.

---

# 26c. Prefix-Sum Localized Roulette

A second, better-targeted fix reused data the softmax pass already computed: each of the 8 OpenMP threads already produces a partial probability sum for its chunk of the vocabulary during the parallel softmax.

Instead of adding a new full-vocabulary pass (as Top-K did), the roulette draw is localized in two cheap steps:

1. Scan the 8 per-thread partial sums (trivial cost) to find which thread's chunk contains the random draw.
2. Serially scan only within that one chunk (~1/8th of the vocabulary) instead of the full 32,000 entries.

Result: a small, consistent, non-regressive improvement — roughly +1–2 tok/s. Small, but in the right direction, unlike Top-K.

Conclusion: the roulette loop was not actually the major bottleneck it appeared to be, likely because it already exits early once accumulated probability mass crosses the threshold — most probability mass concentrates in a handful of high-probability tokens.

---

# 26d. FP16 Output Logits

The final vocabulary projection (`output.weight`, 32,000 × 2,048) was stored and read as FP32 — roughly 262MB read from VRAM on every single decode step, regardless of everything else happening that step.

The weights were converted host-side to FP16 at load time (131MB instead of 262MB) and a matching FP16 GEMV kernel was written to read them, keeping activations in FP32.

Result: a clear, repeatable jump —

```text
~270 tok/s baseline
      ↓
~288–289 tok/s average, confirmed across independent runs
individual windows reaching 300+ tok/s for the first time
```

This was the largest single gain since the FP16 KV-cache change, and pushed the runtime to within a few tok/s of the 300 tok/s target.

---

# 26e. Periodic Slowdown Investigation

Telemetry windows showed a recurring dip in throughput — initially appearing consistently around tokens 101–120 across repeated short (200-token) runs.

The first hypothesis was that this was tied to a fixed token count (e.g. the repetition-penalty window's internal bookkeeping).

A longer run (400 tokens, larger `penalty_window`) tested this directly: if the dip were position-based, it should stay locked to tokens 101–120 regardless of run length. Instead, the dip moved — recurring instead at roughly every 60–120 tokens, at different absolute positions than before.

This rules out a fixed-position code cause and points toward a periodic, external cause — most likely GPU boost-clock/power-state cycling, or a periodic OS/driver polling interval, rather than anything in the decode kernel sequence itself.

Not yet resolved. The next diagnostic step is correlating GPU clock frequency (sampled in real time) against the timing of the dips.

---

# 26f. V6.3 — Control Baseline With CUDA Event Telemetry

To resolve the periodic-dip question properly, the runtime was instrumented with CUDA event timing alongside CPU sampler timing, so GPU execution and host-side work could be separated instead of guessed at.

Representative V6.3 result under this instrumentation:

```text
Generated: 400 tokens
Decode Speed: 309.064 tok/s
Avg Decode Pass: 3.211 ms
Prefill Latency: 58.184 ms
```

Representative per-token breakdown:

```text
Pure GPU:     ~2.84–3.36 ms
CPU Sampler:  ~0.08–0.23 ms
Total/token:  ~3.0–3.5 ms
```

This directly tested two hypotheses raised earlier in the periodic-slowdown investigation:

- **stdout / `std::flush` overhead** — removing it did not materially change throughput. Ruled out.
- **CPU sampler cost** — consistently a small fraction (~0.07–0.23ms) of total per-token time versus ~2.7–3.3ms of GPU time. Ruled out as the dominant factor.

> **The decode bottleneck is primarily GPU-side, not host-side.**

This reframes the periodic-dip investigation: the earlier console-flush and CPU-sampler hypotheses were reasonable given the data available at the time, but direct GPU-vs-host timing separation shows the real cost sits inside GPU execution.

---

# 26g. V6.4 — Q4 Output Projection

The output projection weights (`output.weight`) — already moved from FP32 to FP16 in section 26d — were quantized further, directly in VRAM, from FP16 down to Q4_0.

```text
output.weight
FP16 (131MB) → Q4_0 (~36.8MB)
```

Representative V6.4 result:

```text
Generated: 400 tokens
Decode Speed: 318.661 tok/s
Avg Decode Pass: 3.113 ms
Prefill Latency: 204.686 ms
```

Best observed window: **~336.7 tok/s**.

```text
V6.3 → V6.4
309.064 → 318.661 tok/s
≈ +3.1%
```

However, the sequence-length-dependent slowdown (early windows faster, later windows gradually slower) remained present after this change.

> **Output projection quantization was a real, measurable win — but not the cause of the gradual degradation curve.**

---

# 26h. V6.5 — 2-Warp Split-K

V6.5 kept the V6.4 Q4 output projection and added a different warp-level decomposition for the N=1 GEMV path: **2-warp split-K**.

Representative V6.5 result:

```text
Generated: 400 tokens
Decode Speed: 324.272 tok/s
Avg Decode Pass: 3.059 ms
Prefill Latency: 161.936 ms
```

Best observed window: **340.302 tok/s**.

```text
V6.4 → V6.5
318.661 → 324.272 tok/s
≈ +1.76%

V6.3 → V6.5 overall
309.064 → 324.272 tok/s
≈ +4.92%
```

---

# 26i. The Sequence-Length Degradation Curve

Across V6.3, V6.4 and V6.5 alike, the same qualitative pattern persists: decode speed is highest early in generation and gradually declines as the sequence grows.

Representative V6.5 window progression:

```text
~340 tok/s → ~337 → ~334 → ~331 → ~329 → ~327 tok/s
```

Corresponding per-token GPU time moves from approximately **2.72ms toward 3.17ms**.

This is the same behavior originally flagged in the periodic-slowdown investigation (26e), now confirmed via CUDA event telemetry to be GPU-side rather than host/console-side.

The interpretation splits the decode cost into two separate problems:

1. **Fixed per-token GPU cost** — improved by Q4 output projection and split-K (V6.4, V6.5).
2. **Sequence-dependent GPU cost** — still present after both optimizations, not yet fixed.

The leading suspect for (2) is the attention kernel, `mha_kv_cache_l1_tiled_kernel`: unlike the fixed-dimension GEMVs (QKV, WO, W1/W3, W2), attention must read an increasingly large KV history as the sequence grows.

---

# 26j. Next Phase — Nsight Compute Profiling

The next step is deliberately **not** another blind kernel tweak. V6.5 is preserved as the control build while the following is profiled directly with Nsight Compute:

```text
gemv_q4_0_qkv_fused_warp_opt
mha_kv_cache_l1_tiled_kernel
```

Key metrics of interest: SM utilization, memory throughput, L1/L2 hit rates, DRAM throughput, occupancy, warp stall reasons, and instruction throughput — to determine, concretely, why the GPU spends ~3ms/token, rather than guessing from kernel names.

For the attention kernel specifically, the plan is to compare the same kernel at early, medium and late decode positions and measure whether kernel duration, memory traffic, and stall reasons scale with sequence length — which would make the sequence-dependent bottleneck directly measurable rather than speculative.

**Performance history so far:**

| Version | Main change | Avg decode | Avg ms/token | Best observed |
|---|---|---:|---:|---:|
| V6.3 | Telemetry / control | 309.064 tok/s | 3.211 ms | ~322 tok/s |
| V6.4 | Q4 output.weight | 318.661 tok/s | 3.113 ms | ~337 tok/s |
| V6.5 | 2-warp split-K + Q4 output | **324.272 tok/s** | **3.059 ms** | **340.302 tok/s** |

```text
V6.3 → V6.5 total improvement: ≈ +4.92%
```

**Current question carried forward:**

> **What GPU-side mechanism causes the ~340 → ~327 tok/s degradation as the KV sequence grows?**

---

# 27. What Failed

The project contains a useful collection of rejected ideas:

- asynchronous CUDA streams
- naive float4 vectorization
- N-gram speculative decoding
- WMMA N=1 decode
- GPU sampling experiments
- Q4 → FP16 → cuBLAS per-token conversion
- mapped-memory assumptions
- premature architectural rewrites
- Top-K=40 sampling filter (added serial cost instead of removing it)

These failures are part of the engineering record.

They show what was actually tested rather than hiding unsuccessful experiments.

---

# 28. What Worked

The strongest improvements came from specialization and measurement:

- custom Q4 GEMV
- fused QKV operations
- custom GQA attention
- online softmax / Flash Attention style execution
- FP16 KV cache
- FP16 output logits
- Q4_0 output logits (quantized further past FP16)
- 2-warp split-K GEMV decomposition
- prefix-sum localized sampling
- CUDA Graphs
- WMMA for appropriate prefill workloads
- PTX-level optimization
- reducing unnecessary intermediate buffers
- CPU sampling where it was actually faster
- profiling before making large architectural changes

The general pattern was consistent:

> **Measure the workload → identify the real bottleneck → specialize the hot path.**

---

# 29. What SovereignKernel Became

What started as:

> "I want to understand LLM inference."

became a complete experiment in:

- GPU execution
- memory movement
- quantized inference
- CUDA kernel design
- Tensor Core programming
- attention implementation
- runtime scheduling
- numerical correctness
- profiling
- performance engineering

The interesting part was never simply achieving a large tok/s number.

The interesting part was finding out **why** the number changed.

---

# 30. Current Principle

SovereignKernel follows one rule above everything else:

```text
Do not optimize the story.

Optimize the measured bottleneck.
```

Or, more simply:

> **Measure first.**
