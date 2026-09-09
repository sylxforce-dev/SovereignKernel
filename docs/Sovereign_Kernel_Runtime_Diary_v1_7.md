# Sovereign Kernel: Runtime Diary & Architecture Audit v1.7
 
## 0. Project Foundation
 
Sovereign Kernel was built with the goal of implementing a local LLM inference runtime from the lowest practical software layer: GGUF loading, tensor placement, KV-cache management, CUDA execution, quantized matrix operations, and runtime orchestration without relying on the internals of an existing inference backend.
 
The central architectural principle is that **N=1 decode and large-token prefill are fundamentally different workloads**:
 
- N=1 decode prioritizes latency, launch overhead, memory behavior, and sequential execution.
- Large-token prefill exposes enough parallelism to make fused GEMM/WMMA execution substantially more effective.
The runtime therefore maintains separate execution strategies for Decode and Prefill rather than forcing both workloads through the same computational path.

Source availability: The runtime source code is now publicly available in the repository. This diary documents the architecture, measurements, experiments, validation work, and development history alongside the implementation.
 
---
 
## 1. Historical Foundation
 
The CUDA development phase initially entered the ~151 tok/s range. This was not the beginning of the project itself: a fully functional CPU inference path had already been completed before CUDA development became the primary focus.
 
The historical progression was approximately:
 
- Initial CUDA performance: ~151 tok/s
- Pure CUDA Q4_0 engine: **203.206 tok/s**
- Tensor/Hybrid path: **226.969 tok/s**
- Later Hybrid runs: **240+ tok/s**
- Current steady-state Decode range: approximately **240–260 tok/s**, with individual windows reaching higher values after warm-up
Major architectural work included:
 
- A custom GGUF loader/parser
- Direct VRAM weight residency
- Q4_0 CUDA execution
- FP16 KV-cache
- Custom attention and tensor kernels
- Removal of unnecessary synchronization from hot paths
- CUDA Graph experimentation
- WMMA/Tensor Core experimentation for high-parallelism workloads
The project was not intended merely to reproduce an existing inference framework. The purpose was to expose and control the runtime execution path directly.
 
---
 
## 2. V1 Gold Standard — N=1 Decode
 
The current V1 Decode baseline is designed around single-token autoregressive generation.
 
Measured steady-state performance is approximately:
 
**240–260 tok/s / ~3.7–4.0 ms/token**
 
Repeated telemetry demonstrates that this is not the result of a single isolated throughput spike. After the initial generation phase, token windows repeatedly enter the 240–260 tok/s region, with some windows reaching the mid-260s.
 
Representative telemetry from a later run:
 
```text
1–20       239.35 tok/s | 4.178 ms/token
21–40      256.76 tok/s | 3.895 ms/token
41–60      264.20 tok/s | 3.785 ms/token
61–80      266.64 tok/s | 3.750 ms/token
81–100     256.30 tok/s | 3.902 ms/token
101–120    265.76 tok/s | 3.763 ms/token
121–140    267.33 tok/s | 3.741 ms/token
141–160    245.56 tok/s | 4.073 ms/token
161–180    264.07 tok/s | 3.787 ms/token
181–200    261.11 tok/s | 3.830 ms/token
```
 
The important result is therefore not a single peak number but the existence of a repeatable high-throughput steady-state.
 
---
 
## 3. CUDA and Tensor Core Architecture
 
The GPU execution path was implemented from scratch and subsequently expanded to investigate WMMA/Tensor Core execution.
 
### Decode
 
N=1 decode provides relatively little parallel work and has strong sequential dependencies. Increasing the amount of matrix parallelism solely to activate Tensor Cores can introduce more orchestration overhead than computational benefit.
 
For this reason, the Decode path prioritizes:
 
- low launch overhead
- memory locality
- minimal synchronization
- direct GPU residency
- efficient N=1 kernels
- stable per-token latency
### Prefill
 
Prefill has a fundamentally different workload profile. Hundreds or thousands of prompt tokens provide enough parallelism for fused matrix operations to become highly effective.
 
The runtime therefore uses a fused WMMA-oriented path for high-volume prefill work, including fused QKV computation.
 
A measured prefill result reached:
 
**75.7 ms**
 
This is a measured result for a specific test prompt, not a universal prefill latency for arbitrary prompt lengths. It nevertheless demonstrates that the fused prefill path is operational and capable of sub-100 ms latency under the tested workload.
 
---
 
## 4. CUDA Graphs and Runtime Overhead
 
CUDA Graphs were introduced to reduce repeated kernel-launch and CPU-side submission overhead during Decode.
 
This matters because N=1 inference is not limited purely by GPU arithmetic throughput. When individual operations are small, the following components can become significant portions of total token latency:
 
1. GPU kernel execution
2. CUDA launch overhead
3. CPU-side runtime/scheduling overhead
4. Synchronization and dependency handling
CUDA Graphs do not remove the CPU from the execution path. Their purpose is to reduce the amount of repeated CPU-side dispatch work and associated launch overhead.
 
Future benchmarks must therefore distinguish GPU computation time from runtime submission and synchronization costs rather than treating all latency as a single "GPU speed" number.
 
---
 
## 5. Input and Buffer Engineering
 
A separate engineering problem originated from the Windows console rather than the inference engine itself.
 
The original multiline `std::getline` and empty-line handling interacted poorly with Windows console copy/paste behavior. Long pasted inputs could cause the input loop to wait unexpectedly or interpret the buffer incorrectly.
 
The runtime was simplified to a deterministic single-line input path:
 
```cpp
if (!std::getline(std::cin, user_input))
    break;
```
 
Trailing newline/whitespace cleanup was also simplified.
 
This is not a major GPU optimization, but it is important for reproducible testing: the test harness must not introduce input-parser failures that contaminate runtime experiments.
 
---
 
## 6. TinyLlama Audit — Runtime Speed Does Not Equal Model Intelligence
 
TinyLlama 1.1B Chat in GGUF Q4_0 was used as the primary practical test model.
 
Several tests demonstrated that model-quality failures were independent of CUDA runtime performance.
 
For example:
 
```text
CAN YOU CALCULATE 2X2
```
 
produced fabricated numerical reasoning rather than reliably returning the expected result.
 
Another test requested:
 
```text
Define RAM in exactly one short sentence.
```
 
The model began with a partially relevant definition but then continued generating unrelated material, including conversational fragments and school-related text.
 
The runtime therefore demonstrated an important separation:
 
> **A fast and deterministic inference engine does not automatically make a small language model capable of reasoning reliably.**
 
The system prompt was tightened into a more constrained processing-oriented instruction such as:
 
```text
You are a precise data processing unit...
```
 
This reduced some stylistic behavior but did not eliminate the underlying model limitations.
 
Large `max_tokens` limits also demonstrated that the model can continue generating after an otherwise acceptable answer, effectively beginning a new conversational trajectory on its own.
 
This led to an architectural conclusion: Barkus must provide an external control layer for retrieval, validation, stopping, memory management, and agent state.
 
---
 
## 7. N-Gram / Speculative Decode Experiment
 
An experimental N-Gram history scanner was implemented to search previous token history for repeated patterns and construct small multi-token drafts.
 
The motivation was to increase the amount of parallel work presented to the GPU during Decode and potentially make WMMA execution more productive.
 
The experiment produced an important negative result:
 
**CPU-side orchestration and draft-generation overhead can consume the performance gained from additional GPU parallelism.**
 
The experimental path was therefore reverted.
 
The conclusion is not that speculative decoding is inherently ineffective. The more precise conclusion is:
 
> **A small speculative batch is not automatically beneficial when the orchestration required to produce it costs more than the GPU parallelism saves.**
 
This is now recorded as a Phase 3 experimental result rather than being allowed to contaminate the V1 baseline.
 
---
 
## 8. Llama.cpp Performance Comparison
 
The same general hardware configuration was used for comparison:
 
- GPU: RTX 5060 Ti 8 GB
- Model: TinyLlama 1.1B Chat
- Quantization: GGUF Q4_0
- Llama.cpp: all layers on GPU (`n_gpu_layers=-1`)
- KV cache: VRAM (`offload_kqv=True`)
- CPU threads: `n_threads=8`
- Streaming generation
Measured llama.cpp generation results included:
 
```text
276.38 tok/s
339.43 tok/s
316.05 tok/s
332.75 tok/s
330.83 tok/s
324.51 tok/s
282.12 tok/s
340.35 tok/s
316.22 tok/s
```
 
The observed range was therefore approximately:
 
**276–340 tok/s**
 
or roughly:
 
**2.9–3.5 ms/token**
 
Llama.cpp remains faster in these measurements. However, the gap is no longer a binary "working versus non-working" comparison. Sovereign Kernel has entered the same general performance scale and the remaining gap can now be investigated through specific runtime and GPU bottlenecks.
 
---
 
## 9. CPU Runtime Utilization
 
On the same system, observed CPU utilization during generation was approximately **10–12%** for both Sovereign Kernel and llama.cpp — no meaningful difference between the two engines on this metric.
 
---
 
## 10. Sovereign Kernel Performance Progression
 
The historical ~157–159 tok/s record was surpassed.
 
A major Tensor/Hybrid CUDA run reached:
 
- **226.969 tok/s average decode speed**
- **4.29087 ms average decode pass**
- First 20-token window: **251.018 tok/s / 3.984 ms/token**
- ~300 generated tokens
- CPU usage remained approximately **12%**
This represented roughly a **43% improvement** over the previous ~159 tok/s record.
 
A separate Pure CUDA Q4_0 execution path also exceeded 200 tok/s:
 
- **203.206 tok/s**
- **4.79373 ms average forward pass**
- 300 generated tokens
- 1.47633 s total generation time
This distinction is important.
 
The >200 tok/s result was not exclusively produced by Tensor Core/WMMA execution. Both the standalone Pure CUDA path and the Tensor/Hybrid path improved.
 
Therefore, future performance analysis must keep:
 
```text
Pure CUDA
```
 
and
 
```text
Tensor / Hybrid CUDA
```
 
as separate benchmark categories.
 
---
 
## 11. Warm-Up Behavior
 
A later Hybrid CUDA run provided evidence that generation does not immediately enter steady-state throughput.
 
Measured results included:
 
```text
whats up misha:
220.055 tok/s | 4.430 ms/token
 
misha tell me something good:
240.627 tok/s | 4.034 ms/token
```
 
The latter run generated 297 tokens with:
 
```text
Prefill: 124.331 ms
```
 
The live telemetry was particularly informative.
 
Early windows:
 
```text
1–20       199.753 tok/s | 5.006 ms/token
21–40      197.088 tok/s | 5.074 ms/token
```
 
Later execution reached:
 
```text
~260 tok/s | ~3.84 ms/token
```
 
and subsequently remained mostly within approximately the 225–265 tok/s region.
 
This establishes that the runtime can begin generation below its later steady-state throughput and then accelerate.
 
The current hypothesis space includes:
 
- CUDA/runtime warm-up
- scheduler behavior
- synchronization
- execution-path initialization
- cache state
- other cold-start effects
No single cause has yet been proven.
 
Therefore:
 
> **The first 20 generated tokens must not automatically be treated as representative of steady-state GPU throughput.**
 
---
 
## 12. Sequence-Length Degradation
 
A separate effect occurs later in generation.
 
Telemetry from longer sequences showed decreasing throughput:
 
```text
1–20       ~251 tok/s
61–80      ~242 tok/s
121–140    ~226 tok/s
181–200    ~222 tok/s
221–240    ~209 tok/s
281–300    ~217 tok/s
```
 
This suggests that Decode latency is not fully invariant with respect to sequence length.
 
Potential causes include:
 
- KV-cache growth
- Attention scaling
- memory-access behavior
- cache locality
- increased memory traffic
However, there is not yet sufficient evidence to assign the degradation to one specific component.
 
The scientifically accurate conclusion is:
 
> **Decode latency currently changes as the sequence grows, and KV-cache/Attention scaling is now a primary investigation target.**
 
This should be treated separately from the initial cold-start/warm-up effect.
 
---
 
## 13. Profiling Lessons
 
A micro-profiler produced the following example measurements for one layer:
 
```text
Embed/Init: 1112.48 us
RMSNorm:      48.19 us
QKV GEMV:     88.00 us
RoPE/Cache:  130.02 us
Attention:    50.43 us
WO GEMV:      35.23 us
FFN:         370.75 us
```
 
However, enabling the profiler altered the execution path substantially enough to reduce observed decode performance from approximately:
 
```text
226 tok/s → 97.7 tok/s
```
 
These numbers therefore cannot yet be treated as uncontaminated production bottleneck measurements.
 
The primary lesson is:
 
> **The measurement system must not significantly alter the system being measured.**
 
Future profiling must be increasingly non-invasive, with production execution kept as close as possible to the benchmarked path.
 
*(Note, v1.7: this profiler-induced 226 → 97.7 tok/s drop is unrelated to, and should not be confused with, the separate ~90 tok/s decode-WMMA result recorded in §20 below — the two share a similar magnitude by coincidence, but one is measurement contamination and the other is a genuine architectural finding about launch overhead.)*
 
---
 
## 14. Current V1.5 Baseline
 
The following components are sufficiently stable to be treated as the current baseline:
 
| Component | Status |
|---|---|
| GGUF loading | 🔒 Baseline |
| VRAM weight residency | 🔒 Baseline |
| Q4_0 CUDA execution | 🔒 Baseline |
| N=1 Decode | 🔒 Baseline |
| KV-cache | 🔒 Working baseline; optimization remains open |
| WMMA fused prefill | 🔒 Working baseline |
| CUDA Graphs | 🔒 Runtime component |
| Single-line input | 🔒 Locked |
| Telemetry | 🔒 Benchmark baseline |
| N-Gram speculative path | ❌ Reverted |
| TinyLlama reasoning quality | ⚠️ Model limitation |
| Cold-start behavior | 🧪 Under investigation |
| Sequence-length scaling | 🧪 Under investigation |
| KV/Attention scaling | 🧪 Under investigation |
| llama.cpp parity | 🧪 Not yet achieved |
 
---
 
# 15. Phase 2 — Runtime Baseline 🔒 SEALED
 
**Phase 2 is now locked.**
 
The purpose of the lock is to stop endless V1 micro-optimization from destabilizing the runtime that has already been proven functional.
 
Phase 2 does **not** mean that every kernel is theoretically optimal.
 
It means:
 
> **The runtime is sufficiently stable, measurable, and performant to serve as the foundation for the next system layer.**
 
The Phase 2 baseline includes:
 
- Fully GPU-resident GGUF Q4_0 execution
- Working Pure CUDA and Tensor/Hybrid CUDA paths
- Stable high-throughput N=1 Decode
- Fused WMMA Prefill
- CUDA Graph runtime integration
- Working KV-cache
- Deterministic benchmark telemetry
- Reproducible test harness
Phase 2 code should no longer be modified casually during Phase 3 experimentation.
 
---
 
## 16. Phase 2 Closing Validation Session (v1.7 addendum)
 
*The following work was carried out to close out remaining open questions on the Phase 2 baseline before treating it as fully sealed for validation purposes, not just performance purposes. It targets two gaps §14–15 above left open: the WMMA/flash-attention kernels used in the fused prefill path had never been checked against an independent reference, and an unexplained repetition/coherence bug had been observed in one hybrid-engine test run.*
 
### 16.1 Step2 Re-validation — Race Condition Found and Fixed
 
The existing single-vector WMMA GEMV test (tiled, for arbitrary M/K) was re-run and initially **failed**: max error 7.93 at M=128, K=128 — far beyond FP16 rounding, indicating a logic bug.
 
**Root cause:** the GEMV kernel was launched with 4 warps per thread block, but its `a_tile`/`b_tile`/`c_tile` shared-memory arrays are block-scoped, not per-warp — all 4 warps were racing on the same shared memory while computing different output tiles.
 
**Fix:** one warp per thread block (`threads_per_block = 32`).
 
**Result after fix:** max error 0.00142 (M=128, K=128) — well within the FP16-precision threshold. **PASS.**
 
### 16.2 Step3 — Batched GEMM Isolated Test
 
The batched WMMA GEMM kernel (used exclusively by the fused prefill path) had never been tested against an independent CPU dequant+GEMM reference. A new isolated test covered:
 
| M | K | N | Result |
|---|---|---|---|
| 128 | 128 | 16 | PASS (max error 0.00241) |
| 128 | 128 | 5 | PASS (max error 0.00161) |
| 256 | 256 | 11 | PASS (max error 0.00276) |
| 2048 | 2048 | 7 | PASS (max error 0.00875) |
 
The non-16-multiple N values (5, 11, 7) specifically targeted the tiling padding logic for realistic prompt lengths (real prompts are essentially never exactly 16/32/48 tokens). The 2048×2048 case matches the actual model dimension. **All four configurations passed.**
 
### 16.3 Step4 — Flash Attention Isolated Test
 
The batched causal flash-attention kernel had likewise never been isolated. A new test compared it against a naive O(N²) CPU causal-attention reference:
 
| N | heads | kv_heads | head_dim | Result |
|---|---|---|---|---|
| 16 | 4 | 1 | 64 | PASS (1.8e-7) |
| 5 | 4 | 1 | 64 | PASS (1.2e-7) |
| 100 | 4 | 1 | 64 | PASS (1.8e-7) |
| 37 | 32 | 4 | 64 | PASS (1.8e-7) |
 
The last configuration is the exact TinyLlama GQA setup (32 query heads, 4 KV heads, head_dim=64) at a non-round N. Errors are at pure FP32 rounding level. **All four configurations passed.**
 
With §16.1–16.3 complete, every WMMA kernel used in the production prefill path is now individually validated against an independent reference. **The WMMA math was never the source of the coherence bug reported earlier.**
 
### 16.4 The Actual Bug — Missing Stop-Sequence Detection
 
With the WMMA math cleared, the decode loop's stop condition was inspected directly. It checked only for the `</s>` token ID, gated behind `min_tokens`, with no detection of a hallucinated `<|user|>` turn. Combined with a `min_tokens` value forcing generation well past any natural stopping point, the model was free to talk past the end of its turn and start fabricating a new user message — which read as "repetition" but was actually a hallucinated conversation continuation.
 
**Fix:** a rolling 64-character lookback buffer scans decoded output for the literal sequence `<|user|>` after every token; on match, the partial match is trimmed from the printed output and generation halts immediately, regardless of `min_tokens`.
 
**Verification:** the same repetition-triggering prompt, run again for 200 tokens, produced coherent, non-repeating output with no fake user-turn. Speed was unaffected. **Bug resolved — confirmed unrelated to WMMA.**
 
### 16.5 Negative Result — Pushing WMMA Into Decode
 
Given the WMMA math was now trusted, a deliberate experiment routed three of the four per-layer decode GEMVs (`wo`, the `w1`/`w3` SwiGLU pair, and `w2`) through the validated single-vector WMMA GEMV kernel, each preceded by an FP32→FP16 conversion pass.
 
**Result: decode speed dropped to ~90 tok/s**, down from the 240–265 tok/s scalar baseline.
 
This is a direct empirical confirmation of the architectural principle already recorded in §3: at N=1, the added per-kernel launch overhead (roughly 4 extra kernel launches per layer × 22 layers ≈ 88 extra launches per token) exceeds any compute benefit Tensor Cores provide on a single-vector operand. The batched prefill path remains the only place in this engine where WMMA earns its keep.
 
Per the project's own experiment discipline (§7): **this result is recorded and the decode-WMMA variant is not adopted.** Scalar N=1 decode remains the production path.
 
---
 
# 17. Phase 3 — Experimental Lab 🧪 OPEN
 
Phase 3 is deliberately open-ended.
 
Primary targets include:
 
1. Barkus agentic orchestration
2. Local ChromaDB retrieval
3. Retrieval → context assembly → prefill → reasoning → action
4. Agent memory layers
5. KV-cache optimization
6. Attention scaling
7. Sequence-length behavior
8. CUDA Graph experiments
9. Batching experiments
10. Speculative execution experiments
11. Agent-workload-specific benchmarks
The Phase 3 rule is:
 
> **An experiment may outperform Phase 2, but it must never silently destroy the Phase 2 baseline.**
 
The workflow is:
 
```text
Experiment
    ↓
Benchmark
    ↓
Compare against Phase 2
    ↓
Better?
 ┌──┴──┐
YES    NO
 ↓      ↓
Keep   Revert
 ↓      ↓
Record result in diary
```
 
Experimental improvements should be isolated from the locked baseline until their performance and stability are demonstrated.
 
No specific next architecture is committed to. What comes next in Phase 3 is deliberately undecided.
 
---
 
# 18. Why Barkus Is the Next Logical Layer
 
Single-turn chat does not fully exploit the architectural separation between retrieval, prefill, and low-latency Decode.
 
An autonomous agent does.
 
A typical Barkus execution cycle can be represented as:
 
```text
User request
      ↓
Local retrieval
      ↓
Context assembly
      ↓
GPU prefill
      ↓
N=1 reasoning / action generation
      ↓
Tool execution or memory update
      ↓
Local retrieval
      ↓
Context refresh
      ↓
...
```
 
This is where Sovereign Kernel's low-latency execution becomes a system-level property rather than merely a benchmark result.
 
If retrieval remains local and context processing is fast, an agent can perform multiple reasoning/action cycles without introducing remote inference queues or cloud dispatch latency.
 
The measured sub-100 ms prefill result is particularly relevant here because an agent may refresh its context multiple times during a single user interaction.
 
The next question is therefore no longer simply:
 
> "How many tokens per second can the engine generate?"
 
It becomes:
 
> **"How quickly can the complete retrieval → context → prefill → reasoning → action loop execute?"**
 
That is the benchmark that matters for Barkus.
 
---
 
# 19. Immediate Benchmarking Priority
 
Before modifying additional kernels, the next benchmark should measure generation in token windows from the beginning to the end of the sequence.
 
For example:
 
```text
Tokens 1–20
Tokens 21–40
Tokens 41–60
Tokens 61–80
...
```
 
For every window:
 
```text
tok/s
ms/token
```
 
The same methodology should be applied to llama.cpp.
 
The objective is to distinguish:
 
### A. Cold-start behavior
 
```text
slow initial tokens
        ↓
warm-up
        ↓
steady-state
```
 
from:
 
### B. Sequence-length degradation
 
```text
steady-state
      ↓
KV-cache growth
      ↓
Attention/memory scaling
      ↓
higher ms/token
```
 
If both engines exhibit similar initial ramp-up, the effect may be generic runtime/GPU behavior.
 
If only Sovereign Kernel shows a significant ramp-up, CPU submission, synchronization, scheduler, initialization, or execution-path behavior becomes a stronger candidate.
 
If both engines degrade with sequence length but at different rates, KV-cache/Attention/memory behavior becomes the more relevant comparison.
 
---
 
# 20. Current Scientific Conclusion
 
Sovereign Kernel has **not yet demonstrated that it is faster than llama.cpp**.
 
It has demonstrated something more specific and technically meaningful:
 
1. A custom CUDA inference runtime can be built and operated successfully from the low-level execution layer.
2. GGUF Q4_0 weights can remain fully GPU-resident.
3. Pure CUDA execution has exceeded **200 tok/s**.
4. Tensor/Hybrid CUDA Decode has reached the **226.969 tok/s** measured milestone and later entered the **240–260 tok/s steady-state class**.
5. Fused WMMA Prefill has reached a measured **75.7 ms** result under a specific test workload.
6. CPU utilization during generation is comparable between the two engines (~10–12%) and is not a distinguishing factor.
7. Small speculative batches can lose their theoretical GPU advantage when CPU-side orchestration becomes expensive.
8. TinyLlama's reasoning and stopping behavior are independent model limitations rather than evidence of CUDA runtime failure.
9. Decode currently exhibits both an initial warm-up effect and sequence-length-dependent performance changes.
10. KV-cache, Attention scaling, memory behavior, synchronization, and runtime submission remain active areas of investigation.
11. The remaining llama.cpp performance gap is now a **localized systems problem**, rather than evidence that the custom runtime is fundamentally non-viable.
12. *(v1.7)* Every WMMA kernel used in the production prefill path — batched GEMM and batched causal flash attention — has now been individually validated against an independent CPU reference across multiple shapes, including non-round token counts and the exact production model scale.
13. *(v1.7)* A hybrid-engine coherence bug initially suspected to be a WMMA correctness issue was traced to a missing stop-sequence check in the decode sampling loop, unrelated to any GPU kernel, and has been fixed and verified.
14. *(v1.7)* Deliberately routing decode-path GEMVs through WMMA was tested and rejected: decode throughput fell from ~240–265 tok/s to ~90 tok/s, confirming that Tensor Core use at N=1 costs more in launch overhead than it returns in compute.
The next major engineering target is therefore not arbitrary kernel rewriting.
 
It is the measurement and construction of the complete:
 
```text
RETRIEVAL
    ↓
CONTEXT ASSEMBLY
    ↓
PREFILL
    ↓
REASONING
    ↓
ACTION
    ↓
MEMORY UPDATE
    ↓
RETRIEVAL
```
 
loop.
 
---
 
# 21. INTERNAL SEAL
 
```text
SOVEREIGN KERNEL v1.7
 
Phase 1  FOUNDATION          [SEALED]
Phase 2  RUNTIME BASELINE    [SEALED — validation complete]
Phase 3  EXPERIMENTAL LAB    [OPEN — no committed next architecture]
 
Current Decode:
    ~240–260 tok/s steady-state class (scalar N=1, unchanged from v1.6)
 
Pure CUDA:
    203.206 tok/s measured
 
Tensor/Hybrid:
    226.969 tok/s measured
    240.627 tok/s later measured run
 
Prefill:
    75.7 ms measured record
    124.331 ms in later Hybrid run
 
Observed CPU Runtime Load:
    ~10-12% for both Sovereign Kernel and llama.cpp
 
Phase 2 Validation Coverage (new in v1.7):
    Batched WMMA GEMM        — isolated CPU-reference test, PASS (incl. full model scale, non-round N)
    Batched flash attention  — isolated CPU-reference test, PASS (incl. exact GQA config)
    Single-vector WMMA GEMV  — race condition found + fixed, re-validated PASS
    Stop-token bug           — found + fixed (missing <|user|> detection, min_tokens override)
    Decode-side WMMA         — tested, REJECTED (~90 tok/s, launch overhead exceeds benefit)
 
Current Investigation:
    Cold-start / warm-up
    KV-cache scaling
    Attention behavior
    Memory access
    Runtime synchronization
    CPU submission overhead
 
Next Battlefield:
 
    BARKUS
    CHROMADB
    AGENT LOOP
    RETRIEVAL → PREFILL → REASONING
    KV / ATTENTION SCALING
    END-TO-END AGENT LATENCY
```
 
**Phase 2 is sealed — architecture and correctness both confirmed. Phase 3 is open, undecided.**
 
Sovereign Kernel is now treated as a stable, validated runtime baseline rather than an endlessly changing prototype.
 
