# Serving posture for a kernel / code / math / science workload (2026-08-18)

Measured on `toymaker` + `spark-2949`, both nodes verified quiet, image digest `a839484…`,
checkpoint `7872f01b` (= released `9e165c30` weights), util 0.80, decode window excludes TTFT,
256 forced tokens (`ignore_eos`, min=max), temp 0.6/top_p 0.95, `thinking:false`, n=3,
spec-decode counters read from `/metrics` per trial. Corpus: `bench-gb10/domain_bench.sh`.

## 1. Prose is the worst case. Do not tune this deployment on prose.

At the shipped `MTP_NUM_TOKENS=5`:

| domain | decode tok/s | mean accept len | acceptance | ms/step | GB/step | inferred expert union |
|---|---|---|---|---|---|---|
| **triton kernel** | **75.6** | **5.29** | **0.857** | 69.9 | 10.13 | 37.5 |
| cuda kernel | 61.1 | 4.14 | 0.627 | 67.7 | 9.81 | 36.3 |
| science (plasma MHD) | 60.4 | 4.09 | 0.617 | 67.6 | 9.81 | 36.2 |
| math (finite fields) | 56.8 | 3.78 | 0.556 | 66.5 | 9.64 | 35.6 |
| algorithms (competitive) | 48.8 | 3.27 | 0.454 | 67.0 | 9.71 | 35.9 |
| prose *(control)* | 46.4 | 3.16 | 0.433 | 68.1 | 9.88 | 36.5 |

**Triton at 75.6 tok/s sits at/above the recipe's audited 73–76.** The hardware and config are
healthy. A prose benchmark understates the real workload by **up to 63 %**.

## 2. Routing breadth is NOT content-dependent, and is not a lever

`ms/step` is 66.5–69.9 ms across every domain — a ±2.5 % band — so the per-step expert union is
~36 of 256 experts regardless of whether the model is emitting Triton, a proof, or prose.
Bytes/step is effectively fixed by `num_experts_per_tok × speculative positions`, not by content.

Consequence: **every bit of the 2× throughput spread across domains is draft acceptance**
(0.433 → 0.857). Optimizing "which experts get hit" is a dead end here; optimizing acceptance,
or the number of speculative positions, is not.

(Inference method: decode is MoE-weight-bandwidth bound — see `DECODE-MODEL-2026-08-18.md` —
so `bytes/step = (accept_len / decode_rate) × bandwidth`, and `expert_union = bytes/step ÷
(6.29 MB × 43 layers)`. Absolute union counts carry the documented 0.65 GEMM-efficiency
assumption; the *relative* constancy across domains does not.)

## 3. MTP depth 5 is already optimal — k=8 measured and rejected

Same corpus, same cluster, only `MTP_NUM_TOKENS` changed 5 → 8:

| domain | k=5 tok/s | k=8 tok/s | Δ | accept_len 5 → 8 | ms/step 5 → 8 |
|---|---|---|---|---|---|
| triton | 75.6 | 76.9 | **+1.7 %** | 5.29 → 5.99 | 69.9 → 77.9 |
| cuda | 61.1 | 56.5 | **−7.5 %** | 4.14 → 4.25 | 67.7 → 75.1 |
| science | 60.4 | 54.1 | **−10.4 %** | 4.09 → 4.00 | 67.6 → 73.9 |
| math | 56.8 | 54.1 | −4.8 % | 3.78 → 4.08 | 66.5 → 75.4 |
| algo | 48.8 | 49.1 | +0.5 % | 3.27 → 3.68 | 67.0 → 74.9 |
| prose | 46.4 | 37.8 | −18.5 % | 3.16 → 2.92 | 68.1 → 77.1 |

Deeper speculation does raise accept length, but it widens the per-step expert union
(**36 → 40**, ms/step **+11 %**), and on a bandwidth-bound decode that cost outruns the gain
everywhere except Triton, where it is inside noise. This closes the open
"A/B `MTP_NUM_TOKENS=5` vs `6`" item in `docs/GLM-NEW-REPORT.md:360-362`: **keep 5**.

Note `max_cudagraph_capture_size` is truncated to 48 at k=8 (`6 × (8+1) = 54` requested).

## 4. Recommended posture

| setting | value | why |
|---|---|---|
| Both ranks exclusive to DS4 | **enforced** | worth **~1.7×** (C1 decode 20.9–25.6 → 39.9 on prose; the single largest measured effect). GB10 is unified memory, so a CPU-side hog on either node counts. `bench-gb10/run_arm.sh` refuses to measure otherwise. |
| `MTP_NUM_TOKENS` | **5** | §3 |
| `VLLM_B12X_W4A16_FORCE_*` | **leave inert** | arm A1 (forced tile `128,64,128`, `BLOCKS_MAX_M=0`) was a null result: decode 35.3 vs 39.9, but acceptance moved too (0.278 vs 0.333) and acceptance cannot depend on tile config, so the delta is run-to-run variance. |
| `thinking` | **send `thinking_token_budget`** | reasoning content drafts at 0.306 acceptance vs 0.475 for prose and 0.857 for Triton, so `DEFAULT_THINKING=max` taxes throughput twice — more tokens *and* worse acceptance per token. |
| Benchmark corpus | **domain corpus, never prose** | §1 |

## 5. What is still open

- The residual between `algo` (48.8) and `triton` (75.6) is pure acceptance; no config lever
  found for it. It is a property of how predictable the emitted token stream is.
- `MAX_NUM_BATCHED_TOKENS` 8192 → 16384 is still unmeasured (the server warns
  `max_num_scheduled_tokens is set to 8168 … may lead to suboptimal performance`).
- `GPU_MEMORY_UTILIZATION_TEXT` 0.80 vs the recipe's 0.835 is still an uncontrolled delta.
- Run-to-run variance on n=3–5 short trials is ~10 %; anything smaller than that needs more repeats.
