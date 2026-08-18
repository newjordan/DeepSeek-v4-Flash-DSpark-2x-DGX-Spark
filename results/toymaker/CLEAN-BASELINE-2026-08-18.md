# Clean baseline — quiet cluster, shipped defaults (2026-08-18)

Supersedes the retracted `BASELINE-2026-08-18.md`, whose decode numbers were taken while an
unrelated 12-job sweep held the worker's GPU. That sweep finished at **14:48 CDT**; both nodes
verified quiet (`bench-gb10/busy_check.sh` empty on head and worker) before and during these runs.

Config: shipped recipe defaults at `689ba32` (incl. the new #79 GB10 spin-wait hotfix),
`GPU_MEMORY_UTILIZATION_TEXT=0.80` (local), image digest `a839484…`, checkpoint `7872f01b`
(= released `9e165c30` weights). Fresh container, warmed before measuring.
KV at 0.80: **13.18 GiB / 1,824,834 tokens / 1.74×**.

## Contention was worth ~1.7×

`bench-gb10/decode_bench.sh --think off --temp 0.6 --tokens 128 --repeat 5`, decode window
excludes TTFT, cold prefill per trial, natural-prose prompts:

| | contended (worker sweep resident) | **quiet** | gain |
|---|---|---|---|
| C1 decode, median | 20.9–25.6 tok/s | **39.9** | **+56–91 %** |
| TTFT | 0.28–0.36 s | 0.27 s | — |

`bench-gb10/conc_bench.sh` (512 tok/stream, `thinking_token_budget=2048`, temp 0):

| c | contended agg | **quiet agg** | quiet per-stream |
|---|---|---|---|
| 1 | 24.6 | **28.5** | 28.5 |
| 2 | 33.7 | **52.8** | 26.4 |
| 4 | 46.6 | **66.3** | 16.6 |
| 6 | 55.5 | **71.4** | 11.9 |

**Operational rule this establishes:** in TP=2 the slowest rank sets the pace, and GB10 is
unified memory, so *any* GPU or bandwidth-heavy job on either node is a first-order serving
regression. `bench-gb10/run_arm.sh` now refuses to measure unless both nodes are quiet
(`ALLOW_CONTENTION=1` to override deliberately).

## The remaining gap is draft acceptance, and reasoning is a double tax

Still short of the audited 73–76 tok/s C1 / 168–180 c=6 aggregate. Per the bandwidth model
(`DECODE-MODEL-2026-08-18.md`), decode = `mean_accept_len / (bytes_per_step / bandwidth)`, and
with bytes and bandwidth now healthy the only free term left is acceptance.

Measured on one prompt, same server, varying only generation mode:

| mode | overall acceptance | mean accept length |
|---|---|---|
| natural prose, temp 0.6 | **0.475** | **3.38** |
| natural, greedy temp 0 | 0.453 | 3.27 |
| forced `ignore_eos`, min=max=256 | 0.432 | 3.16 |
| **with reasoning (`thinking_token_budget=1024`)** | **0.306** | **2.53** |

Two conclusions:

1. **`ignore_eos` is not the artifact** — forcing generation past the natural stop costs only
   0.475 → 0.432. The recipe's own bench uses `ignore_eos`, so this is comparable.
2. **Reasoning content drafts far worse than prose** (0.306 vs 0.475, −36 %). Since
   `DEFAULT_THINKING=max` is the shipped default, every request without an explicit budget pays
   twice: it emits far more tokens *and* each token is ~36 % more expensive in decode steps.
   This compounds with the client-side finding in `DS4_ANGELX_HARNESS_REPORT_20260818.md`.

Audited acceptance is 68–75 % (mean ~4.5). At 3.38 we are at ~75 % of that; closing it would
take C1 39.9 → ~53 tok/s, and the residual to 73–76 remains unexplained — candidates are the
0.80 vs 0.835 util delta and prompt/content differences in the audit corpus.

## Method fixes applied after errors in the first pass

- The bench aborts if `/v1/models` is not serving yet, and if any trial returns <2 completion
  tokens. A server still loading weights previously produced a *negative* decode rate
  (0 tokens / tiny window) instead of an error.
- Prompts are natural prose, not a cycled word pool — `scripts/EVAL.md` documents that
  repeated-token prompts collapse the drafter and read as a fake regression.
- The contention gate samples repeatedly (a batch sweep spawns one process per job, so a single
  `nvidia-smi` sample can land in the gap between jobs) and keys on process *name*, not cmdline,
  so monitor shells mentioning a job path do not trip it.
- `ps pcpu` is a lifetime average and is useless for a freshly restarted container; the sampler
  uses instantaneous CPU instead.
