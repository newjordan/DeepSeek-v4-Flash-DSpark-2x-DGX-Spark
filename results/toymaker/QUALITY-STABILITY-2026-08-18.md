# Quality & long-context stability audit — toymaker/spark-2949, 2026-08-18

Priority order for this deployment is **answer quality and long-chain stability first, raw
speed second**. This is the evidence for the first two.

Cluster verified quiet (`bench-gb10/busy_check.sh` empty on both nodes). Shipped recipe
defaults at `689ba32`, `MTP_NUM_TOKENS=5`, util 0.80, image digest `a839484…`, checkpoint
`7872f01b` (= released `9e165c30` weights). Endpoint `127.0.0.1:18888`, model
`deepseek-v4-flash-dspark`.

## Headline: everything passes, at real depth

| gate | depth | result |
|---|---|---|
| RULER-lite (retrieval, 3-key retrieval, 4-hop vartrack, common-word aggregation) | 8,199–8,204 | **4/4** |
| RULER-lite | 32,775–32,781 | **4/4** |
| RULER-lite | 131,079–131,084 | **4/4** (~80 s/case) |
| RULER-lite | 262,144–262,167 | **4/4** (165–184 s/case) |
| Cold-prefill garble sweep (prompt echo, schema dump, secret leak, mid-word start, special tokens) | 2k / 8k / 32k / 131k × 2 runs | **ALL CLEAN** |
| Deep-context tool battery (single call, multi-turn, complex args, issue-55 truncation) | 32,768 | **4/4** |
| Deep-context tool battery | 131,072 | **4/4** |
| Warm prefix-cache chain reuse, 8 concurrent lanes | ~86k cached/lane | **8/8 lanes, hit ratio 0.9735** on both warm waves (cold wave 0.0000, as required) |

Prefix caching is demonstrably live: in the garble sweep the 131k cold run took **71.4 s** and
its identical replay took **3.9 s**.

## Defect found and fixed: the long-context gate was not testing long context

`scripts/ruler-lite.py::pad_to_length()` appended a single ~25-token `HAYSTACK_SENTENCE` per
`/tokenize` round trip under a fixed `guard < 200`, capping achievable context at
**~5.6k tokens**. Every requested depth ≥ 8192 was silently evaluated at that ceiling *and
still printed as PASS at the requested depth*.

Observed here before the fix, `--lengths 8192,32768`:

```
ctx=4840  4868  5396  5631      <- "8k" arm
ctx=4840  4869  5396  5631      <- "32k" arm — identical
=== RULER-lite: 8/8 passed ===
```

So `AUDIT.md`'s expected value *"RULER-lite: 8/8 at 8k/32k"* had never gated long context, and
neither had any CI run of it. Fixed to pad in deficit-sized chunks and to **raise RuntimeError
rather than return a short prompt**, so failing to reach depth can no longer be scored as a
pass. The table above is the first run of this gate at its nominal depths.

Second, smaller fix: `scripts/reproduce-issue26-live.py` hardcoded `127.0.0.1:8888` and
`deepseek-v4-flash-0731`, so it could not run against any deployment with a different port or
`SERVED_MODEL_NAME`. It now honours `DSPARK_API` / `DSPARK_MODEL`, exactly as its sibling
`reproduce-issue43-live.py` already did.

## Acceptance is a speed axis, not a quality axis

Worth stating explicitly because it is easy to read the acceptance numbers the wrong way. The
shipped image performs speculative decoding via `vllm/v1/sample/rejection_sampler.py::rejection_sample()`
with probabilistic draft probs — i.e. distribution-preserving rejection sampling. A rejected
draft token costs a decode step; it cannot change the sampled output distribution. So the 2×
acceptance spread across content domains (0.433 prose → 0.857 Triton, see
`SERVING-POSTURE-2026-08-18.md`) is a **throughput** phenomenon only, and low acceptance is not
evidence of degraded answers.

## Reproduce

```bash
cd ~/orion/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
B=http://127.0.0.1:18888/v1 ; M=deepseek-v4-flash-dspark
# NOTE: spark-gpu-admit-hook blocks any Bash containing `python3` while DS4 holds unified
# memory. These five audit scripts import only stdlib + urllib and open no CUDA context, so
# SPARK_GPU_ADMIT=force is correct for them specifically — verify with:
#   grep -E '^import |^from ' scripts/<script>.py
SPARK_GPU_ADMIT=force python3 scripts/ruler-lite.py --base-url $B --model $M --lengths 8192,32768,131072,262144
SPARK_GPU_ADMIT=force python3 scripts/context-garble-sweep.py --url $B --model $M --lengths 2048,8192,32768,131072 --runs 2
SPARK_GPU_ADMIT=force python3 scripts/deepctx-tool-battery.py $B/chat/completions $M 32768,131072
SPARK_GPU_ADMIT=force DSPARK_API=http://127.0.0.1:18888 DSPARK_MODEL=$M python3 scripts/reproduce-issue26-live.py 8 32768
```

## Not yet run

- `scripts/reproduce-issue43-live.py` — decode fairness across concurrent long prefills
  (32K/62K × 2/4/8). Relevant only if several long chains run at once.
- RULER-lite at 524k / 900k. The 262k case already costs ~180 s of cold prefill per task.
