# Why DS4 decode is what it is on 2× DGX Spark — a bandwidth model

Written 2026-08-18 on `toymaker`/`spark-2949`. Purpose: pick optimization targets by their
projected effect on end-to-end tok/s instead of by local kernel speedup.

## 1. Decode is MoE-weight-bandwidth bound, and the arithmetic closes

DeepSeek-V4-Flash-0731 (`config.json`): `hidden_size=4096`, `moe_intermediate_size=2048`,
`n_routed_experts=256`, `num_experts_per_tok=6`, `num_hidden_layers=43`, experts at 4 bits
(`expert_dtype: fp4`, served through `B12X_MXFP4` / `B12xExperts`). TP=2 shards the MoE
intermediate to 1024/rank.

Per expert per rank per layer: `fc1(gate+up) = 2·1024·4096` + `fc2(down) = 4096·1024`
= 12.58 M params → 6.29 MB at 4 bits.

| quantity | value |
|---|---|
| all 256 experts, 43 layers, per rank | **69.3 GB** |
| one token's top-6 experts, 43 layers | **1.62 GB** |
| one spec-decode step, ~36-expert union (6 positions × top-6, near-disjoint routing) | **9.74 GB** |

Measured device copy bandwidth (`probe_gb10_peak.py`, read+write): **223 GB/s** on an idle
GB10; the campaign's healthy reference for this box family is 97 bf16 / 205 fp8 TFLOP/s,
and the head reproduces it (97.9 / 207.6).

Predicted decode = `mean_accept_len / (bytes_per_step / bandwidth)`:

| rank bandwidth | ms/step | predicted tok/s @ accept 3.16 | observed |
|---|---|---|---|
| 223 GB/s (idle) | 43.7 | **72** | recipe's audited **73–76** |
| 105 GB/s (contended) | 92.8 | **34** | this rig measured **21–26** |

The model reproduces the recipe's published number to within ~4 % without any fitting, and
reproduces our degraded number once the worker's contended bandwidth is used. Two
consequences:

1. **The recipe's 73–76 tok/s is not optimistic marketing — it is what a healthy GB10 pair
   delivers when decode is bound by MoE weight traffic.**
2. **In TP=2 the slowest rank sets the pace.** Anything sharing the worker's GPU (or its
   LPDDR bandwidth — GB10 is unified memory, so a *CPU-side* hog counts too) is a
   first-order throughput regression, not a background nuisance.

## 2. Ranked levers, by projected effect on end-to-end tok/s

`tok/s = mean_accept_len / (bytes_per_step / effective_bandwidth)`. Only three terms.

| # | lever | mechanism | projected effect |
|---|---|---|---|
| 1 | **Keep both ranks' GPUs exclusive to DS4** | restores `effective_bandwidth` 105 → 223 GB/s | **up to 2.1×** |
| 2 | **Recover draft acceptance** — this rig 43.3 % overall / p0..p4 = 0.788/0.566/0.385/0.254/0.169, mean accept **3.16**; recipe audits 68–75 % and p0 0.93 → p4 0.33 | raises the numerator; costs no extra bytes (the 36-expert union is already paid) | **up to +40 %** (3.16 → ~4.4) |
| 3 | **W4A16 tile config** (§3) | changes efficiency of the same traffic | unknown, plausibly 5–15 % |
| 4 | **Concurrency** | more tokens amortize one expert-union read | already scales: agg 24.6 → 55.5 tok/s from c=1 → c=6 *while contended* |
| 5 | `max_num_batched_tokens` 8192 → 16384 | server warns `max_num_scheduled_tokens is set to 8168 … may lead to suboptimal performance` | prefill/decode interleave; no A/B exists in the repo |

Note that **acceptance is a model/sampling property and is unaffected by GPU contention**, so
levers 1 and 2 are independent and multiply.

## 3. The W4A16 tile-config space is exactly three configs — and one is illegal

Read out of the shipped image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
`b12x/moe/fused/w4a16/kernel.py`). `_select_tile_config` enumerates candidates, discards
those that do not fit shared memory, and **keeps whichever maximizes `blocks_per_sm`** —
i.e. it optimizes occupancy.

Tile → register-table key mapping is `cta_n_blocks = tile_n/16`, `cta_k_blocks = tile_k/16`.
`_W4A16_REGS_SM121` has 15 entries; for the decode path (`cta_m_blocks = 1`, i.e.
`moe_block_size ≤ 16`) only three tiles are legal:

| tile (TILE_K, TILE_N, CTA_THREADS) | regs key | blocks/SM | note |
|---|---|---|---|
| **(64, 128, 128)** | (128,1,8,4) | **3** | **always chosen**, for every M from 1→64, on both fc1 and fc2 |
| (128, 64, 128) | (128,1,4,8) | 2 | deeper K tile, half the occupancy |
| (128, 128, 256) | (256,1,8,8) | 1 | deepest tile, lowest occupancy |
| (64, 256, 256) | (256,1,16,4) | — | **illegal at cta_m_blocks=1** — no register entry; raises `ValueError` |

Occupancy-maximizing is a defensible default for compute-bound GEMMs, but this GEMM is
**memory-bound** (§1), where a deeper K tile can cut redundant weight/scale traffic. That is
exactly what the override exists for, and the repo contains **zero measurements** of it.

Override semantics (`vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py`):

- `VLLM_B12X_W4A16_FORCE_TILE_CONFIG="TILE_K,TILE_N,CTA_THREADS"` — forced tile; silently
  ignored if `_forced_b12x_w4a16_tile_blocks_per_sm` finds it doesn't fit (fail-safe).
- `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=N` (0 = don't force) — overrides only the occupancy term.
- `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=M` — the override applies **only when `problem_m ≤ M`**.
  Default **16**. `0` means *no limit*. A decode step can carry 6 seqs × 6 spec positions = 36
  rows, so **the shipped default of 16 excludes much of the decode path**; set `0` to cover it.
- The override installs at import **only if** `FORCE_BLOCKS_PER_SM > 0` or `FORCE_TILE_CONFIG`
  is non-empty. With the shipped defaults (`0` / `16` / empty) it is **inert** — so the three
  knobs currently do nothing at all on this deployment.

Each variant needs a container restart (env is read per call, but installation happens at import).

## 4. Method notes

- Bench harness: `bench-gb10/` (curl+jq). The repo's python benches cannot run here while DS4
  holds unified memory — `spark-gpu-admit-hook` refuses commands that may open a CUDA context —
  and they default to port 8888 / model `deepseek-v4-flash-0731` rather than 18888 /
  `deepseek-v4-flash-dspark`.
- Prompts must be natural prose. `scripts/EVAL.md` documents that repeated-token prompts
  collapse the drafter (p0→p4 0.93→0.33 healthy vs 0.72→0.11 pathological) and read as a fake
  decode regression.
- Every arm must record both nodes' SM MHz / W **sampled during the work**, and must state
  whether any non-DS4 GPU workload was resident on either node.
