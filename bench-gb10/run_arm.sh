#!/usr/bin/env bash
# run_arm.sh — one A/B arm of the DS4 serving campaign, end to end.
#
#   run_arm.sh <arm-label> [KEY=VALUE ...]
#
# Each KEY=VALUE is written into .env.dspark (replacing or appending) before the
# restart; pass none to measure the shipped defaults. The previous .env.dspark is
# restored on exit so arms never leak into each other.
#
# Refuses to run if any non-DS4 GPU workload is resident on either node — in TP=2
# the slowest rank sets the pace, so a contended arm is a wasted 15 minutes.
# Override deliberately with ALLOW_CONTENTION=1.
set -uo pipefail
ARM="${1:?arm label}"; shift || true
DIR="$HOME/orion/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark"
WORKER="${DSPARK_WORKER:?set DSPARK_WORKER to the worker node address}"
OUT="$DIR/results/toymaker/arm-${ARM}"
mkdir -p "$OUT"
cd "$DIR"

say(){ printf '\n== %s\n' "$*"; }

# ---- 0. exclusivity gate ----------------------------------------------------
# A batch sweep spawns one process per job, so a single nvidia-smi sample can land
# in the gap BETWEEN jobs and see an idle GPU. Sample across a window, and also
# match the driving shell by name — that survives the gaps.
CHK="$DIR/bench-gb10/busy_check.sh"
probe_busy() { # $1 = ssh target ("" = local). Sampled repeatedly: a batch sweep
               # spawns one process per job, so a single sample can land in the gap.
  local t="$1" out="" raw
  for _ in 1 2 3 4 5; do
    if [ -z "$t" ]; then raw=$(timeout 15 bash "$CHK" 2>/dev/null)
    else raw=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=6 "$t" bash -s < "$CHK" 2>/dev/null); fi
    raw=$(echo "$raw" | grep -v '^[[:space:]]*$' | head -2)
    if [ -n "$raw" ]; then out="$raw"; break; fi
    sleep 3
  done
  echo "$out"
}
H_BUSY=$(probe_busy ""); W_BUSY=$(probe_busy "$WORKER")
if [ -n "$H_BUSY$W_BUSY" ] && [ "${ALLOW_CONTENTION:-0}" != "1" ]; then
  echo "REFUSING: non-DS4 GPU workload resident (TP=2 is gated by the slowest rank)."
  [ -n "$H_BUSY" ] && echo "  head:   $H_BUSY"
  [ -n "$W_BUSY" ] && echo "  worker: $W_BUSY"
  echo "  re-run with ALLOW_CONTENTION=1 only if you intend a contended measurement."
  exit 3
fi

# ---- 1. apply env deltas ----------------------------------------------------
cp .env.dspark "$OUT/../.env.restore.$ARM"   # outside results/, for the restore trap only
trap 'cp "$OUT/../.env.restore.$ARM" "$DIR/.env.dspark"; rm -f "$OUT/../.env.restore.$ARM"; echo "(.env.dspark restored)"' EXIT
DELTAS=""
for kv in "$@"; do
  k="${kv%%=*}"
  grep -q "^${k}=" .env.dspark && sed -i "s|^${k}=.*|${kv}|" .env.dspark || printf '%s\n' "$kv" >> .env.dspark
  DELTAS="$DELTAS $kv"
done
say "arm=$ARM deltas:${DELTAS:- (shipped defaults)}"
# Record only the deltas, never a copy of .env.dspark: that file is gitignored
# because it carries this deployment's hosts, fabric addresses and paths, and
# copying it under results/ would publish exactly what the ignore rule prevents.
printf '%s\n' "${DELTAS:-(shipped defaults)}" > "$OUT/deltas.txt"

# ---- 2. restart -------------------------------------------------------------
say "restart"
./stop-deepseek-v4-flash-dspark.sh   > "$OUT/restart.log" 2>&1
./start-deepseek-v4-flash-dspark.sh >> "$OUT/restart.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ] || ! curl -s --max-time 8 http://127.0.0.1:18888/v1/models 2>/dev/null | grep -q deepseek; then
  echo "START FAILED (rc=$RC) — see $OUT/restart.log"; tail -20 "$OUT/restart.log"; exit 1
fi
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$OUT/restart.log" | tail -3 | tee "$OUT/kv.txt"
grep -iE "selector override|W4A16|tile_config" "$OUT/restart.log" | tail -5 | tee "$OUT/selector.txt"

# ---- 3. warm (first request after restart eats 6-10s of autotune) ----------
say "warm"
bench-gb10/decode_bench.sh "${ARM}-warm" --think off --tokens 64 --repeat 2 >/dev/null 2>&1

# ---- 4. measure -------------------------------------------------------------
say "decode (n=5, cold prefill, natural prose, decode window excludes TTFT)"
M0=$(curl -s --max-time 10 http://127.0.0.1:18888/metrics | awk '
  /^vllm:spec_decode_num_drafts_total/{d=$2} /^vllm:spec_decode_num_draft_tokens_total/{t=$2}
  /^vllm:spec_decode_num_accepted_tokens_total/{a=$2} END{print d,t,a}')
bench-gb10/decode_bench.sh "$ARM" --think off --temp 0.6 --tokens 128 --repeat 5 2>&1 | tee "$OUT/decode.txt" | tail -4
M1=$(curl -s --max-time 10 http://127.0.0.1:18888/metrics | awk '
  /^vllm:spec_decode_num_drafts_total/{d=$2} /^vllm:spec_decode_num_draft_tokens_total/{t=$2}
  /^vllm:spec_decode_num_accepted_tokens_total/{a=$2} END{print d,t,a}')
echo "$M0|$M1" | awk -F'|' '{split($1,a," ");split($2,b," ");
  dd=b[1]-a[1]; dt=b[2]-a[2]; da=b[3]-a[3];
  if(dt>0) printf "spec: drafts=%d draft_tok=%d accepted=%d  overall_acc=%.3f  mean_accept_len=%.2f\n",dd,dt,da,da/dt,1+da/dd;
  else print "spec: no draft activity"}' | tee "$OUT/spec.txt"

say "concurrency"
for C in 1 6; do bench-gb10/conc_bench.sh "$ARM" "$C" 512 2048; done 2>&1 | tee "$OUT/conc.txt"

say "clocks/power under load"
( curl -s --max-time 200 http://127.0.0.1:18888/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":"Write a long detailed essay about distributed consensus."}],"max_tokens":900,"min_tokens":900,"ignore_eos":true,"chat_template_kwargs":{"thinking":false}}' >/dev/null 2>&1 & )
sleep 4; bench-gb10/sample_load.sh "$ARM" 40 2 2>&1 | tail -4 | tee "$OUT/power.txt"

say "arm $ARM done -> $OUT"
