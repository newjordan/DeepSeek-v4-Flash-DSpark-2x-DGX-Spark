#!/usr/bin/env bash
# conc_bench.sh — aggregate decode throughput vs concurrency on the DS4 endpoint.
# Usage: conc_bench.sh <label> <concurrency> [max_tokens] [think_budget]
# Bash/curl/jq only (the spark-gpu-admit-hook blocks python3 while DS4 holds memory).
set -uo pipefail
LABEL="${1:?label}"; C="${2:?concurrency}"; MT="${3:-512}"; TB="${4:-2048}"
URL=http://127.0.0.1:18888/v1/chat/completions
DIR="$HOME/orion/ds4bench/run-${LABEL}-c${C}"
mkdir -p "$DIR"

# Distinct prompts so prefix caching cannot serve one request from another.
prompt_for() {
  case $(( $1 % 6 )) in
    0) echo "Explain how a B-tree index answers a range query, step by step." ;;
    1) echo "Describe the memory hierarchy of a modern GPU and why coalescing matters." ;;
    2) echo "Walk through the CAP theorem with a concrete partition scenario." ;;
    3) echo "Explain how a Kalman filter fuses noisy sensor measurements." ;;
    4) echo "Describe how a JIT compiler decides what to inline, with tradeoffs." ;;
    5) echo "Explain the mechanics of TCP congestion control under packet loss." ;;
  esac
}

START=$(date +%s.%N)
for i in $(seq 1 "$C"); do
  P="$(prompt_for "$i") (variant $i)"
  BODY=$(jq -nc --arg p "$P" --argjson mt "$MT" --argjson tb "$TB" \
    '{model:"deepseek-v4-flash-dspark",messages:[{role:"user",content:$p}],max_tokens:$mt,thinking_token_budget:$tb,temperature:0}')
  ( t0=$(date +%s.%N)
    curl -s --max-time 900 "$URL" -H 'content-type: application/json' -d "$BODY" > "$DIR/r$i.json"
    t1=$(date +%s.%N)
    echo "$t1 - $t0" | bc > "$DIR/r$i.wall" ) &
done
wait
END=$(date +%s.%N)

WALL=$(echo "$END - $START" | bc)
TOT=$(cat "$DIR"/r*.json | jq -s '[.[].usage.completion_tokens // 0] | add')
PTOT=$(cat "$DIR"/r*.json | jq -s '[.[].usage.prompt_tokens // 0] | add')
OK=$(cat "$DIR"/r*.json | jq -s '[.[] | select(.choices[0].finish_reason != null)] | length')
printf '%-26s c=%-3s wall=%6.2fs  completion_tok=%-6s agg=%6.2f tok/s  per-stream=%5.2f tok/s  ok=%s/%s prompt_tok=%s\n' \
  "$LABEL" "$C" "$WALL" "$TOT" "$(echo "$TOT / $WALL" | bc -l)" "$(echo "$TOT / $WALL / $C" | bc -l)" "$OK" "$C" "$PTOT"
