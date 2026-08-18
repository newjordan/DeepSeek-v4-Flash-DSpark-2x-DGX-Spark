#!/usr/bin/env bash
# decode_bench.sh — decode-rate + spec-decode-acceptance bench for the DS4 endpoint.
#
# Follows the recipe's own EVAL.md rules:
#   * cold prefill (unique nonce busts the prefix cache)
#   * decode rate EXCLUDES TTFT   (t_first_token -> t_last_token)
#   * token count from usage.completion_tokens (SSE chunk-counting undercounts ~4.5x)
#   * warm the server before trusting a number (first request after restart = 6-10s autotune)
# Bash/curl/jq only: the spark-gpu-admit-hook blocks python3 while DS4 holds memory.
#
# Usage: decode_bench.sh <label> [opts]
#   --think max|high|low|off   default max (server default)
#   --budget N                 thinking_token_budget (omit = none)
#   --temp F                   default 0.6
#   --tokens N                 min=max completion tokens, default 128
#   --repeat N                 trials, default 3 (reports median)
#   --prompt-words N           filler length, default 24 (~1.56 tok/word)
set -uo pipefail
LABEL="${1:?label}"; shift || true
THINK=max; BUDGET=""; TEMP=0.6; TOK=128; REP=3; PW=24
while [ $# -gt 0 ]; do
  case "$1" in
    --think) THINK="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --temp) TEMP="$2"; shift 2;;
    --tokens) TOK="$2"; shift 2;;
    --repeat) REP="$2"; shift 2;;
    --prompt-words) PW="$2"; shift 2;;
    *) echo "unknown opt $1" >&2; exit 2;;
  esac
done
BASE=http://127.0.0.1:18888
MODEL=deepseek-v4-flash-dspark

# Readiness gate: a server still loading weights answers /v1/models with a
# connection error and every chat request instantly with 0 tokens, which used to
# surface as a nonsensical negative decode rate instead of an error.
if ! curl -s --max-time 8 "$BASE/v1/models" 2>/dev/null | grep -q "$MODEL"; then
  echo "ABORT: $BASE is not serving $MODEL yet (still loading weights?)." >&2
  docker logs --tail 3 deepseek-v4-flash-vllm-dspark-1 2>&1 | tail -2 >&2
  exit 4
fi
OUT="$HOME/orion/ds4bench/decode-${LABEL}.tsv"
mkdir -p "$(dirname "$OUT")"
printf 'trial\tttft_s\tdecode_tok_s\tcompletion_tok\tfinish\taccept_len\tacc_pos\n' > "$OUT"

# --- spec-decode counters from /metrics -------------------------------------
metrics_snap() {
  curl -s --max-time 10 "$BASE/metrics" 2>/dev/null | awk '
    /^vllm:spec_decode_num_draft_tokens_total/ {d=$2}
    /^vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
    /^vllm:spec_decode_num_accepted_tokens_per_pos_total\{/ {
      if (match($0,/position="[0-9]+"/)) {
        p=substr($0,RSTART+10,RLENGTH-11); pos[p]+=$2
      }
    }
    END {printf "%s|%s|", d+0, a+0; for(i=0;i<8;i++) if(i in pos) printf "%s,", pos[i]; print ""}'
}

# Natural prose, NOT a repeated-word pool. scripts/EVAL.md: pathological
# repeated-token prompts collapse the drafter (per-position 0.93->0.33 good vs
# 0.72->0.11 bad) and read as a fake decode regression. Rotate real sentences.
filler() { # $1 = approx words (rounded up to whole sentences), $2 = trial nonce
  local w=$1 n=$2 s="" i=0
  local -a S=(
"The storage engine writes each committed transaction to a durable log before acknowledging the client."
"Cache coherence protocols must decide whether a line is shared, exclusive, or invalid at every access."
"A scheduler that ignores tail latency will happily starve the requests that users actually notice."
"Register pressure decides whether the compiler keeps a value in flight or spills it to the stack."
"Network partitions are not hypothetical; every long-lived cluster eventually loses a link mid-write."
"Profiling before optimizing avoids the classic mistake of tuning a loop that never dominated runtime."
"Garbage collectors trade throughput for pause time, and the right trade depends on the workload shape."
"A well-chosen index turns a full table scan into a handful of page reads and a short traversal."
)
  while [ $(echo "$s" | wc -w) -lt "$w" ]; do
    s="$s ${S[$(( (i + n) % ${#S[@]} ))]}"; i=$((i+1))
    [ $i -gt 40 ] && break
  done
  echo "$s"
}

case "$THINK" in
  off) CTK='{"thinking":false}' ;;
  *)   CTK="{\"thinking\":true,\"reasoning_effort\":\"$THINK\"}" ;;
esac

for t in $(seq 1 "$REP"); do
  NONCE="$LABEL-$t-$$-$(od -An -N4 -tu4 </dev/urandom | tr -d ' ')"
  PROMPT="[$NONCE] Continue this technical note in prose:$(filler "$PW" "$t")"
  BODY=$(jq -nc --arg m "$MODEL" --arg p "$PROMPT" --argjson ctk "$CTK" \
    --argjson tk "$TOK" --argjson tp "$TEMP" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$tk,min_tokens:$tk,
      ignore_eos:true,temperature:$tp,top_p:0.95,stream:true,
      stream_options:{include_usage:true},chat_template_kwargs:$ctk}')
  [ -n "$BUDGET" ] && BODY=$(echo "$BODY" | jq -c --argjson b "$BUDGET" '.thinking_token_budget=$b')

  M0=$(metrics_snap)
  T0=$(date +%s.%N)
  # stamp each SSE data line with its arrival time
  curl -s -N --max-time 900 "$BASE/v1/chat/completions" \
       -H 'content-type: application/json' -d "$BODY" 2>/dev/null \
    | while IFS= read -r line; do printf '%s\t%s\n' "$(date +%s.%N)" "$line"; done > "/tmp/dbench.$$"
  T_END=$(date +%s.%N)
  M1=$(metrics_snap)

  # first line carrying a real token delta, and last such line
  TFIRST=$(awk -F'\t' '$2 ~ /^data: / {j=substr($2,7); if (j ~ /"(content|reasoning)":"[^"]/) {print $1; exit}}' "/tmp/dbench.$$")
  TLAST=$(awk -F'\t'  '$2 ~ /^data: / {j=substr($2,7); if (j ~ /"(content|reasoning)":"[^"]/) l=$1} END{print l}' "/tmp/dbench.$$")
  CTOK=$(awk -F'\t' '$2 ~ /^data: / {j=substr($2,7); if (j ~ /"completion_tokens"/) print j}' "/tmp/dbench.$$" \
          | tail -1 | jq -r '.usage.completion_tokens // empty')
  FIN=$(awk -F'\t' '$2 ~ /^data: / {j=substr($2,7); if (j ~ /"finish_reason":"[a-z]/) print j}' "/tmp/dbench.$$" \
          | tail -1 | jq -r '.choices[0].finish_reason // empty')
  rm -f "/tmp/dbench.$$"

  TTFT=$(echo "${TFIRST:-$T_END} - $T0" | bc -l)
  WIN=$(echo "${TLAST:-$T_END} - ${TFIRST:-$T0}" | bc -l)
  CTOK=${CTOK:-0}
  if [ "${CTOK:-0}" -lt 2 ]; then
    echo "ABORT: trial $t returned $CTOK completion tokens (server not serving, or request rejected)." >&2
    exit 5
  fi
  RATE=$(echo "if ($WIN > 0) ($CTOK - 1) / $WIN else 0" | bc -l)

  # acceptance deltas
  D0=${M0%%|*}; R0=${M0#*|}; A0=${R0%%|*}; P0=${R0#*|}
  D1=${M1%%|*}; R1=${M1#*|}; A1=${R1%%|*}; P1=${R1#*|}
  DD=$(echo "$D1 - $D0" | bc -l); AA=$(echo "$A1 - $A0" | bc -l)
  # mean accepted length = 1 + accepted/ (draft/k)  -> report accepted/drafted ratio + per-pos deltas
  ACCLEN=$(echo "if ($DD > 0) $AA / $DD else 0" | bc -l)
  ACCPOS=$(paste -d, <(echo "$P0" | tr ',' '\n') <(echo "$P1" | tr ',' '\n') \
            | awk -F, 'NF==2 && $2!="" {d=$2-$1; printf "%s ", d}')

  printf '%s\t%.3f\t%.2f\t%s\t%s\t%.3f\t%s\n' "$t" "$TTFT" "$RATE" "$CTOK" "${FIN:-?}" "$ACCLEN" "$ACCPOS" >> "$OUT"
done

echo "== $LABEL  think=$THINK budget=${BUDGET:-none} temp=$TEMP tokens=$TOK n=$REP =="
column -t "$OUT"
awk -F'\t' 'NR>1 {r[NR-1]=$3; f[NR-1]=$2; a[NR-1]=$6; n++}
  END {asort(r); asort(f); asort(a);
    printf "MEDIAN  decode=%.2f tok/s   ttft=%.3f s   accepted/drafted=%.3f\n",
      r[int((n+1)/2)], f[int((n+1)/2)], a[int((n+1)/2)]}' "$OUT" 2>/dev/null \
 || awk -F'\t' 'NR>1{s+=$3;t+=$2;c+=$6;n++} END{printf "MEAN    decode=%.2f tok/s   ttft=%.3f s   accepted/drafted=%.3f\n", s/n, t/n, c/n}' "$OUT"
echo "raw: $OUT"
