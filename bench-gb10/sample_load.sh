#!/usr/bin/env bash
# sample_load.sh — sample CPU / SM-clock / power on both Spark nodes while DS4 decodes.
# Usage: sample_load.sh <label> <seconds> [interval]
# Writes ~/orion/ds4bench/samples-<label>.tsv and prints a summary.
set -uo pipefail
LABEL="${1:?label}"; DUR="${2:-60}"; IVAL="${3:-2}"
WORKER=100.124.153.1
OUT="$HOME/orion/ds4bench/samples-${LABEL}.tsv"
mkdir -p "$(dirname "$OUT")"
printf 'ts\tnode\tsm_mhz\tpower_w\ttemp_c\tcpu_busy_pct\tvllm_cpu_pct\n' > "$OUT"

# Per-node one-shot sample. $1 = node label, empty ssh target = local.
sample_node() {
  local node="$1" ssh_t="${2:-}"
  local cmd='
    g=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
    # CPU busy% over a 1s window from /proc/stat
    read _ a b c idle rest < /proc/stat; t1=$((a+b+c+idle)); i1=$idle
    sleep 1
    read _ a b c idle rest < /proc/stat; t2=$((a+b+c+idle)); i2=$idle
    dt=$((t2-t1)); di=$((i2-i1))
    busy=$(( dt>0 ? (100*(dt-di))/dt : 0 ))
    # summed CPU% of vLLM worker/engine processes
    v=$(ps -eo pcpu,comm,args --no-headers | grep -E "VLLM::|vllm" | grep -v grep | awk "{s+=\$1} END{printf \"%.0f\", s}")
    echo "$g|$busy|${v:-0}"
  '
  local raw
  if [ -z "$ssh_t" ]; then raw=$(bash -c "$cmd" 2>/dev/null)
  else raw=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$ssh_t" "bash -s" <<<"$cmd" 2>/dev/null); fi
  [ -z "$raw" ] && return 0
  local gpu="${raw%%|*}" rest="${raw#*|}"
  local busy="${rest%%|*}" vcpu="${rest##*|}"
  local sm="${gpu%%,*}"; local pw_t="${gpu#*,}"; local pw="${pw_t%%,*}"; local tp="${pw_t##*,}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%H:%M:%S)" "$node" "$sm" "$pw" "$tp" "$busy" "$vcpu" >> "$OUT"
}

END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  sample_node head &
  sample_node worker "$WORKER" &
  wait
  sleep "$IVAL"
done

echo "== $LABEL (n=$(( $(wc -l < "$OUT") - 1 )) samples) =="
awk -F'\t' 'NR>1 {
  n[$2]++; sm[$2]+=$3; pw[$2]+=$4; tp[$2]+=$5; cb[$2]+=$6; vc[$2]+=$7
  if ($3+0>smx[$2]) smx[$2]=$3; if ($4+0>pwx[$2]) pwx[$2]=$4
} END {
  printf "%-7s %8s %8s %8s %8s %8s %9s\n","node","sm_avg","sm_max","W_avg","W_max","cpu%","vllm_cpu%"
  for (k in n) printf "%-7s %8.0f %8.0f %8.1f %8.1f %8.0f %9.0f\n", k, sm[k]/n[k], smx[k], pw[k]/n[k], pwx[k], cb[k]/n[k], vc[k]/n[k]
}' "$OUT"
echo "raw: $OUT"
