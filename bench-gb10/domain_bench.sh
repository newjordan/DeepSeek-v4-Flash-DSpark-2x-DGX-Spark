#!/usr/bin/env bash
# domain_bench.sh — decode rate + draft acceptance + INFERRED MoE expert-union,
# per workload domain. Prose is a control, not the target.
#
# Why inferred routing works: decode on this deployment is MoE-weight-bandwidth
# bound (results/toymaker/DECODE-MODEL-2026-08-18.md). Therefore
#     ms_per_step   = mean_accept_len / decode_tok_s
#     bytes_per_step= ms_per_step * effective_bandwidth
#     expert_union  = bytes_per_step / (bytes_per_expert_layer * layers)
# so measuring decode+acceptance on a domain tells you how many distinct experts
# that domain's routing touches per step. Concentrated routing => fewer bytes =>
# faster decode at equal acceptance.
#
# Usage: domain_bench.sh <label> [tokens] [repeat]
set -uo pipefail
LABEL="${1:-dom}"; TOK="${2:-256}"; REP="${3:-3}"
BASE=http://127.0.0.1:18888; MODEL=deepseek-v4-flash-dspark
BW_GBS="${BW_GBS:-223}"       # measured idle copy bandwidth, GB/s
EFF="${EFF:-0.65}"            # GEMM fraction of copy peak (documented assumption)
BYTES_EXPERT_LAYER=6291456    # 12.58M params * 4 bit / 8, per rank
LAYERS=43
OUT="$HOME/orion/ds4bench/domain-${LABEL}.tsv"
mkdir -p "$(dirname "$OUT")"

curl -s --max-time 8 "$BASE/v1/models" 2>/dev/null | grep -q "$MODEL" || { echo "ABORT: $MODEL not serving" >&2; exit 4; }

snap(){ curl -s --max-time 10 "$BASE/metrics" | awk '
  /^vllm:spec_decode_num_drafts_total/{d=$2}
  /^vllm:spec_decode_num_draft_tokens_total/{t=$2}
  /^vllm:spec_decode_num_accepted_tokens_total/{a=$2} END{print d+0,t+0,a+0}'; }

# ---- corpus: the operator's actual domains. Prose last, as a control. -------
domain_prompt() {
case "$1" in
cuda) cat <<'P'
Here is a CUDA kernel fragment for a Blackwell (sm_120) FP8 attention epilogue:

  __global__ void v_fp8_tilemax(const __nv_bfloat16* __restrict__ V,
                                __nv_fp8_e4m3* __restrict__ Vq,
                                float* __restrict__ amax, int N, int D) {
    extern __shared__ float smem[];
    const int tile = blockIdx.x, lane = threadIdx.x & 31;
    float m = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
      m = fmaxf(m, fabsf(__bfloat162float(V[(size_t)tile * D + i])));
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    ...
  }

Rewrite this so the amax reduction uses a two-stage warp then block reduction with
no shared-memory bank conflicts, and the quantization pass reuses the loaded V
values instead of re-reading global memory. Show the corrected kernel and explain
the occupancy and register-pressure consequences on sm_120.
P
;;
triton) cat <<'P'
Write a Triton kernel implementing a fused RMSNorm + split-half RoPE for a VAE
attention block, with these constraints: BLOCK_M and BLOCK_D are constexpr, the
head dimension is 128, rotary is applied to the first half and the second half is
passed through unchanged, and the epilogue writes bf16 while accumulating in fp32.
Include the autotune configs you would search, state which ones you expect to win
on a memory-bound shape, and explain how you would verify bit-level agreement
against the eager PyTorch reference.
P
;;
algo) cat <<'P'
Competitive programming. You are given an array a[1..n] (n up to 2*10^5) and q
queries (q up to 2*10^5). Each query gives l, r, k and asks for the k-th smallest
value of a[i] for i in [l, r], but with the twist that values equal to a previously
answered query's result are excluded from consideration for all later queries.
Design an algorithm meeting an O((n + q) log^2 n) bound, prove the complexity,
give the exact data structures, and write the complete C++17 implementation with
fast IO.
P
;;
math) cat <<'P'
Let p be an odd prime and let f(x) = x^d + c over F_p with gcd(d, p-1) = e > 1.
Consider the directed functional graph of f on F_p. Prove a sharp bound on the
number of periodic points of f in terms of e and p, determine when the bound is
attained, and give the structure of the connected components. State every lemma
you use, prove each one, and give a counterexample showing the bound fails when
the coprimality hypothesis is dropped.
P
;;
science) cat <<'P'
A tokamak plasma has a q-profile q(r) = q0 (1 + (r/a)^2)^(3/2) with q0 = 0.85.
Derive the location and growth rate of the m=1, n=1 internal kink mode using
reduced MHD, state the resistive correction that produces the sawtooth crash,
and estimate the crash time for T_e = 4 keV, n_e = 10^20 m^-3, a = 0.6 m.
Show every step of the derivation and give the dimensional analysis.
P
;;
prose) cat <<'P'
Explain how a modern database query planner chooses between a hash join and a
nested loop join, with concrete examples of when each wins and why the cardinality
estimate matters more than the join algorithm itself.
P
;;
esac
}

printf 'domain\ttrial\tdecode_tok_s\taccept_len\tacc_ratio\tms_step\tGB_step\texpert_union\n' > "$OUT"
for dom in cuda triton algo math science prose; do
  P="$(domain_prompt "$dom")"
  for t in $(seq 1 "$REP"); do
    NONCE="$LABEL-$dom-$t-$(od -An -N4 -tu4 </dev/urandom | tr -d ' ')"
    BODY=$(jq -nc --arg m "$MODEL" --arg p "[$NONCE] $P" --argjson tk "$TOK" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$tk,min_tokens:$tk,
        ignore_eos:true,temperature:0.6,top_p:0.95,stream:true,
        stream_options:{include_usage:true},chat_template_kwargs:{thinking:false}}')
    M0=$(snap); T0=$(date +%s.%N)
    curl -s -N --max-time 900 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
      -d "$BODY" 2>/dev/null | while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done > "/tmp/db.$$"
    M1=$(snap)
    TF=$(awk -F'\t' '$2~/^data: /{j=substr($2,7); if(j~/"(content|reasoning)":"[^"]/){print $1;exit}}' "/tmp/db.$$")
    TL=$(awk -F'\t' '$2~/^data: /{j=substr($2,7); if(j~/"(content|reasoning)":"[^"]/)l=$1} END{print l}' "/tmp/db.$$")
    CT=$(awk -F'\t' '$2~/^data: /{j=substr($2,7); if(j~/"completion_tokens"/)print j}' "/tmp/db.$$" | tail -1 | jq -r '.usage.completion_tokens//empty')
    rm -f "/tmp/db.$$"
    [ "${CT:-0}" -lt 2 ] && { echo "ABORT: $dom trial $t returned ${CT:-0} tokens" >&2; exit 5; }
    echo "$M0|$M1|$TF|$TL|$CT" | awk -F'|' -v d="$dom" -v t="$t" -v bw="$BW_GBS" -v eff="$EFF" \
      -v bel="$BYTES_EXPERT_LAYER" -v L="$LAYERS" -v out="$OUT" '{
        split($1,a," "); split($2,b," ");
        dd=b[1]-a[1]; dt=b[2]-a[2]; da=b[3]-a[3];
        win=$4-$3; ct=$5;
        rate=(win>0)?(ct-1)/win:0;
        acc=(dt>0)?da/dt:0; alen=(dd>0)?1+da/dd:0;
        ms=(rate>0)?alen/rate*1000:0;
        gb=ms/1000*bw*eff;
        eu=(gb*1e9)/(bel*L);
        printf "%s\t%s\t%.2f\t%.2f\t%.3f\t%.1f\t%.2f\t%.1f\n",d,t,rate,alen,acc,ms,gb,eu >> out
      }'
  done
done

echo "== $LABEL  (tokens=$TOK, n=$REP, bandwidth=${BW_GBS} GB/s x eff ${EFF}) =="
awk -F'\t' 'NR>1{n[$1]++;r[$1]+=$3;a[$1]+=$4;c[$1]+=$5;m[$1]+=$6;g[$1]+=$7;e[$1]+=$8}
 END{printf "%-9s %10s %11s %8s %9s %9s %8s\n","domain","decode t/s","accept_len","acc","ms/step","GB/step","experts";
   for(k in n) printf "%-9s %10.2f %11.2f %8.3f %9.1f %9.2f %8.1f\n",k,r[k]/n[k],a[k]/n[k],c[k]/n[k],m[k]/n[k],g[k]/n[k],e[k]/n[k]}' "$OUT" | sort -k2 -rn
echo "raw: $OUT"
