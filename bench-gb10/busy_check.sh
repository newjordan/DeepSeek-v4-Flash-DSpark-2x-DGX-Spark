#!/usr/bin/env bash
# busy_check.sh — print non-DS4 GPU/compute activity on THIS node; empty = quiet.
# Run locally, or remotely with:  ssh <host> bash -s < busy_check.sh
#
# Two independent signals:
#   (a) GPU residency by any non-vLLM process (authoritative, but a batch sweep
#       spawns one process per job so a single sample can land between jobs);
#   (b) a real compute process burning CPU, keyed on the PROCESS NAME so that
#       monitor shells merely mentioning a job path do not count.
nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
  | grep -vi vllm | grep -v '^[[:space:]]*$'
ps -eo pcpu=,comm=,args= 2>/dev/null \
  | awk '$1 > 50 && $2 ~ /^(python|python3|torchrun|pt_main_thread|ray)/ { print }' \
  | grep -vi vllm | cut -c1-110
exit 0
