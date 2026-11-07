#!/usr/bin/env bash
# Runs ON a GB10 host: build the attention probe (flash_attn.cu + candidates) with the box's nvcc and
# the fatbin build's flags, then run it. usage: run_probe.sh TAG T TKVS REPS KERNELS
# Output: logs/probe-TAG.log (build resource usage + timing lines).
set -u
D=$HOME/.cache/memra-attn-gb10
TAG=$1; T=$2; TKVS=$3; REPS=$4; KS=$5
mkdir -p "$D/logs"
L=$D/logs/probe-$TAG.log
cd "$D/probe" || exit 1
{
  date -Is; sha256sum flash_attn.cu attn_cand.cuh probe_attn.cu
  nice -n 10 /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I. --resource-usage \
    probe_attn.cu -o probe_attn 2>&1 | grep -A2 -E "Function properties for (fa_prefill_qw_h2|cand_|diag_)" \
    | grep -E "properties|Used|spill"
  echo "nvcc rc=${PIPESTATUS[0]}"
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader
  ./probe_attn "$T" "$TKVS" "$REPS" "$KS"
  echo "probe rc=$?"
  nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
  date -Is
} > "$L" 2>&1
cat "$L"
