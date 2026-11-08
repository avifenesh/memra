#!/usr/bin/env bash
# Runs ON a GB10 host: build the round-4 attention probe (flash_attn.cu of a tree + attn_pp.cuh) with
# the box's nvcc and the fatbin build's flags, then run it. usage: run_probe.sh TAG TREE T TKVS REPS KERNELS
# Output: logs/probe-TAG.log (sources' sha256, build resource usage, timing lines).
set -u
D=$HOME/.cache/memra-gb10-r4
TAG=$1; TREE=$2; T=$3; TKVS=$4; REPS=$5; KS=$6
P=$D/probe
mkdir -p "$D/logs"
L=$D/logs/probe-$TAG.log
cp "$TREE/crates/memra-engine/cu/flash_attn.cu" "$P/flash_attn.cu" || exit 1
cd "$P" || exit 1
{
  date -Is; git -C "$TREE" log --oneline -1; sha256sum flash_attn.cu attn_pp.cuh probe_pp.cu
  /usr/local/cuda/bin/nvcc --version | tail -n 1
  nice -n 10 /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I. -I"$TREE/crates/memra-engine/cu" \
    --resource-usage probe_pp.cu -o probe_pp 2>&1 | grep -A2 -E "Function properties for (fa_prefill_qw_h2p?|cand_)" \
    | grep -E "properties|Used|spill"
  echo "nvcc rc=${PIPESTATUS[0]}"
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader
  timeout 900 ./probe_pp "$T" "$TKVS" "$REPS" "$KS"
  echo "probe rc=$?"
  nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
  date -Is
} > "$L" 2>&1
cat "$L"
