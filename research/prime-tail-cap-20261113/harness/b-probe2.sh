#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): the served cold prime's footprint by
# prompt length on the diag build (MEMRA_ALLOC_TRACE=1), pool drained before each prime; two
# boots with the lengths in opposite orders. usage: b-probe2.sh [ARM] [UP_WORDS] [DOWN_WORDS]
set -u
ARM=${1:-diag3}
UP=${2:-2540,2556,2570,2620,2870,3582,5112}
DOWN=${3:-5112,3582,2870,2620,2570,2556,2540}
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/run4/$ARM-$(date +%H%M%S)
mkdir -p "$OUT"
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; } > "$OUT/guard.log"
rc=0
for arm in "up $UP" "down $DOWN"; do
  set -- $arm
  setsid python3 "$B/b-probe.py" --bin "$S/bin/memra-server-$ARM-120a" --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
    --draft "$M/Qwen3.8-27B-DFlash2-memra" --out "$OUT/$1" --trim 1 --words "$2" > "$OUT/$1.log" 2>&1 &
  P=$!
  echo "$P" >> "$S/pids.txt"
  trap 'kill -TERM -- -"$P" 2>/dev/null; exit 143' TERM INT
  wait "$P" || rc=$?
  kill -- -"$P" 2>/dev/null
  echo "$1 done rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
done
exit $rc
