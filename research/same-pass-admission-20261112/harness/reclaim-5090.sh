#!/usr/bin/env bash
# Laptop, inside a gpulock lease on the RTX 5090 (memra#1255 review round 2): one run of
# harness/reclaim-cell.py against an arm's binary. usage: reclaim-5090.sh NAME ARM
# The power limit is recorded, never changed.
set -u
NAME=$1; ARM=$2
S=$HOME/.cache/memra-1255
M=$HOME/models/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$S/run/5090/reclaim/$NAME
mkdir -p "$OUT"
LOG=$OUT/cell.log
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  uptime; } > "$LOG"
setsid python3 "$S/harness/reclaim-cell.py" --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  --draft "$M/Qwen3.8-27B-DFlash2-memra" --bin "$S/bin/memra-server-$ARM-120a" --out "$OUT" >> "$LOG" 2>&1 &
GP=$!
echo "$GP" >> "$S/pids.txt"
trap 'kill -TERM -- -"$GP" 2>/dev/null; exit 143' TERM INT
wait "$GP"; rc=$?
kill -- -"$GP" 2>/dev/null
echo "reclaim $NAME rc=$rc $(date -Is)" | tee -a "$LOG"
tail -n 4 "$LOG" | cut -c1-400
exit $rc
