#!/usr/bin/env bash
# Laptop, inside a gpulock lease on the RTX 5090 (memra#1255): one boot of the normal
# concurrency cell (closed-loop clients at c = 1, 2, 4, 8, streamed, greedy chat) on
# Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter attached. usage: cell-5090.sh ARM
# The power limit is recorded, never changed.
set -u
ARM=$1
S=$HOME/.cache/memra-1255
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
TAG=$ARM-$(date +%H%M%S)
OUT=$S/run/5090/${CELLS_SUB:-cells}/$TAG
mkdir -p "$OUT"
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  uptime; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader -l 1 > "$OUT/telemetry.csv" 2>/dev/null &
TP=$!
export CELL_MODEL=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_PORT=18174
export PLAN=${PLAN:-1,2,4,8} REQS=${REQS:-3} MAXTOK=${MAXTOK:-160} STREAM=1 TAG
export EXTRA_ENV="MEMRA_DSPARK_SPEC=1;MEMRA_DSPARK_DRAFT=${CELL_DRAFT:-$HOME/models/q38-gguf/Qwen3.8-27B-DFlash2-memra}${CELL_EXTRA:+;$CELL_EXTRA}"
setsid python3 "$S/harness/normal_cell.py" "$OUT" "$S/bin/memra-server-$ARM-120a" "$ARM" > "$OUT/cell.log" 2>&1 &
CP=$!
echo "$CP" > "$OUT/cell.pgid"
echo "$CP" >> "$S/pids.txt"
trap 'kill -TERM -- -"$CP" 2>/dev/null; kill "$TP" 2>/dev/null; exit 143' TERM INT
wait "$CP"; rc=$?
kill "$TP" 2>/dev/null
kill -- -"$CP" 2>/dev/null
echo "cell $TAG rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
grep -c . "$OUT/normal.jsonl" 2>/dev/null
exit $rc
