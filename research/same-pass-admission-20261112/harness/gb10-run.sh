#!/usr/bin/env bash
# Runs ON a GB10 host (memra#1255), inside a sparklock lease: one boot of KIND for ARM.
#   KIND cells:   the normal concurrency cell (closed-loop c = 1, 2, 4, 8, 6 requests per client,
#                 greedy chat, 160 tokens, streamed), Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter.
#   KIND agentic: the agentic replay at c = 2 in the 5090 cell's shape (DFlash2, FR-Spec trim,
#                 MEMRA_MTP_SKIP=1, ctx 32,768, 8 sessions, prefix cache 4,000 MB, T = 1 seed 1234).
# usage: gb10-run.sh KIND ARM      (binary bins/memra-server-ARM-121a)
set -u
KIND=$1; ARM=$2
R=$HOME/.cache/memra-1255
M=$HOME/models/memra-q38
G=$M/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
BIN=$R/bins/memra-server-$ARM-121a
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
TAG=$ARM-$(date +%H%M%S)
OUT=$R/run/gb10/$KIND/$TAG
mkdir -p "$OUT"
echo "$$" >> "$R/pids.txt"
{ date -Is; sha256sum "$BIN"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
  uptime; free -g | sed -n 2p; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv,noheader -lms 250 \
  > "$OUT/telemetry.csv" 2>/dev/null &
TP=$!
rc=0
if [ "$KIND" = cells ]; then
  export CELL_MODEL=$G CELL_PORT=18174 PLAN=1,2,4,8 REQS=6 MAXTOK=160 STREAM=1 TAG
  export EXTRA_ENV="MEMRA_DSPARK_SPEC=1;MEMRA_DSPARK_DRAFT=$DR"
  setsid python3 "$R/harness/normal_cell.py" "$OUT" "$BIN" "$ARM" > "$OUT/cell.log" 2>&1 &
  CP=$!
  trap 'kill -TERM -- -"$CP" 2>/dev/null; kill "$TP" 2>/dev/null; exit 143' TERM INT
  wait "$CP"; rc=$?
  kill -- -"$CP" 2>/dev/null
else
  W=$R/harness/ab-workload-5090.json
  setsid env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$G" MEMRA_ADDR=127.0.0.1:18175 \
    MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DR \
    MEMRA_FRSPEC_TRIM=$M/Qwen3.8-27B-NVFP4-MTP-GGUF/q38-ranks-sxc32768.gguf.txt MEMRA_PREFIX_STABLE_BOUNDARY=0 \
    "$BIN" > "$OUT/serve.log" 2>&1 &
  SP=$!
  trap 'kill -TERM -- -"$SP" 2>/dev/null; kill "$TP" 2>/dev/null; exit 143' TERM INT
  if waitfor --http http://127.0.0.1:18175/v1/models --timeout 10m --every 5s --fail-after 120 --quiet && kill -0 "$SP" 2>/dev/null; then
    CLIENT_MAX_TIME=1500 BENCH_BASE=http://127.0.0.1:18175 BENCH_MODEL=q38 timeout 2400 \
      python3 "$R/harness/agent_turns.py" "$W" "$OUT" "$TAG" --users 2 --think 1 --temperature 1.0 --seed 1234 \
      >> "$OUT/cells.log" 2>&1 || rc=$?
    curl -s http://127.0.0.1:18175/metrics > "$OUT/metrics.json" 2>/dev/null
  else
    echo "boot FAILED" | tee -a "$OUT/guard.log"; rc=3
  fi
  kill -TERM "$SP" 2>/dev/null; waitfor --gone "$SP" --timeout 90s --quiet || kill -KILL -- -"$SP" 2>/dev/null
  for pat in 'drafter refused' 'VRAM defer' 'capacity reject' 'out of memory' 'settle wait' 'settled:'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/serve.log")" >> "$OUT/guard.log"
  done
fi
kill "$TP" 2>/dev/null
echo "end $(date -Is) rc=$rc" | tee -a "$OUT/guard.log"
exit $rc
