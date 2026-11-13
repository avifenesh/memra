#!/usr/bin/env bash
# Rig (RTX 5090 Laptop GPU, under the caller's gpulock lease), memra#1221 part 2: one boot of the
# 29-turn agentic replay on the MTP route (box/agentic_cell.sh's workload, driver and request
# settings: ctx 32,768, 8 sessions, prefix cache 4,000 MB, no host tier, c = 2, thinking on,
# temperature 0, seed 1234, 768 tokens; MEMRA_TICK_TRACE=1), then the leak-check trim (both arms
# carry tools/trim_probe.py). The enforced power limit is recorded, never changed.
# usage: rtx_agentic.sh CAMPAIGN NN-ARM BIN_PATH   -> ~/.cache/memra-1221b/runs/CAMPAIGN/NN-ARM
set -u
D=$HOME/.cache/memra-1221b
CAMP=$1; NAME=$2; BIN=$3
CD=$D/runs/$CAMP; OUT=$CD/$NAME; mkdir -p "$OUT"
M=/data/ai-ml/models/q38-gguf
PORT=8191
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy; refusing" >> "$CD/campaign.log"; exit 4; fi
{ date -Is; echo "$NAME"; sha256sum "$BIN" "$D/box/ab-workload.json" | sed "s|$HOME|~|";
  nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,temperature.gpu --format=csv,noheader; uptime; } > "$OUT/guard.log"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 \
MEMRA_PREFIX_CACHE_MB=4000 MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
MEMRA_MTP_DRAFT="$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_TICK_TRACE=1 \
PROBE_TRIM_FILE="$OUT/trim-now" setsid python3 "$D/box/tsrun.py" "$BIN" > "$OUT/server.log" 2>&1 &
sp=$!
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu,memory.used \
  --format=csv,noheader -l 1 > "$OUT/telemetry.csv" 2>/dev/null &
tp=$!
rc=0
if waitfor --until "curl -sf -o /dev/null http://127.0.0.1:$PORT/readyz && exit 0 || exit 1" --fail-if "! kill -0 $sp" --every 5s --timeout 10m > "$OUT/run.log" 2>&1; then
  CLIENT_MAX_TIME=900 BENCH_BASE=http://127.0.0.1:$PORT BENCH_MODEL=q38 timeout 1500 \
    python3 "$D/box/agent_turns.py" "$D/box/ab-workload.json" "$OUT" "$NAME" --users 2 --think 1 \
    --temperature 0 --seed 1234 --max-tokens 768 >> "$OUT/cells.log" 2>&1 || rc=$?
  curl -s http://127.0.0.1:$PORT/metrics > "$OUT/metrics.json" 2>/dev/null
  touch "$OUT/trim-now"
  # The worker checks the file at a tick top; an idle worker waits for a command, so a
  # one-token request wakes it (the trim runs at the top of the tick that admits it).
  curl -s -m 120 -o /dev/null http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"q38","prompt":"wake","max_tokens":1,"temperature":0}'
  waitfor --grep '\[probe-trim\] after trim' "$OUT/server.log" --timeout 3m >> "$OUT/run.log" 2>&1 || rc=4
else
  echo "boot FAILED" >> "$OUT/run.log"; rc=3
fi
kill -TERM "$sp" 2>/dev/null
waitfor --gone "$sp" --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL -- -"$sp" 2>/dev/null
kill "$tp" 2>/dev/null
{ for pat in 'rewind refused below a lent prefix row' '[spec] rewind below a lent prefix row' '[prefix-loan] reconcile' '[prefix-loan] compact' \
             'insert (spec-boundary)' '[prefix-cache] hit' 'spec restore' 'out of memory' '[probe-trim]'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server.log")"
  done; grep -a -F '[probe-trim]' "$OUT/server.log" | cut -c1-240; } > "$OUT/summary.txt"
tail -n 1 "$OUT/cells.log" | cut -c1-400 | sed "s/^/$NAME /" >> "$CD/campaign.log"
gzip -f "$OUT/server.log" "$OUT/telemetry.csv"
[ $rc = 0 ] && echo "AGENTIC-CELL-DONE $NAME" >> "$CD/campaign.log"
exit $rc
