#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one boot of memra-server as the timed cell of
# research/spec-draft-accept-leftovers-20261110 boots it (context 8,192, two sessions, every
# switch at the device default), the six requests of req_driver.py, then stop.
#   MODE nsys   the server under Nsight Systems (CUDA trace with graph nodes, no CPU sampling,
#               no context-switch trace), then the sqlite export
#   MODE phase  no profiler; MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 (the engine's own per-burst
#               wall split and accept histogram, printed to the server log)
#   MODE plain  no profiler, no door
# usage: prof_cell.sh MODE NAME
set -u
MODE=$1; NAME=$2
D=$HOME/.cache/memra-1186
OUT=$D/run/$NAME; mkdir -p "$OUT"
BIN=$D/bin/memra-server
PORT=18196
NSYS=/usr/local/cuda/bin/nsys
MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ENVV="MEMRA_COMPAT=openai MEMRA_MODELS=q=$MODEL MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=2"
[ "$MODE" = phase ] && ENVV="$ENVV MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1"
# MODE setup: no profiler; MEMRA_SPEC_SETUP_TRACE=1 (the engine's per-burst wall split: init feed,
# capture, fill, rounds, tail; the door adds one stream synchronize at each burst's tail)
[ "$MODE" = setup ] && ENVV="$ENVV MEMRA_SPEC_SETUP_TRACE=1"
# MODE burstN: no profiler; MEMRA_SPEC_BURST=N, the documented burst length parameter (default 32
# tokens): the probe of what a burst boundary costs each class
case "$MODE" in burst[0-9]*) ENVV="$ENVV MEMRA_SPEC_BURST=${MODE#burst}";; esac
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
{ echo "$(date -Is) start MODE=$MODE"; sha256sum "$BIN"; echo "env: $ENVV" | sed "s#$HOME#~#g";
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; uptime; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 250 > "$OUT/telemetry.csv" 2>/dev/null &
smi=$!
if [ "$MODE" = nsys ]; then
  "$NSYS" profile -t cuda,nvtx -f true --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    --cuda-flush-interval=1000 -o "$OUT/prof" /usr/bin/env $ENVV "$BIN" > "$OUT/server.log" 2>&1 &
else
  /usr/bin/env $ENVV "$BIN" > "$OUT/server.log" 2>&1 &
fi
pid=$!
echo $pid > "$OUT/server.pid"
if ! waitfor --until "curl -sf -o /dev/null http://127.0.0.1:$PORT/v1/models && exit 0 || exit 1" \
     --fail-if "! kill -0 $pid" --every 3s --timeout 12m > "$OUT/run.log" 2>&1; then
  echo "$(date -Is) server never became ready" | tee -a "$OUT/run.log"
  kill -INT $pid; wait $pid; kill $smi; echo "CELL-FAIL $NAME"; exit 3
fi
echo "$(date -Is) ready" >> "$OUT/run.log"
sleep 3
python3 "$D/box/req_driver.py" "$OUT/reqs.jsonl" "$PORT" 3 400 >> "$OUT/run.log" 2>&1
echo "$(date -Is) driver rc=$?" >> "$OUT/run.log"
if [ "$MODE" = nsys ]; then
  sleep 20   # the trace keeps the last calls: buffers flush every second
  kill -INT $pid   # nsys ends the session and stops the server
  waitfor --until "test -f $OUT/prof.nsys-rep && ! kill -0 $pid 2>/dev/null && exit 0 || exit 1" --every 5s --timeout 15m >> "$OUT/run.log" 2>&1
  wait $pid
  ls -la "$OUT"/prof.* >> "$OUT/run.log" 2>&1
  "$NSYS" export --type sqlite --force-overwrite=true -o "$OUT/prof.sqlite" "$OUT/prof.nsys-rep" >> "$OUT/run.log" 2>&1
  ls -la "$OUT"/prof.* >> "$OUT/run.log" 2>&1
else
  kill -TERM $pid
  waitfor --gone $pid --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL $pid
  wait $pid
fi
kill $smi 2>/dev/null
echo "$(date -Is) server exit" >> "$OUT/guard.log"
cut -c1-170 "$OUT/run.log" | tail -n 16
echo "CELL-DONE $NAME"
