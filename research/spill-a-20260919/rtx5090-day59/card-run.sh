#!/usr/bin/env bash
# WP-A day 59 (DAY59.md), the fanout attribution on the local RTX 5090 Laptop GPU, fixed before it runs. ONE bounded hold
# of /tmp/memra-5090.lock (60 x 120 s behind the other lanes; then no compute app and >= 20000 MiB free, bounded 15 x 60
# s): stall_cell.py fanout against prime-short at 72 words, o1 = fanout prime-short x5, o2 = prime-short fanout x5, door
# ON, MEMRA_MAX_SESSIONS=8, MEMRA_PREFIX_CACHE_MB=256, MEMRA_KV_HOST_MB=8192, MEMRA_SERVE_SPEC=0, readiness bounded to
# 480 s; each boot's start temperature and SM clock; 250 ms telemetry; day54-reading.py and day59-reading.py.
# Executed-not-qualified. usage: card-run.sh <out_root> <model.gguf> <memra-server>
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
mkdir -p "$ROOT/short/ab"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BIN" > "$ROOT/binary.sha256"
git rev-parse HEAD > "$ROOT/tree.sha"
log "model sha256 (outside the hold): $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 60); do
  if flock -n 9; then held=1; break; fi
  log "lock busy, attempt $attempt of 60, waiting 120 s"; sleep 120
done
[ "$held" = 1 ] || { log "NOT RUN: the 5090 lock stayed busy"; exit 2; }
free_ok=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 | head -1)
  if [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] 2>/dev/null; then free_ok=1; break; fi
  log "card not idle under the hold (apps=[${apps//$'\n'/; }] free=${free} MiB), attempt $attempt of 15"; sleep 60
done
[ "$free_ok" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
log "hold taken"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$ROOT/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18159}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 log
  memra_port_guard d59-5090 "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=8 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 "$BIN" > "$1" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
stop() {
  [[ -n $SERVER_PID ]] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
finish() { stop; kill "$SAMPLER" 2>/dev/null || true; }
trap finish EXIT
i=0
for order in o1 o2; do
  arms="fanout prime-short fanout prime-short fanout prime-short fanout prime-short fanout prime-short"
  [ $order = o2 ] && arms="prime-short fanout prime-short fanout prime-short fanout prime-short fanout prime-short fanout"
  for mode in $arms; do
    i=$((i+1)); D=$ROOT/short/ab/$order/$(printf 'b%02d-%s' "$i" "$mode"); mkdir -p "$D"
    echo "mode=$mode order=$order bin=$(sha256sum "$BIN" | cut -c1-16)" > "$D/BOOT.txt"
    echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
    if ! boot "$D/server.log"; then log "$order $mode boot NOT READY"; echo boot-failed >> "$D/BOOT.txt"; stop; continue; fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode "$mode" --n 5 --server-log "$D/server.log" \
      --out "$D/$mode" --tag "d59-5090-$mode" > "$D/$mode.log" 2>&1
    log "$order $mode rc=$? $(grep -h 'STALL rule' "$D/$mode.log" | cut -c1-110)"
    stop
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/$mode/receipt.json"; } \
      >> "$ROOT/short/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
flock -u 9
python3 research/spill-a-20260919/day59-reading.py "$ROOT" > "$ROOT/reading-day59.log" 2>&1
log "done; reading rc=$? $(tail -1 "$ROOT/reading-day59.log")"
