#!/usr/bin/env bash
# DAY49 section 1's cell in ONE collector hold: `tier-battery.py --rig pro-single --external-lock --execute bash cell.sh
# @COLLECTOR_LOCK_FD@`. The attribution binary, door ON, the PRO environment: nofree (MEMRA_KV_HOST_MB=8192) and free
# (MEMRA_KV_HOST_MB=480) interleaved five boots each, stall_cell.py --mode demote --n 5; then long x3
# (--mode demote-long --n 5, MEMRA_KV_HOST_MB=8192, MEMRA_PREFIX_CACHE_MB=448); each boot's start temperature and SM
# clock; then promote x3 (DAY50, --mode promote --n 5); 250 ms telemetry; day49-reading.py and day50-reading.py. A compute app at the hold's start: bounded 15 x 60 s, then NOT RUN.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-d49
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/tip/memra-server
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
mkdir -p "$R/cell"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/cell/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/cell/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
sha256sum "$BIN" > "$R/cell/binary.sha256"
idle=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  [ -z "$apps" ] && { idle=1; break; }
  log "idle wait $attempt under the hold: apps=[${apps//$'\n'/; }]"; sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$R/cell/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 host MB $2 prefix MB $3 log
  memra_port_guard d49-cell "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 "MEMRA_PREFIX_CACHE_MB=$2" "MEMRA_KV_HOST_MB=$1" \
    MEMRA_KV_HOST_CONTRACTS=1 "$BIN" > "$3" 2>&1 &
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
declare -A N=([nofree]=0 [free]=0 [long]=0 [promote]=0)
run() { # $1 arm
  local arm=$1 host=8192 px=256 mode=demote
  [ "$arm" = free ] && host=480
  [ "$arm" = long ] && { px=448; mode=demote-long; }
  [ "$arm" = promote ] && mode=promote
  N[$arm]=$((N[$arm] + 1)); local D=$R/cell/$arm/$(printf 'b%02d' "${N[$arm]}"); mkdir -p "$D"
  echo "arm=$arm host_mb=$host prefix_mb=$px mode=$mode bin=$(sha256sum "$BIN" | cut -c1-16)" > "$D/BOOT.txt"
  echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
  if ! boot "$host" "$px" "$D/server.log"; then log "$arm boot NOT READY within 480 s"; echo boot-failed >> "$D/BOOT.txt"; stop; return; fi
  python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode "$mode" --n 5 --server-log "$D/server.log" \
    --out "$D/$mode" --tag "d49-$arm" > "$D/$mode.log" 2>&1
  log "$arm $(basename "$D") rc=$? $(grep -h 'STALL rule' "$D/$mode.log" | cut -c1-100)"
  stop
  { echo "== $arm/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/$mode/receipt.json"; } >> "$R/cell/replays.log" 2>&1
}
for _ in 1 2 3 4 5; do run nofree; run free; done
for _ in 1 2 3; do run long; done
# DAY50 (OWED item 9): the promote block on the same binary, a reading (day50-reading.py).
for _ in 1 2 3; do run promote; done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/cell/compute-apps.after.csv" 2>&1
python3 research/spill-a-20260919/day49-reading.py "$R/cell" > "$R/cell/reading-day49.log" 2>&1
python3 research/spill-a-20260919/day50-reading.py "$R"/cell/promote/b* > "$R/cell/reading-day50.log" 2>&1
log "cell done; reading rc=$? $(tail -1 "$R/cell/reading-day49.log")"
