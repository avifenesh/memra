#!/usr/bin/env bash
# OWED item 15's long-entry hump reading (DAY43.md section 1, term (4)) in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash hump-long.sh @COLLECTOR_LOCK_FD@`. Four door-ON boots
# xg4 xg3 xg3 xg4, each stall_cell.py --mode demote-long --n 8 (16 long demotes), the environment of ab-long.sh, each
# boot's start temperature and SM clock in BOOT.txt, then item15-hump-reading.py and item15-reading.py.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-i15
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
CACHE_MB=${MEMRA_ITEM15_CACHE_MB:-448}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
O=$R/hump
mkdir -p "$O"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$O/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL") cache_mb=$CACHE_MB"
sha256sum "$R/bins/g4/memra-server" "$R/bins/g3/memra-server" > "$O/binaries.sha256"
idle=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  [ -z "$apps" ] && { idle=1; break; }
  log "idle wait $attempt under the hold: apps=[${apps//$'\n'/; }]"; sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$O/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
  memra_port_guard i15-hump "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 "$1" > "$2" 2>&1 &
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
for arm in xg4 xg3 xg3 xg4; do
  i=$((i+1)); D=$O/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
  bin=$R/bins/${arm#x}/memra-server
  echo "arm=$arm bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
  echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
  if ! boot "$bin" "$D/server.log"; then log "hump $arm boot NOT READY"; echo boot-failed >> "$D/BOOT.txt"; stop; continue; fi
  python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote-long --n 8 --server-log "$D/server.log" \
    --out "$D/demote-long" --tag "i15-hump-$arm" > "$D/demote-long.log" 2>&1
  log "hump $arm rc=$? $(grep -h 'STALL rule' "$D/demote-long.log" | cut -c1-100)"
  stop
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
python3 research/spill-a-20260919/item15-hump-reading.py "$O" > "$O/reading-hump.log" 2>&1
log "hump done; reading rc=$?"
python3 research/spill-a-20260919/item15-reading.py "$R" > "$R/reading-item15.log" 2>&1
log "item15 reading rc=$? $(tail -1 "$R/reading-item15.log")"
