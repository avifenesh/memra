#!/usr/bin/env bash
# DAY39 section 5's target cell (design T, item 3, on this slower-CPU host class) in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash item3.sh @COLLECTOR_LOCK_FD@`.
# stall_cell.py --mode promote --n 5 (ten promote runs per boot, nine steady; DAY39 section 5a), four arms: hk (the tip plus the
# day-32 helper fill, door ON), ft (the tip, design T, door ON), f1 (the tip at one fill thread, door ON), off (the ft binary,
# door OFF); o1 = hk ft f1 off x5, o2 = off f1 ft hk x5, 40 boots; the PRO environment (DAY34's double-park boot); readiness
# bounded to 480 s; the day39-reading.py layout (item3/ab/o{1,2}/bNN-<arm>). 250 ms telemetry. A compute app at the
# hold's start: bounded 15 x 60 s, then NOT RUN. Executed-not-qualified.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-g3
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
mkdir -p "$R/item3/ab"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/item3/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/item3/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
sha256sum "$R/bins/hk/memra-server" "$R/bins/g3/memra-server" "$R/bins/f1/memra-server" > "$R/item3/binaries.sha256"
idle=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  [ -z "$apps" ] && { idle=1; break; }
  log "idle wait $attempt under the hold: apps=[${apps//$'\n'/; }]"; sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$R/item3/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log $3 door (on|off)
  memra_port_guard g3-item3 "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
    $3 "$1" > "$2" 2>&1 &
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
  arms="hk ft f1 off hk ft f1 off hk ft f1 off hk ft f1 off hk ft f1 off"
  [ $order = o2 ] && arms="off f1 ft hk off f1 ft hk off f1 ft hk off f1 ft hk off f1 ft hk"
  for arm in $arms; do
    i=$((i+1)); D=$R/item3/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
    bin=$R/bins/g3/memra-server; door=MEMRA_KV_HOST_CONTRACTS=1
    [ $arm = hk ] && bin=$R/bins/hk/memra-server
    [ $arm = f1 ] && bin=$R/bins/f1/memra-server
    [ $arm = off ] && door=""
    echo "arm=$arm order=$order door=${door:-off} bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.before.csv" 2>&1
    if ! boot "$bin" "$D/server.log" "$door"; then
      log "item3 $order $arm boot NOT READY within 480 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
    fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode promote --n 5 --server-log "$D/server.log" \
      --out "$D/promote" --tag "stall-promote-$arm" > "$D/promote.log" 2>&1
    log "item3 $order $arm rc=$? $(grep -h 'STALL rule' "$D/promote.log" | cut -c1-120)"
    stop
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.after.csv" 2>&1
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/promote/receipt.json"; } \
      >> "$R/item3/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/item3/compute-apps.after.csv" 2>&1
python3 research/spill-a-20260919/day39-reading.py "$R/item3" target > "$R/item3/reading-day39-target.log" 2>&1
log "item3 done; reading rc=$?"
