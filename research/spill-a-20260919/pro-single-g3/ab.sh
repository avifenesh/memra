#!/usr/bin/env bash
# The G''' target-card A/B (DAY38 section 14: section 3's cell whole, base against G''') in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash ab.sh @COLLECTOR_LOCK_FD@`.
# stall_cell.py --mode demote --n 5, base (80039a8de) against G''' (the tip), o1 = base g x5, o2 = g base x5, door ON, the PRO
# demote environment (DAY36 section 3's), readiness bounded to 480 s; the day38-reading.py layout (ab/o{1,2}/bNN-{base,g}).
# 250 ms telemetry. A compute app at the hold's start: bounded 15 x 60 s, then NOT RUN. Executed-not-qualified.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-g3
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
mkdir -p "$R/g/ab"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/g/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/g/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
sha256sum "$R/bins/base/memra-server" "$R/bins/g3/memra-server" > "$R/g/binaries.sha256"
idle=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  [ -z "$apps" ] && { idle=1; break; }
  log "idle wait $attempt under the hold: apps=[${apps//$'\n'/; }]"; sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$R/g/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
  memra_port_guard g3-ab "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
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
for order in o1 o2; do
  arms="base g base g base g base g base g"; [ $order = o2 ] && arms="g base g base g base g base g base"
  for arm in $arms; do
    i=$((i+1)); D=$R/g/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
    bin=$R/bins/base/memra-server; [ $arm = g ] && bin=$R/bins/g3/memra-server
    echo "arm=$arm order=$order bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.before.csv" 2>&1
    if ! boot "$bin" "$D/server.log"; then
      log "ab $order $arm boot NOT READY within 480 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
    fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 5 --server-log "$D/server.log" \
      --out "$D/demote" --tag "stall-demote-$arm" > "$D/demote.log" 2>&1
    log "ab $order $arm rc=$? $(grep -h 'STALL rule' "$D/demote.log" | cut -c1-120)"
    stop
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.after.csv" 2>&1
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/demote/receipt.json"; } \
      >> "$R/g/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/g/compute-apps.after.csv" 2>&1
python3 research/spill-a-20260919/day38-reading.py "$R/g" > "$R/g/reading-day38-target.log" 2>&1
log "ab done; reading rc=$?"
