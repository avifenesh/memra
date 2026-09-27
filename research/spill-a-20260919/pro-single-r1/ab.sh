#!/usr/bin/env bash
# R1's paired cell (DAY62.md section 8, (b) to (d)) in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash ab.sh @COLLECTOR_LOCK_FD@ <cell> <modeA> <modeB> <prefix_mb>`.
# Two binaries (base, r1) by three stall_cell.py modes: o1 = base-A r1-A base-B r1-B base-C r1-C x5, o2 the reverse x5 (60 boots),
# each boot one binary and one mode at --n 5,
# door ON, the PRO environment of the S sittings (MEMRA_KV_HOST_MB=8192, MEMRA_SERVE_SPEC=0), the prefix cache at
# <prefix_mb>, readiness bounded to 480 s; each boot's start temperature and SM clock in BOOT.txt; 250 ms telemetry;
# the layout day54-reading.py reads (<cell>/ab/o{1,2}/bNN-<mode>). A compute app at the hold's start: bounded
# 15 x 60 s, then NOT RUN. Executed-not-qualified.
set -uo pipefail
fd=$1; CELL=$2; MA=$3; MB=$4; MC=$5; PX=$6
R=/root/spill-receipts/a-r1
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
O=$R/$CELL
mkdir -p "$O/ab"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$O/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL") cell=$CELL modes=$MA,$MB,$MC prefix_mb=$PX"
sha256sum "$R/bins/base/memra-server" "$R/bins/r1/memra-server" > "$O/binaries.sha256"
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
boot() { # $1 log $2 bin
  memra_port_guard r1-ab "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 "MEMRA_PREFIX_CACHE_MB=$PX" MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 "$2" > "$1" 2>&1 &
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
  seq1="base:$MA r1:$MA base:$MB r1:$MB base:$MC r1:$MC"; [ $order = o2 ] && seq1="r1:$MC base:$MC r1:$MB base:$MB r1:$MA base:$MA"
  arms="$seq1 $seq1 $seq1 $seq1 $seq1"
  for am in $arms; do
    arm=${am%%:*}; mode=${am#*:}; BIN=$R/bins/$arm/memra-server
    i=$((i+1)); D=$O/ab/$order/$(printf 'b%02d-%s-%s' "$i" "$arm" "$mode"); mkdir -p "$D"
    echo "arm=$arm mode=$mode order=$order cell=$CELL bin=$(sha256sum "$BIN" | cut -c1-16)" > "$D/BOOT.txt"
    echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
    if ! boot "$D/server.log" "$BIN"; then
      log "$CELL $order $arm $mode boot NOT READY within 480 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
    fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode "$mode" --n 5 --server-log "$D/server.log" \
      --out "$D/$mode" --tag "r1-$CELL-$arm-$mode" > "$D/$mode.log" 2>&1
    log "$CELL $order $arm $mode rc=$? $(grep -h 'STALL rule' "$D/$mode.log" | cut -c1-120)"
    stop
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/$mode/receipt.json"; } \
      >> "$O/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/compute-apps.after.csv" 2>&1
log "$CELL done"
