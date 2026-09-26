#!/usr/bin/env bash
# Design P2's target-card A/B (DAY52.md section 1: base against p2; the chain cell also p) for ONE cell in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash ab.sh @COLLECTOR_LOCK_FD@ <cell>`, cell one of
#   demote   stall_cell.py --mode demote,       MEMRA_KV_HOST_MB=8192, MEMRA_PREFIX_CACHE_MB=256  ((b), (c))
#   free     stall_cell.py --mode demote,       MEMRA_KV_HOST_MB=480,  MEMRA_PREFIX_CACHE_MB=256  ((f))
#   promote  stall_cell.py --mode promote,      MEMRA_KV_HOST_MB=8192, MEMRA_PREFIX_CACHE_MB=256  ((d))
#   chain    stall_cell.py --mode promote-long, MEMRA_KV_HOST_MB=8192, MEMRA_PREFIX_CACHE_MB=448  ((g))
# --n 5, o1 = base p2 x5, o2 = p2 base x5 (chain: o1 = base p p2 x5, o2 = p2 p base x5), door ON, the PRO environment of the S sittings, readiness bounded to 480 s;
# each boot's start temperature and SM clock in BOOT.txt, its server VmRSS and VmHWM before the stop in rss.txt; 250 ms
# telemetry; the layout day52-reading.py reads (<cell>/ab/o{1,2}/bNN-{base,p2,p}). A compute app at the hold's start:
# bounded 15 x 60 s, then NOT RUN. Executed-not-qualified.
set -uo pipefail
fd=$1; CELL=$2
R=/root/spill-receipts/a-p2l2
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
case "$CELL" in
  demote) MODE=demote; HOST=8192; PX=256 ;;
  free) MODE=demote; HOST=480; PX=256 ;;
  promote) MODE=promote; HOST=8192; PX=256 ;;
  chain) MODE=promote-long; HOST=8192; PX=448 ;;
  *) echo "cell $CELL"; exit 2 ;;
esac
O=$R/$CELL
mkdir -p "$O/ab"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$O/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/run.log"; }
log "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL") cell=$CELL mode=$MODE host_mb=$HOST prefix_mb=$PX"
sha256sum "$R/bins/base/memra-server" "$R/bins/p2/memra-server" > "$O/binaries.sha256"
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
  memra_port_guard p2-ab "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 "MEMRA_PREFIX_CACHE_MB=$PX" "MEMRA_KV_HOST_MB=$HOST" \
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
  arms="base p2 base p2 base p2 base p2 base p2"; [ $order = o2 ] && arms="p2 base p2 base p2 base p2 base p2 base"
  for arm in $arms; do
    i=$((i+1)); D=$O/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
    bin=$R/bins/$arm/memra-server
    echo "arm=$arm order=$order cell=$CELL mode=$MODE bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
    echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
    if ! boot "$bin" "$D/server.log"; then
      log "$CELL $order $arm boot NOT READY within 480 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
    fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode "$MODE" --n 5 --server-log "$D/server.log" \
      --out "$D/$MODE" --tag "p2-$CELL-$arm" > "$D/$MODE.log" 2>&1
    log "$CELL $order $arm rc=$? $(grep -h 'STALL rule' "$D/$MODE.log" | cut -c1-120)"
    grep -E '^(VmRSS|VmHWM):' "/proc/$SERVER_PID/status" > "$D/rss.txt" 2>&1
    stop
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/$MODE/receipt.json"; } \
      >> "$O/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/compute-apps.after.csv" 2>&1
log "$CELL done"
