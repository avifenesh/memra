#!/usr/bin/env bash
# DAY38 section 13j's bisection in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash diag8.sh @COLLECTOR_LOCK_FD@`.
# Six door-ON boots, x1 x27 x2 x2 x27 x1 (x1 = the tip binary; x2 = bins/x2; x27 = the diag8 patch: the D2D classes,
# digest kernels and copies, on the receipt stream beside the D2H receipt, the copy stream kernel-free), each stall_cell.py
# --mode demote --n 8 (16 demote runs), the PRO demote environment of ab.sh, then the hump reader.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day38
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
D0=$R/diag8
mkdir -p "$D0"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$D0/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D0/run.log"; }
sha256sum "$R"/bins/{tip,x2,x27}/memra-server > "$D0/binaries.sha256"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$D0/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
  memra_port_guard day38-diag8 "$PORT" MEMRA_GATE_PORT || return 1
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
for arm in x1 x27 x2 x2 x27 x1; do
  i=$((i+1)); D=$D0/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
  bin=$R/bins/$arm/memra-server; [ $arm = x1 ] && bin=$R/bins/tip/memra-server
  echo "arm=$arm bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
  if ! boot "$bin" "$D/server.log"; then log "diag $arm boot NOT READY"; echo boot-failed >> "$D/BOOT.txt"; stop; continue; fi
  python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 8 --server-log "$D/server.log" \
    --out "$D/demote" --tag "diag8-$arm" > "$D/demote.log" 2>&1
  log "diag $arm rc=$? $(grep -h 'STALL rule' "$D/demote.log" | cut -c1-100)"
  stop
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
python3 research/spill-a-20260919/day38-hump-reading.py "$D0" > "$D0/reading-hump.log" 2>&1
log "diag done; reading rc=$?"
