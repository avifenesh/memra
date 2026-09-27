#!/usr/bin/env bash
# Design S2's trace reading on the target card (DAY42 section 1, the reading without a clause) in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash trace.sh @COLLECTOR_LOCK_FD@`. One g4 and one s2 boot
# under Nsight Systems (--trace=cuda,osrt, no sampling), each stall_cell.py --mode demote --n 4 (8 demote runs), the
# PRO demote environment of ab-demote.sh; each report exported to sqlite and read by day42-trace-reading.py. The
# reports and exports stay under $R/trace-reports (mirrored by hash only). A missing nsys: NOT RUN, typed.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-s2
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
D0=$R/trace
mkdir -p "$D0" "$R/trace-reports"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$D0/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D0/run.log"; }
command -v nsys >/dev/null 2>&1 || { log "NOT RUN: nsys is not on this host"; exit 2; }
sha256sum "$R/bins/g4/memra-server" "$R/bins/s2/memra-server" > "$D0/binaries.sha256"
export TMPDIR=$R/trace-reports/tmp
mkdir -p "$TMPDIR"
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
for arm in g4 s2; do
  D=$D0/nsys-$arm; T=$R/trace-reports/nsys-$arm; mkdir -p "$D" "$T"
  bin=$R/bins/g4/memra-server; [ $arm = s2 ] && bin=$R/bins/s2/memra-server
  memra_port_guard s2-trace "$PORT" MEMRA_GATE_PORT || { log "port"; continue; }
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
    --output "$T/trace" "$bin" > "$D/server.log" 2>&1 &
  NSYS_PID=$!
  ready=0
  for _ in $(seq 1 240); do curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }; sleep 2; done
  if [ $ready = 1 ]; then
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 4 --server-log "$D/server.log" \
      --out "$D/demote" --tag "trace-$arm" > "$D/demote.log" 2>&1
    log "trace $arm stall rc=$?"
  else
    log "trace $arm NOT READY"
  fi
  kill -INT "$NSYS_PID" 2>/dev/null
  for _ in $(seq 1 180); do kill -0 "$NSYS_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$NSYS_PID" 2>/dev/null; wait "$NSYS_PID" 2>/dev/null
  nsys export --type sqlite --force-overwrite=true --output "$T/trace.sqlite" "$T/trace.nsys-rep" > "$D/export.log" 2>&1
  sha256sum "$T"/trace.nsys-rep "$T"/trace.sqlite > "$D/trace.sha256" 2>&1
  python3 research/spill-a-20260919/day42-trace-reading.py "$T/trace.sqlite" > "$D/reading.log" 2>&1
  log "trace $arm reading rc=$? $(tail -1 "$D/reading.log" | cut -c1-200)"
done
rm -rf "$TMPDIR"
log "trace done"
