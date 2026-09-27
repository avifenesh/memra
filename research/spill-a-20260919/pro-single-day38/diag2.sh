#!/usr/bin/env bash
# DAY38 section 13 in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash diag2.sh @COLLECTOR_LOCK_FD@`.
# 1. Section 12's registered follow-up for `not reproduced`: the section-3 A/B (ab.sh's cell) re-run once, unchanged
#    binaries, into g-rerun/.
# 2. One X1 and one X2 boot under Nsight Systems (CUDA and OS runtime trace, no sampling), each stall_cell.py --mode demote
#    --n 8 (16 demote runs), exported to sqlite and read by day38-nsys-reading.py; the reports stay on the box, their
#    sha256 and the reader's output are mirrored.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day38
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
D0=$R/diag2
mkdir -p "$D0"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$D0/LOCK.json"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D0/run.log"; }
# 1. the A/B re-run: ab.sh writes under $R/g; point it at g-rerun through a copy with only the directory changed.
sed -e 's|mkdir -p "$R/g/ab"|mkdir -p "$R/g-rerun/ab"|' -e 's|\$R/g/|$R/g-rerun/|g' -e 's|"\$R/g"|"$R/g-rerun"|g' \
  research/spill-a-20260919/pro-single-day38/ab.sh > "$D0/ab-rerun.sh"
diff research/spill-a-20260919/pro-single-day38/ab.sh "$D0/ab-rerun.sh" > "$D0/ab-rerun.diff"
bash "$D0/ab-rerun.sh" "$fd"
log "ab rerun rc=$?"
# 2. the traces
PORT=${MEMRA_GATE_PORT:-18137}
. tools/port-guard.sh
NSYS_PID=""
for arm in x1 x2; do
  D=$D0/nsys-$arm; mkdir -p "$D"
  bin=$R/bins/tip/memra-server; [ $arm = x2 ] && bin=$R/bins/x2/memra-server
  memra_port_guard day38-nsys "$PORT" MEMRA_GATE_PORT || { log "port"; continue; }
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
    --output "$D/trace" "$bin" > "$D/server.log" 2>&1 &
  NSYS_PID=$!
  ready=0
  for _ in $(seq 1 300); do curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }; sleep 2; done
  if [ $ready = 1 ]; then
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 8 --server-log "$D/server.log" \
      --out "$D/demote" --tag "nsys-$arm" > "$D/demote.log" 2>&1
    log "nsys $arm stall rc=$?"
  else
    log "nsys $arm NOT READY"
  fi
  # stop the traced server: SIGINT to nsys ends the collection and the app; bounded wait for the report
  kill -INT "$NSYS_PID" 2>/dev/null
  for _ in $(seq 1 180); do kill -0 "$NSYS_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$NSYS_PID" 2>/dev/null; wait "$NSYS_PID" 2>/dev/null
  ls -la "$D"/trace.* > "$D/report.ls" 2>&1
  nsys export --type sqlite --force-overwrite=true --output "$D/trace.sqlite" "$D/trace.nsys-rep" > "$D/export.log" 2>&1
  sha256sum "$D"/trace.nsys-rep "$D"/trace.sqlite > "$D/trace.sha256" 2>&1
  python3 research/spill-a-20260919/day38-nsys-reading.py "$D/trace.sqlite" "$D/demote/receipt.json" > "$D/reading.log" 2>&1
  log "nsys $arm reading rc=$?"
done
log "diag2 done"
