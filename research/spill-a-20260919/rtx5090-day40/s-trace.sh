#!/usr/bin/env bash
# WP-A day 40 section 6 on the local RTX 5090 Laptop GPU, fixed before it runs: ONE bounded hold of /tmp/memra-5090.lock
# (60 x 120 s; then no compute app and >= 20000 MiB free, 15 x 60 s); one g4 and one s boot under Nsight Systems
# (--trace=cuda,osrt, no sampling), each stall_cell.py --mode demote --n 4 (8 demote runs), the day-35 5090 environment;
# each report exported to sqlite under /tmp/wt-a-d40trace and read by day40-trace-reading.py; the reports stay out of the
# tree (their sha256 banked).
# usage: s-trace.sh <out_root> <model.gguf> <bin_g4> <bin_s>
set -uo pipefail
ROOT=$1; MODEL=$2; BG4=$3; BS=$4
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
mkdir -p "$ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BG4" "$BS" > "$ROOT/binaries.sha256"
exec 9>/tmp/memra-5090.lock
held=0
for attempt in $(seq 1 60); do
    if flock -w 120 9; then held=1; break; fi
    log "hold attempt $attempt: the lock stayed busy for 120 s"
done
[ "$held" = 1 ] || { log "NOT RUN: the lock never came free"; exit 2; }
log "hold taken (fd 9)"
idle=0
for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then idle=1; break; fi
    log "idle wait $attempt under the hold: free=${free_mib}MiB apps=[${apps//$'\n'/; }]"
    sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; flock -u 9; exit 2; }
PORT=${MEMRA_GATE_PORT:-18147}
. tools/port-guard.sh
for arm in g4 s; do
    D=$ROOT/nsys-$arm; mkdir -p "$D"
    T=/tmp/wt-a-d40trace/nsys-$arm; mkdir -p "$T"   # the reports stay outside the tree (their sha256 banked)
    bin=$BG4; [ $arm = s ] && bin=$BS
    memra_port_guard day40-trace "$PORT" MEMRA_GATE_PORT || { log "port"; continue; }
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192 \
        MEMRA_KV_HOST_CONTRACTS=1 nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
        --output "$T/trace" "$bin" > "$D/server.log" 2>&1 &
    NSYS_PID=$!
    ready=0
    for _ in $(seq 1 90); do curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }; sleep 2; done
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
    python3 research/spill-a-20260919/day40-trace-reading.py "$T/trace.sqlite" > "$D/reading.log" 2>&1
    log "trace $arm reading rc=$? $(tail -1 "$D/reading.log" | cut -c1-200)"
done
flock -u 9
log "hold released; DAY40-TRACE-DONE"
