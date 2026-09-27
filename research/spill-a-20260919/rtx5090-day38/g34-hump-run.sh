#!/usr/bin/env bash
# WP-A day 38 section 20 on the local RTX 5090 Laptop GPU, fixed before it runs: ONE bounded hold of /tmp/memra-5090.lock
# (60 x 120 s behind the other lanes, never inside another hold; then no compute app and >= 20000 MiB free, bounded 15 x
# 60 s); eight door-ON boots xbase xg4 xg3 xgpp xgpp xg3 xg4 xbase (xbase 80039a8de, G4 the tip, G''' 9ab5c1265, G''
# 358749c9f), each stall_cell.py --mode demote --n 8 (16 demote runs), the day-35 5090 environment, readiness bounded to
# 60 s; 250 ms card telemetry; read by day38-hump-reading.py. Executed-not-qualified.
# usage: g34-hump-run.sh <out_root> <model.gguf> <bin_base> <bin_g4> <bin_g3> <bin_gpp>
set -uo pipefail
ROOT=$1; MODEL=$2; BBASE=$3; BG4=$4; BG3=$5; BGPP=$6
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
mkdir -p "$ROOT/hump"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BBASE" "$BG4" "$BG3" "$BGPP" > "$ROOT/binaries.sha256"
sha256sum research/spill-a-20260919/stall_cell.py > "$ROOT/harness.sha256"
log "model sha256 (outside the hold): $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
exec 9>"$MEMRA_GPU_LOCK"
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
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
    --format=csv -lms 250 > "$ROOT/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18143}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
    memra_port_guard day38-g34 "$PORT" MEMRA_GATE_PORT || return 1
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192 \
        MEMRA_KV_HOST_CONTRACTS=1 "$1" > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 30); do
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
j=0
for arm in xbase xg4 xg3 xgpp xgpp xg3 xg4 xbase; do
    j=$((j+1)); D=$ROOT/hump/$(printf 'b%02d-%s' "$j" "$arm"); mkdir -p "$D"
    case $arm in xbase) bin=$BBASE;; xg4) bin=$BG4;; xg3) bin=$BG3;; xgpp) bin=$BGPP;; esac
    echo "arm=$arm bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
    log "hump $arm boot at $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
    if ! boot "$bin" "$D/server.log"; then
        log "hump $arm boot NOT READY within 60 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
    fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 8 --server-log "$D/server.log" \
        --out "$D/demote" --tag "hump-$arm" > "$D/demote.log" 2>&1
    log "hump $arm rc=$? $(grep -h 'STALL rule' "$D/demote.log" | cut -c1-120)"
    stop
    { echo "== hump/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/demote/receipt.json"; } \
        >> "$ROOT/hump/replays.log" 2>&1
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
python3 research/spill-a-20260919/day38-hump-reading.py "$ROOT/hump" > "$ROOT/hump/reading-hump.log" 2>&1
flock -u 9
log "hold released; DAY38-G34-HUMP-DONE $(grep -h 'arm=' "$ROOT/hump/reading-hump.log" | tr '\n' ' ')"
