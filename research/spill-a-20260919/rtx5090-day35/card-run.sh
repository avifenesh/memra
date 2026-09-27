#!/usr/bin/env bash
# WP-A day 35 on the local RTX 5090 Laptop GPU (DAY35.md section 1), the F decision cell, fixed before it runs.
# ONE bounded hold of /tmp/memra-5090.lock (60 x 120 s behind the other lanes, never inside another hold; then no compute
# app and >= 20000 MiB free, bounded 15 x 60 s). stall_cell.py --mode promote --n 5 byte for byte per boot; three arms:
# hk (the HK binary, door ON), fk (the FK binary, door ON), off (the FK binary, door OFF); o1 = hk fk off x5, o2 = off fk
# hk x5 (o1 reversed), 30 boots; each server's readiness bounded to 60 s (not ready: stopped, recorded, the next boot
# runs; the reader then reads the cell INCOMPLETE). 250 ms card telemetry across the hold. Executed-not-qualified.
# usage: card-run.sh <out_root> <model.gguf> <bin_hk> <bin_fk>   (both binaries named memra-server)
set -uo pipefail
ROOT=$1; MODEL=$2; BHK=$3; BFK=$4
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
mkdir -p "$ROOT/ab"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BHK" "$BFK" > "$ROOT/binaries.sha256"
sha256sum research/spill-a-20260919/stall_cell.py > "$ROOT/harness.sha256"
log "model sha256 (outside the hold): $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 60); do
    if flock -w 120 9; then held=1; break; fi
    log "hold attempt $attempt: the lock stayed busy for 120 s"
done
[ "$held" = 1 ] || { log "NOT RUN: the lock never came free"; exit 2; }
log "hold taken (fd 9) tree=$TREE"
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
PORT=${MEMRA_GATE_PORT:-18136}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log $3 door (on|off)
    memra_port_guard day35-ab "$PORT" MEMRA_GATE_PORT || return 1
    local door=""
    [ "$3" = on ] && door="MEMRA_KV_HOST_CONTRACTS=1"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192 \
        $door "$1" > "$2" 2>&1 &
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
i=0
for order in o1 o2; do
    arms="hk fk off hk fk off hk fk off hk fk off hk fk off"
    [ $order = o2 ] && arms="off fk hk off fk hk off fk hk off fk hk off fk hk"
    for arm in $arms; do
        i=$((i+1)); D=$ROOT/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
        bin=$BFK; door=on
        [ $arm = hk ] && bin=$BHK
        [ $arm = off ] && door=off
        echo "arm=$arm order=$order door=$door bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.before.csv" 2>&1
        if ! boot "$bin" "$D/server.log" "$door"; then
            log "ab $order $arm boot NOT READY within 60 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
        fi
        python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode promote --n 5 --server-log "$D/server.log" \
            --out "$D/promote" --tag "stall-promote-$arm" > "$D/promote.log" 2>&1
        log "ab $order $arm rc=$? $(grep -h 'STALL rule' "$D/promote.log" | cut -c1-120)"
        stop
        nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.after.csv" 2>&1
        { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/promote/receipt.json"; } \
            >> "$ROOT/ab/replays.log" 2>&1
    done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/compute-apps.after.csv" 2>&1
flock -u 9
log "hold released; DAY35-CARD-RUN-DONE"
