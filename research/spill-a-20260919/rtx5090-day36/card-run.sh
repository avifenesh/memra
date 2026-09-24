#!/usr/bin/env bash
# WP-A day 36 on the local RTX 5090 Laptop GPU (DAY36.md section 1), the restore price reading, fixed before it runs. ONE
# bounded hold of /tmp/memra-5090.lock (60 x 120 s behind the other lanes, never inside another hold; then no compute app
# and >= 20000 MiB free, bounded 15 x 60 s). The day-26 restore arm (pro-single-day26/restore-arm.sh) byte for byte in
# its boot and harness: one door-ON boot (MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1
# MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4), stall_cell.py --mode restore --n 50, the receipt replayed; the
# server's readiness bounded to 60 s. 250 ms card telemetry. Executed-not-qualified.
# usage: card-run.sh <out_root> <model.gguf> <bin>   (the binary named memra-server)
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
EV=$ROOT/restore-arm/ev
mkdir -p "$EV"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BIN" > "$ROOT/binary.sha256"
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
{
    echo "tree=$TREE"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=one ON boot, --mode restore --n 50 (100 timed restore runs)"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
PORT=${MEMRA_GATE_PORT:-18138}
. tools/port-guard.sh
SERVER_PID=""
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
extra="MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1"
echo "arm=on env=$extra" > "$EV/BOOT.txt"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$EV/card.before.csv" 2>&1
ready=0
if memra_port_guard day36-restore "$PORT" MEMRA_GATE_PORT; then
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 $extra "$BIN" > "$EV/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 30); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
fi
if [ "$ready" = 1 ]; then
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode restore --n 50 --server-log "$EV/server.log" \
        --out "$EV/restore" --tag "stall-restore-on" > "$EV/restore.log" 2>&1
    log "restore arm rc=$? $(grep -h 'STALL rule' "$EV/restore.log" | cut -c1-120)"
else
    log "restore arm boot NOT READY within 60 s; stopped"
fi
stop
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$EV/card.after.csv" 2>&1
{ echo "== restore-arm/restore arm=on"; python3 research/spill-a-20260919/stall_cell.py --replay "$EV/restore/receipt.json"; } \
    > "$EV/replays.log" 2>&1
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/compute-apps.after.csv" 2>&1
flock -u 9
log "hold released; DAY36-CARD-RUN-DONE"
