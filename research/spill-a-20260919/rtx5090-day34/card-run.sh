#!/usr/bin/env bash
# WP-A day 34 on the local RTX 5090 Laptop GPU (DAY34.md section 2), ONE bounded hold of /tmp/memra-5090.lock (60 x 120 s
# behind the other lanes, never inside another hold; then no compute app and >= 20000 MiB free, bounded 15 x 60 s):
#  1. the unit cells from prebuilt test binaries: the door's GPU cells (option_b_, option_c_) and the engine's H2D cells
#     (h2d_: the span, filled, deferred-checksum and owner-hold cells), serial;
#  2. the pre-registered A/B: stall_cell.py --mode promote --n 5 (byte for byte), the day-33 binary (with the timeline
#     fields) against the day-34 binary, o1 = d33 d34 x5, o2 = d34 d33 x5, the day-33 stall-pair boot environment, each
#     server's readiness bounded to 60 s (not ready: stopped, recorded, the next boot runs);
#  3. the gates with the day-34 binary, each through --external-lock 9: identity x4, failure ON (acceptance (a)'s cell),
#     fault default and plain, hit OFF and ON.
# Executed-not-qualified. usage: card-run.sh <out_root> <model.gguf> <bin_d33> <bin_d34> <server_tests> <engine_tests>
set -uo pipefail
ROOT=$1; MODEL=$2; B33=$3; B34=$4; ST=$5; ET=$6
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
mkdir -p "$ROOT"
sha256sum "$B33" "$B34" "$ST" "$ET" > "$ROOT/binaries.sha256"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
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
# 1. unit cells
mkdir -p "$ROOT/unit"
(cd crates/memra-server && "$ST" --ignored --test-threads=1 option_b_ option_c_ > "$ROOT/unit/server-door-cells.log" 2>&1); log "unit server rc=$? $(grep -h '^test result' "$ROOT/unit/server-door-cells.log")"
(cd crates/memra-engine && "$ET" --ignored --test-threads=1 --nocapture h2d_ d2h_span > "$ROOT/unit/engine-span-cells.log" 2>&1); log "unit engine rc=$? $(grep -h '^test result' "$ROOT/unit/engine-span-cells.log")"
# 2. the A/B
PORT=${MEMRA_GATE_PORT:-18135}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
    memra_port_guard day34-ab "$PORT" MEMRA_GATE_PORT || return 1
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
trap stop EXIT
i=0
for order in o1 o2; do
    arms="d33 d34 d33 d34 d33 d34 d33 d34 d33 d34"; [ $order = o2 ] && arms="d34 d33 d34 d33 d34 d33 d34 d33 d34 d33"
    for arm in $arms; do
        i=$((i+1)); D=$ROOT/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
        bin=$B33; [ $arm = d34 ] && bin=$B34
        echo "arm=$arm order=$order bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
        if ! boot "$bin" "$D/server.log"; then log "ab $order $arm boot NOT READY within 60 s; stopped"; stop; continue; fi
        python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode promote --n 5 --server-log "$D/server.log" \
            --out "$D/promote" --tag "stall-promote-$arm" > "$D/promote.log" 2>&1
        log "ab $order $arm rc=$? $(grep -h 'STALL rule' "$D/promote.log" | cut -c1-120)"
        stop
        python3 research/spill-a-20260919/stall_cell.py --replay "$D/promote/receipt.json" >> "$ROOT/ab/replays.log" 2>&1
    done
done
trap - EXIT
# 3. the gates
cell() { # $1 name $2 env-string $3.. command (--external-lock 9 inside; $OUT/ev is its evidence dir)
    local name=$1 envs=$2; shift 2
    local OUT=$ROOT/$name; mkdir -p "$OUT"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
    # shellcheck disable=SC2086
    env $envs "$@" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    echo "$rc" > "$OUT/gate.exit"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
    {
        echo "cell=$name"; echo "env=$envs"; echo "lock=$MEMRA_GPU_LOCK owner=hold-fd9"; echo "tree=$TREE"
        echo "binary_sha256=$(sha256sum "$B34" | cut -d' ' -f1)"; echo "model=$(basename "$MODEL")"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    log "gate $name rc=$rc $(grep -hE 'GATE: ' "$OUT/gate.log" | tail -1)"
}
C=MEMRA_HOSTGATE_CACHE_MB=64
cell identity-default-off "$C" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$B34"
cell identity-default-on "$C MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$B34"
cell identity-plain-off "$C MEMRA_SERVE_SPEC=0" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$B34"
cell identity-plain-on "$C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$B34"
cell failure-on "$C MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-failure-gate.sh --external-lock 9 "$MODEL" "$B34"
cell fault-default "$C" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$B34"
cell fault-plain "$C MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$B34"
cell hit-off "" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$B34"
cell hit-on "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$B34"
flock -u 9
log "hold released; DAY34-CARD-RUN-DONE"
