#!/usr/bin/env bash
# WP-A day 38 design G4 on the local RTX 5090 Laptop GPU (DAY38.md section 17), fixed before it runs. ONE bounded hold of
# /tmp/memra-5090.lock (60 x 120 s behind the other lanes, never inside another hold; then no compute app and >= 20000
# MiB free, bounded 15 x 60 s):
#  1. the unit cells from prebuilt test binaries of the G4 tree: the door's GPU cells (option_b_, option_c_), serial, and
#     the engine's native tier_transfer cells in parallel (each owns a pool context);
#  2. section 3's A/B: stall_cell.py --mode demote --n 5 (byte for byte), base (80039a8de) against G4 (the tip), o1 =
#     base g x5, o2 = g base x5, door ON, the day-35 5090 environment, readiness bounded to 60 s; 250 ms telemetry;
#  3. (f) the hump cell: four door-ON boots xgpp xg4 xg4 xgpp (G'' the pre-G''' tip as the control, G4 the tip), each
#     stall_cell.py --mode demote --n 8 (16 demote runs), read by day38-hump-reading.py;
#  4. the gates with the G4 binary, each through --external-lock 9: identity x4, failure OFF and ON, fault default and
#     plain (every cell, day 41's three included), hit OFF and ON (binary named memra-server).
# Executed-not-qualified. usage: g4-card-run.sh <out_root> <model.gguf> <bin_base> <bin_g4> <bin_gpp> <server_tests> <engine_tests>
set -uo pipefail
ROOT=$1; MODEL=$2; BBASE=$3; BM=$4; BGPP=$5; ST=$6; ET=$7
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
mkdir -p "$ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
sha256sum "$BBASE" "$BM" "$BGPP" "$ST" "$ET" > "$ROOT/binaries.sha256"
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
# 1. unit cells
mkdir -p "$ROOT/unit"
(cd crates/memra-server && "$ST" --ignored --test-threads=1 option_b_ option_c_ > "$ROOT/unit/server-door-cells.log" 2>&1); log "unit server rc=$? $(grep -h '^test result' "$ROOT/unit/server-door-cells.log")"
(cd crates/memra-engine && "$ET" --ignored --nocapture tier_transfer::tests:: > "$ROOT/unit/engine-native-cells.log" 2>&1); log "unit engine rc=$? $(grep -h '^test result' "$ROOT/unit/engine-native-cells.log")"
# 2. the A/B
PORT=${MEMRA_GATE_PORT:-18141}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 log
    memra_port_guard day38-g4 "$PORT" MEMRA_GATE_PORT || return 1
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
i=0
for order in o1 o2; do
    arms="base g base g base g base g base g"; [ $order = o2 ] && arms="g base g base g base g base g base"
    for arm in $arms; do
        i=$((i+1)); D=$ROOT/ab/$order/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
        bin=$BBASE; [ $arm = g ] && bin=$BM
        echo "arm=$arm order=$order bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.before.csv" 2>&1
        if ! boot "$bin" "$D/server.log"; then
            log "ab $order $arm boot NOT READY within 60 s; stopped"; echo "boot-failed" >> "$D/BOOT.txt"; stop; continue
        fi
        python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode demote --n 5 --server-log "$D/server.log" \
            --out "$D/demote" --tag "stall-demote-$arm" > "$D/demote.log" 2>&1
        log "ab $order $arm rc=$? $(grep -h 'STALL rule' "$D/demote.log" | cut -c1-120)"
        stop
        nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$D/card.after.csv" 2>&1
        { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/demote/receipt.json"; } \
            >> "$ROOT/ab/replays.log" 2>&1
    done
done
# 3. (f) the hump cell
mkdir -p "$ROOT/hump"
j=0
for arm in xgpp xg4 xg4 xgpp; do
    j=$((j+1)); D=$ROOT/hump/$(printf 'b%02d-%s' "$j" "$arm"); mkdir -p "$D"
    bin=$BM; [ $arm = xgpp ] && bin=$BGPP
    echo "arm=$arm bin=$(sha256sum "$bin" | cut -c1-16)" > "$D/BOOT.txt"
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
python3 research/spill-a-20260919/day38-hump-reading.py "$ROOT/hump" > "$ROOT/hump/reading-hump.log" 2>&1
log "hump reading $(grep -h 'arm=' "$ROOT/hump/reading-hump.log" | tr '\n' ' ')"
trap - EXIT
# 4. the gates
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
        echo "binary_sha256=$(sha256sum "$BM" | cut -d' ' -f1)"; echo "model=$(basename "$MODEL")"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    log "gate $name rc=$rc $(grep -hE 'GATE: ' "$OUT/gate.log" | tail -1)"
}
C=MEMRA_HOSTGATE_CACHE_MB=64
cell identity-default-off "$C" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BM"
cell identity-default-on "$C MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BM"
cell identity-plain-off "$C MEMRA_SERVE_SPEC=0" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BM"
cell identity-plain-on "$C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BM"
cell failure-off "$C" bash tools/kv-host-spill-failure-gate.sh --external-lock 9 "$MODEL" "$BM"
cell failure-on "$C MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-failure-gate.sh --external-lock 9 "$MODEL" "$BM"
cell fault-default "$C" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BM"
cell fault-plain "$C MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BM"
cell hit-off "" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BM"
cell hit-on "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BM"
kill "$SAMPLER" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/compute-apps.after.csv" 2>&1
flock -u 9
log "hold released; DAY38-G4-CARD-RUN-DONE"
