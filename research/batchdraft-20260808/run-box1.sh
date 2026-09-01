#!/usr/bin/env bash
# Reproducible box1 measurement for cross-request speculative draft/verify anatomy.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/batchdraft-20260808
RAW=$LANE/raw/box1
mkdir -p "$RAW"

TS=${BATCHDRAFT_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${BATCHDRAFT_PORT:-8137}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS=${BATCHDRAFT_REPS:-5}
MAX_TOKENS=${BATCHDRAFT_MAX_TOKENS:-96}
BLOCKS=${BATCHDRAFT_BLOCKS:-all}
TARGET=${CARGO_TARGET_DIR:-/opt/scratch/nvme/cx-batchdraft-target}
SERVER=$TARGET/release/memra-server
MSCALE=$TARGET/release/verify-mscale
TRUNK=${STEP35:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${STEP35_DRAFT:-$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}

DRIVER=$RAW/driver-$TS.log
SERVER_LOG=$RAW/server-$TS.log
CLIENT=$RAW/client-$TS.jsonl
GPU=$RAW/gpu-serving-$TS.csv
MSCALE_LOG=$RAW/verify-mscale-$TS.log
MSCALE_GPU=$RAW/gpu-mscale-$TS.csv

exec > >(tee "$DRIVER") 2>&1

for path in "$SERVER" "$MSCALE" "$TRUNK" "$DRAFT"; do
    test -f "$path" || { echo "FAIL missing $path"; exit 1; }
done

echo "=== batchdraft box1 $TS ==="
echo "host=$(hostname) utc=$(date -u +%FT%TZ) commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "status-before:"
git status --short
echo "mounts:"
df -h "$TRUNK" /opt/scratch/nvme
echo "artifacts:"
stat -c '%n %s bytes' "$TRUNK" "$DRAFT"
sha256sum "$TRUNK" "$DRAFT" > "$RAW/artifact-sha256-$TS.txt"
sha256sum "$SERVER" "$MSCALE" > "$RAW/binary-sha256-$TS.txt"
cat "$RAW/artifact-sha256-$TS.txt"
cat "$RAW/binary-sha256-$TS.txt"

gpu_state() {
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw,utilization.gpu \
        --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>&1 || true
}

wait_ready() {
    local pid=$1
    for _ in $(seq 1 240); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 5
    done
    return 1
}

stop_pid() {
    local pid=${1:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

run_serving_block() (
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL serving lock timeout"; exit 75; }
    echo "SERVING_LOCK_ACQUIRED $(date -u +%FT%TZ)"
    gpu_state > "$RAW/gpu-serving-pre-$TS.txt"

    env \
        -u CUDA_VISIBLE_DEVICES \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE_LOW \
        -u MEMRA_SPEC_GATE_HIGH \
        MEMRA_MODELS="step35=${TRUNK}+${DRAFT}" \
        MEMRA_ADDR="$ADDR" \
        MEMRA_CTX=4096 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_SPEC_GATE=0 \
        MEMRA_SPEC_K=3 \
        MEMRA_SPEC_BURST=32 \
        MEMRA_TICK_TRACE=1 \
        MEMRA_SPEC_PHASE=1 \
        "$SERVER" > "$SERVER_LOG" 2>&1 &
    server_pid=$!
    sampler_pid=
    cleanup() {
        test -z "$sampler_pid" || stop_pid "$sampler_pid"
        stop_pid "$server_pid"
        gpu_state > "$RAW/gpu-serving-post-$TS.txt"
        flock -u 9 || true
        echo "SERVING_LOCK_RELEASED $(date -u +%FT%TZ)"
    }
    trap cleanup EXIT

    if ! wait_ready "$server_pid"; then
        echo "FAIL server did not become ready"
        tail -100 "$SERVER_LOG" || true
        exit 1
    fi
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,power.draw,memory.used \
        --format=csv,nounits -lms 100 > "$GPU" 2>&1 &
    sampler_pid=$!

    python3 "$LANE/load-interleaved.py" \
        --base "$BASE" \
        --model step35 \
        --out "$CLIENT" \
        --reps "$REPS" \
        --max-tokens "$MAX_TOKENS"

    # An explicit K suppresses the automatic gate-policy line. The per-request tick receipt is
    # the authoritative proof that the measured requests actually ran speculative K=3 rounds.
    grep -m1 -E '\[tick-spec\].* k=3$' "$SERVER_LOG"
    curl -sf "$BASE/metrics" > "$RAW/metrics-$TS.txt"
    stop_pid "$sampler_pid"
    sampler_pid=
    stop_pid "$server_pid"
    server_pid=
    trap - EXIT
    gpu_state > "$RAW/gpu-serving-post-$TS.txt"
    flock -u 9
    echo "SERVING_LOCK_RELEASED $(date -u +%FT%TZ)"
)

run_mscale_block() (
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL mscale lock timeout"; exit 75; }
    echo "MSCALE_LOCK_ACQUIRED $(date -u +%FT%TZ)"
    gpu_state > "$RAW/gpu-mscale-pre-$TS.txt"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,power.draw,memory.used \
        --format=csv,nounits -lms 100 > "$MSCALE_GPU" 2>&1 &
    sampler_pid=$!
    cleanup() {
        stop_pid "$sampler_pid"
        gpu_state > "$RAW/gpu-mscale-post-$TS.txt"
        flock -u 9 || true
        echo "MSCALE_LOCK_RELEASED $(date -u +%FT%TZ)"
    }
    trap cleanup EXIT

    env \
        -u CUDA_VISIBLE_DEVICES \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MSCALE_INTERLEAVE=1 \
        MEMRA_MSCALE_NOEAGER=1 \
        "$MSCALE" "$TRUNK" 256 25 4,8,12,16 > "$MSCALE_LOG" 2>&1

    stop_pid "$sampler_pid"
    sampler_pid=
    trap - EXIT
    gpu_state > "$RAW/gpu-mscale-post-$TS.txt"
    flock -u 9
    echo "MSCALE_LOCK_RELEASED $(date -u +%FT%TZ)"
)

case "$BLOCKS" in
    all)
        run_serving_block
        run_mscale_block
        ;;
    serving)
        run_serving_block
        ;;
    mscale)
        run_mscale_block
        ;;
    *)
        echo "FAIL BATCHDRAFT_BLOCKS must be all, serving, or mscale (got $BLOCKS)"
        exit 2
        ;;
esac

echo "error scan:"
for log in "$SERVER_LOG" "$MSCALE_LOG"; do
    test -f "$log" || continue
    grep -En 'CUDA_ERROR|out of memory|panicked|fatal|illegal address|Xid|FAIL' "$log" || true
done
echo "BATCHDRAFT_BOX1_DONE $(date -u +%FT%TZ)"
