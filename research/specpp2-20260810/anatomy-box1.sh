#!/usr/bin/env bash
# One bounded, exclusive c=1 forced-spec anatomy block on box1.
set -euo pipefail

ROOT=${SPEC_PP2_ROOT:-/home/ubuntu/memra-cx-specpp2}
OUT=${SPEC_PP2_OUT:-/home/ubuntu/specpp2-receipts/anatomy}
PORT=${SPEC_PP2_PORT:-8139}
ADDR=127.0.0.1:${PORT}
BASE=http://${ADDR}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
BIN=${ROOT}/target/release/memra-server

mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$BIN"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
    echo "FAIL: port ${PORT} already listening"
    exit 1
fi

wait_up() {
    local pid=$1
    for _ in $(seq 1 360); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

stop_server() {
    local pid=$1
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

server_pid=
trap 'test -z "$server_pid" || stop_server "$server_pid"' EXIT

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "ANATOMY_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) source=7cd010c9"
sha256sum "$BIN" > "$OUT/binary-sha256.txt"
stat -c '%n %s bytes' "$MODEL" "$DRAFT" > "$OUT/artifacts.txt"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true

env \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    MEMRA_SPEC_GATE=0 \
    MEMRA_SPEC_K=1 \
    MEMRA_SPEC_STATS=1 \
    MEMRA_SPEC_PP_ANATOMY=1 \
    MEMRA_MODELS="step37=${MODEL}+${DRAFT}" \
    MEMRA_ADDR="$ADDR" \
    "$BIN" > "$OUT/server.log" 2>&1 &
server_pid=$!

if ! wait_up "$server_pid"; then
    echo "FAIL: anatomy server did not become ready"
    tail -100 "$OUT/server.log" || true
    exit 1
fi

python3 tools/load-serve.py \
    --base "$BASE" \
    --model step37 \
    --concurrency 1 \
    --requests 1 \
    --max-tokens 128 \
    --greedy \
    --warmup 0 \
    --label anatomy-k1-c1 \
    --out "$OUT/point.jsonl" \
    > "$OUT/load.log" 2>&1

curl -sf "$BASE/metrics" > "$OUT/metrics.txt"
stop_server "$server_pid"
server_pid=

grep -E '\[spec-(pp-)?anatomy\]|\[spec-phase\]|\[spec-stats\]' "$OUT/server.log" \
    > "$OUT/anatomy-lines.log"
grep -E -i 'error|illegal|sentinel|panic|abort|CUDA_ERROR' "$OUT/server.log" \
    > "$OUT/error-scan.log" || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post.csv"
echo "ANATOMY_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
echo "ANATOMY_DONE"

