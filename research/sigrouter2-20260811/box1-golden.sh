#!/usr/bin/env bash
# Ten fresh-boot Step-3.7 golden checks for the zero-DtoH sigmoid-router lane.
set -euo pipefail

REPO=${SIGROUTER2_REPO:-/home/ubuntu/memra-cx-sigrouter2}
OUT=${SIGROUTER2_GOLDEN_OUT:-$REPO/research/sigrouter2-20260811/raw/box1-golden}
MODEL_ROOT=${SIGROUTER2_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf
GOLDEN=${SIGROUTER2_GOLDEN:-/home/ubuntu/darktrain2/golden-response.bin}
EXPECTED=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PORT=${SIGROUTER2_GOLDEN_PORT:-18456}
BASE=http://127.0.0.1:$PORT

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

wait_idle() {
    for _ in $(seq 1 120); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    tail -200 "$log"
    return 1
}

run_boot() {
    local ordinal=$1 boot label log
    boot=$(printf '%s/boot-%02d' "$OUT" "$ordinal")
    label=$(printf 'sigrouter2-boot-%02d' "$ordinal")
    log=$boot/server.log
    mkdir -p "$boot"
    snapshot "$boot/thermal-before.log" "$label-before"
    echo "boot=$ordinal start=$(date -u +%FT%TZ)"
    env \
        -u MEMRA_SIG_ROUTER \
        -u MEMRA_MOE_DEV \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_K \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_BG_JOB \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="step37=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PRIME_BATCH_HOLD_MS=4 \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    "$QOS" \
        --base "$BASE" \
        --model step37 \
        --label "$label" \
        --requests 1 \
        --max-tokens 64 \
        --golden "$GOLDEN" \
        --rows "$boot/qos-rows.jsonl" \
        --summary "$boot/qos-summary.json" \
        2>&1 | tee "$boot/qos.log"
    stop_server
    if grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel' \
        "$log"; then
        echo "FAIL: server failure signature in $log"
        return 1
    fi
    grep -q '"exactness": "match"' "$boot/qos-summary.json"
    grep -q "\"expected_sha256\": \"$EXPECTED\"" "$boot/qos-summary.json"
    snapshot "$boot/thermal-after.log" "$label-after"
    echo "boot=$ordinal hash=$EXPECTED done=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GOLDEN_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" >"$OUT/SHA256SUMS"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED"
snapshot "$OUT/thermal-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

for ordinal in $(seq 1 10); do
    run_boot "$ordinal"
done

snapshot "$OUT/thermal-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "GOLDEN_PASS $(date -u +%FT%TZ)"
