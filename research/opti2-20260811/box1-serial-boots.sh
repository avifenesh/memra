#!/usr/bin/env bash
# Ten fresh-process serial exactness boots for OPTIPIPE increment 2.
set -euo pipefail

ROOT=${OPTI2_ROOT:-/home/ubuntu/memra-opti2}
OUT=${OPTI2_SERIAL_OUT:-/home/ubuntu/opti2-receipts/serial-boots-2}
PORT=${OPTI2_SERIAL_PORT:-8158}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
GOLDEN=/home/ubuntu/darktrain2/golden-response.bin
SERVER=${ROOT}/target/release/memra-server
QOS=${ROOT}/research/p0iso-20260810/qos_probe.py
ADDR=127.0.0.1:${PORT}
BASE=http://${ADDR}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

server_pid=
stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            for _ in $(seq 1 90); do
                test -z "$(compute_apps)" && return 0
                sleep 1
            done
            compute_apps
            echo "FAIL: GPU processes remained after server shutdown"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

wait_ready() {
    local pid=$1 log=$2
    for _ in $(seq 1 450); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || { tail -100 "$log"; return 1; }
        sleep 2
    done
    tail -100 "$log"
    return 1
}

snapshot() {
    local path=$1
    {
        date -u +%FT%TZ
        nvidia-smi \
            --query-gpu=index,name,memory.used,memory.total,temperature.gpu,pstate,clocks.sm,power.draw \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } > "$path" 2>&1
}

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }

source_commit=$(git -C "$ROOT" rev-parse HEAD)
server_sha=$(sha256sum "$SERVER" | awk '{print $1}')
golden_sha=$(sha256sum "$GOLDEN" | awk '{print $1}')
echo "SERIAL_BOOT_START $(date -u +%FT%TZ) source=$source_commit"
echo "server_sha256=$server_sha"
echo "golden_sha256=$golden_sha"
git -C "$ROOT" status --short --branch
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" > "$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log"
test -z "$(compute_apps)" || { echo "FAIL: box1 not GPU-idle"; exit 1; }

for i in $(seq 1 10); do
    printf -v boot 'boot-%02d' "$i"
    dir="$OUT/$boot"
    log="$dir/server.log"
    mkdir -p "$dir"
    snapshot "$dir/nvidia-smi-before.log"
    echo "=== $boot $(date -u +%FT%TZ) ==="
    env \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_PIPE \
        -u MEMRA_SPEC_STREAM \
        -u MEMRA_SPEC_ADAPT \
        -u MEMRA_SPEC_REPLAY \
        -u MEMRA_OPTI_CONTROLLER_Q \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PREFIX_DEDUP=1 \
        MEMRA_PRIME_BATCH_HOLD_MS=4 \
        MEMRA_SPEC_GATE=0 \
        MEMRA_SPEC_K=1 \
        MEMRA_SPEC_STATS=1 \
        MEMRA_SPEC_DEVACC=1 \
        MEMRA_MODELS="step37=${MODEL}+${DRAFT}" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="$ADDR" \
        "$SERVER" > "$log" 2>&1 &
    server_pid=$!
    wait_ready "$server_pid" "$log"
    python3 "$QOS" \
        --base "$BASE" \
        --model step37 \
        --label serial \
        --requests 1 \
        --max-tokens 64 \
        --golden "$GOLDEN" \
        --rows "$dir/qos-rows.jsonl" \
        --summary "$dir/qos-summary.json"
    hash=$(sed -n 's/.*"text_sha256": "\([^"]*\)".*/\1/p' "$dir/qos-rows.jsonl")
    test -n "$hash"
    printf '%s\n' "$hash" | tee -a "$OUT/hashes.txt"
    stop_server
    if grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|illegal|sentinel|spec pending flush failed' \
        "$log"; then
        echo "FAIL: server failure signature in $log"
        exit 1
    fi
    snapshot "$dir/nvidia-smi-after.log"
done

test "$(wc -l < "$OUT/hashes.txt")" -eq 10
test "$(sort -u "$OUT/hashes.txt" | wc -l)" -eq 1
test "$(sort -u "$OUT/hashes.txt")" = "$golden_sha"
snapshot "$OUT/nvidia-smi-after.log"
test -z "$(compute_apps)"
echo "SERIAL_BOOT_PASS 10/10 hash=$golden_sha $(date -u +%FT%TZ)"
