#!/usr/bin/env bash
# One-lock Step-3.7 correctness battery for the device sigmoid router lane.
set -euo pipefail

REPO=${SIGROUTER_REPO:-/home/ubuntu/memra-cx-sigrouter}
OUT=${SIGROUTER_CORRECTNESS_OUT:-$REPO/research/sigrouter-20260811/raw/box1-correctness}
MODEL_ROOT=${SIGROUTER_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SIGROUTER_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${SIGROUTER_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
GOLDEN=${SIGROUTER_GOLDEN:-/home/ubuntu/darktrain2/golden-response.bin}
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
MODEL_NAME=step37
PORT=${SIGROUTER_CORRECTNESS_PORT:-18453}
BASE=http://127.0.0.1:$PORT
KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PROMPT=$REPO/tools/fast-gate/prompts/probe.txt

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

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel|spec pending flush failed' \
        "$log" || true)
    if [[ -n $failures ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "gate=$label start=$(date -u +%FT%TZ)"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$log.rc"
    test "$rc" -eq 0
    echo "gate=$label done=$(date -u +%FT%TZ)"
}

run_golden_boot() {
    local ordinal=$1 boot label log
    boot=$(printf '%s/golden-boot-%02d' "$OUT" "$ordinal")
    label=$(printf 'sigrouter-boot-%02d' "$ordinal")
    log=$boot/server.log
    mkdir -p "$boot"
    snapshot "$boot/thermal-before.log" "$label-before"
    echo "boot=$ordinal start=$(date -u +%FT%TZ)"
    env \
        -u MEMRA_SIG_ROUTER \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_K \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_BG_JOB \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
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
    curl -sf "$BASE/metrics" >"$boot/metrics-before.txt"
    "$QOS" \
        --base "$BASE" \
        --model "$MODEL_NAME" \
        --label "$label" \
        --requests 1 \
        --max-tokens 64 \
        --golden "$GOLDEN" \
        --rows "$boot/qos-rows.jsonl" \
        --summary "$boot/qos-summary.json" \
        2>&1 | tee "$boot/qos.log"
    curl -sf "$BASE/metrics" >"$boot/metrics-after.txt"
    stop_server
    assert_server_clean "$log"
    grep -q '"exactness": "match"' "$boot/qos-summary.json"
    grep -q "\"expected_sha256\": \"$EXPECTED_GOLDEN\"" "$boot/qos-summary.json"
    snapshot "$boot/thermal-after.log" "$label-after"
    echo "boot=$ordinal hash=$EXPECTED_GOLDEN done=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" "$QOS" "$PROMPT"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "CORRECTNESS_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$KERNEL" "$RUN_GEN" "$RUN_SPEC" "$SERVER" "$QOS" >"$OUT/SHA256SUMS"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_logged kernel-check "$OUT/kernel-check.log" \
    env -u MEMRA_SIG_ROUTER -u MEMRA_KC_FAST CUDA_VISIBLE_DEVICES=0,1 "$KERNEL" "$MODEL"
grep -q 'moe sigmoid router vs host oracle (cases=68, near_tie_cases=2, Step t=1..64): idx_mismatch=0 masked_pick=0 tie_mismatch=0 weight_bit_mismatch=0 max_weight_ulp=0 OK' "$OUT/kernel-check.log"
grep -q 'ALL GREEN: kernels match CPU reference' "$OUT/kernel-check.log"
wait_idle

run_logged run-gen "$OUT/run-gen.log" \
    env -u MEMRA_SIG_ROUTER CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT" "$RUN_GEN" "$MODEL"
grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen.log"
grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/run-gen.log"
wait_idle

run_logged run-spec "$OUT/run-spec.log" \
    env -u MEMRA_SIG_ROUTER CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" \
    "$RUN_SPEC" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"
wait_idle

for ordinal in $(seq 1 10); do
    run_golden_boot "$ordinal"
done

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "CORRECTNESS_PASS $(date -u +%FT%TZ)"
