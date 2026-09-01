#!/usr/bin/env bash
# Step-3.7-Flash PP-2 cache/spec accumulation receipt on box1.
set -uo pipefail

export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
REPO=${REPO:-$HOME/memra-cx-cachespec}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_DIR=${MODEL_DIR:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_DIR/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_DIR/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18129}
BASE=http://127.0.0.1:$PORT
KEY=${KEY:-cachespec-20260809}
TS=${CACHESPEC_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CACHESPEC_OUT:-$REPO/research/cachespec-20260809/raw/box1/$TS}
WORKLOAD=${WORKLOAD:-$OUT/workload.json}
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}
SEQUENTIAL=${SEQUENTIAL:-12}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-768}
BASE_NOTES=${BASE_NOTES:-80}
SERVER_PID=
SAMPLE_PID=

mkdir -p "$OUT"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

stop_sampler() {
    if [[ -n ${SAMPLE_PID:-} ]]; then
        kill "$SAMPLE_PID" 2>/dev/null || true
        wait "$SAMPLE_PID" 2>/dev/null || true
        SAMPLE_PID=
    fi
}

stop_server() {
    stop_sampler
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}
trap stop_server EXIT INT TERM

wait_up() {
    local pid=$1
    for _ in $(seq 1 360); do
        if curl -sf -H "Authorization: Bearer $KEY" "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

sample_gpu() {
    local out=$1
    while true; do
        nvidia-smi --query-gpu=timestamp,index,temperature.gpu,pstate,clocks.sm,power.draw,utilization.gpu,utilization.memory,memory.used,memory.free \
            --format=csv,noheader,nounits
        sleep 1
    done >"$out" 2>&1
}

gpu_state() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,pstate,clocks.sm,power.draw,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader || true
}

boot_server() {
    local arm=$1
    local spec=$2
    local log=$OUT/$arm/server.log
    mkdir -p "$OUT/$arm/responses"
    local spec_env=()
    if [[ $spec == off ]]; then
        spec_env+=(MEMRA_SERVE_SPEC=0)
    elif [[ $spec == forced ]]; then
        spec_env+=(MEMRA_SPEC_GATE=0)
    fi
    env \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_SPEC_GATE_LOW \
        -u MEMRA_SPEC_GATE_HIGH \
        -u MEMRA_SPEC_K \
        -u MEMRA_REUSE_POOL \
        -u MEMRA_MAX_SESSIONS \
        -u MEMRA_PRIME_CHUNK \
        -u MEMRA_ADMIT_RESERVE_MB \
        -u MEMRA_STEP_OOM_RETRIES \
        "${spec_env[@]}" \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_CTX=262144 \
        MEMRA_PREFIX_CACHE_MB=2048 \
        MEMRA_API_KEY="$KEY" \
        MEMRA_TTFT_TRACE=1 \
        MEMRA_SPEC_STATS=1 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: $arm server did not become ready"
        tail -120 "$log" || true
        return 1
    fi
    if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
        | grep -q "pid=$SERVER_PID,"; then
        echo "FAIL: $arm responder is not child pid $SERVER_PID"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    fi
    sample_gpu "$OUT/$arm/gpu.csv" &
    SAMPLE_PID=$!
    echo "$arm ready pid=$SERVER_PID $(date -u +%FT%TZ)"
}

run_arm_locked() {
    local arm=$1
    local spec=$2
    local mode=$3
    echo "=== arm=$arm spec=$spec mode=$mode lock acquired $(date -u +%FT%TZ)"
    gpu_state >"$OUT/$arm/gpu-pre.txt" 2>&1
    if ! boot_server "$arm" "$spec"; then
        stop_server
        return 1
    fi
    local mode_args=(--mode "$mode")
    timeout 10800 python3 research/cachespec-20260809/replay.py \
        --base "$BASE" \
        --model step \
        --api-key "$KEY" \
        "${mode_args[@]}" \
        --workload "$WORKLOAD" \
        --out "$OUT/$arm/requests.jsonl" \
        --raw-dir "$OUT/$arm/responses" \
        --sequential "$SEQUENTIAL" \
        --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_TOKENS" \
        --base-notes "$BASE_NOTES" \
        >"$OUT/$arm/client.log" 2>&1
    local rc=$?
    curl -sf -H "Authorization: Bearer $KEY" "$BASE/metrics" \
        >"$OUT/$arm/metrics-final.json" 2>"$OUT/$arm/metrics-final.err" || true
    stop_server
    sleep 5
    gpu_state >"$OUT/$arm/gpu-post.txt" 2>&1
    echo "=== arm=$arm rc=$rc lock released $(date -u +%FT%TZ)"
    if [[ $rc -ne 0 ]]; then
        echo "--- client tail"
        tail -100 "$OUT/$arm/client.log" || true
        echo "--- server tail"
        tail -160 "$OUT/$arm/server.log" || true
    fi
    return "$rc"
}

run_arm() {
    local arm=$1
    local spec=$2
    local mode=$3
    mkdir -p "$OUT/$arm"
    (
        exec 9>/tmp/memra-gpu.lock
        flock -w "${LOCK_WAIT:-14400}" 9 || {
            echo "FAIL: GPU lock timeout for $arm"
            exit 75
        }
        run_arm_locked "$arm" "$spec" "$mode"
    )
}

echo "=== cachespec receipt ts=$TS source_commit=$SOURCE_COMMIT host=$(hostname)"
echo "repo=$REPO binary=$BIN"
echo "shape: sequential=$SEQUENTIAL c=$CONCURRENCY max_tokens=$MAX_TOKENS base_notes=$BASE_NOTES"
echo "serve: Step-3.7-Flash PP-2 dev01 ctx=262144 prefix_cache=2048MB spec-policy=default"
test -f "$MODEL" || { echo "FAIL: missing trunk $MODEL"; exit 1; }
test -f "$DRAFT" || { echo "FAIL: missing draft $DRAFT"; exit 1; }

echo "=== artifact manifest"
find "$MODEL_DIR/IQ4_XS" -maxdepth 1 -name 'Step-3.7-flash-IQ4_XS-*.gguf' -print0 \
    | sort -z | xargs -0 sha256sum >"$OUT/artifact-sha256.txt"
sha256sum "$DRAFT" >>"$OUT/artifact-sha256.txt"
cat "$OUT/artifact-sha256.txt"

echo "=== release build"
(
    echo "source_commit=$SOURCE_COMMIT"
    date -u +start=%FT%TZ
    cargo build --release -p memra-server
    rc=$?
    date -u +end=%FT%TZ
    echo "rc=$rc"
    exit "$rc"
) >"$OUT/build.log" 2>&1
build_rc=$?
cat "$OUT/build.log"
[[ $build_rc -eq 0 ]] || exit "$build_rc"
sha256sum "$BIN" >"$OUT/binary-sha256.txt"
cat "$OUT/binary-sha256.txt"

if [[ ${RUN_DEFAULT:-1} == 1 ]]; then
    echo "=== default-policy arm"
    run_arm default default record || exit $?
fi
if [[ ${RUN_SPEC_OFF:-1} == 1 ]]; then
    test -f "$WORKLOAD" || { echo "FAIL: replay workload missing: $WORKLOAD"; exit 1; }
    echo "=== spec-off arm"
    run_arm spec-off off replay || exit $?
fi
if [[ ${RUN_FORCED_SPEC:-0} == 1 ]]; then
    test -f "$WORKLOAD" || { echo "FAIL: replay workload missing: $WORKLOAD"; exit 1; }
    echo "=== forced-spec affinity control"
    run_arm forced-spec forced replay || exit $?
fi

echo "CACHESPEC_RUN_DONE out=$OUT $(date -u +%FT%TZ)"
