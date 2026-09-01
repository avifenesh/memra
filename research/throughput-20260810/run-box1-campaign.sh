#!/usr/bin/env bash
# Run the cx-throughput baseline or the focused prefill-tick knob block on box1.
set -uo pipefail

MODE=${1:-baseline}
case "$MODE" in
    baseline|knob|pilot) ;;
    *) echo "usage: $0 baseline|knob|pilot" >&2; exit 2 ;;
esac

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/cx-throughput-2d9359df}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18331}
BASE=http://127.0.0.1:$PORT
SOURCE_COMMIT=${SOURCE_COMMIT:-2d9359df}
STAMP=${CX_TP_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CX_TP_OUT:-$REPO/research/throughput-20260810/remote-$MODE-$STAMP}
SERVER_PID=
GPU_PID=
OVERALL_RC=0
LAST_CLIENT_RC=0

mkdir -p "$OUT"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 120); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

stop_sampler() {
    if [[ -n ${GPU_PID:-} ]]; then
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
        GPU_PID=
    fi
}

cleanup() {
    stop_server
    stop_sampler
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

wait_up() {
    local pid=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

boot_server() {
    local cell=$1
    local grouped=$2
    local tick=$3
    local trace=$4
    mkdir -p "$cell"
    nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
        --format=csv,noheader,nounits -l 1 >"$cell/gpu.csv" 2>&1 &
    GPU_PID=$!
    local -a tick_env=()
    local -a trace_env=()
    if [[ $tick != default ]]; then tick_env+=("MEMRA_PREFILL_TICK=$tick"); fi
    if [[ $trace == 1 ]]; then trace_env+=("MEMRA_TICK_TRACE=1"); fi
    env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_PREFILL_TICK -u MEMRA_PRIME_BATCH \
        -u MEMRA_TICK_TRACE -u MEMRA_TTFT_TRACE -u MEMRA_API_KEY \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_CTX=262144 \
        MEMRA_MAX_SESSIONS=64 \
        MEMRA_REUSE_POOL=2 \
        MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_PREFIX_DEDUP=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MOE_GROUPED="$grouped" \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        "${tick_env[@]}" "${trace_env[@]}" \
        "$BIN" >"$cell/server.log" 2>&1 &
    SERVER_PID=$!
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: server did not become ready for $cell"
        tail -160 "$cell/server.log" || true
        return 1
    fi
    echo "ready cell=$cell pid=$SERVER_PID grouped=$grouped tick=$tick trace=$trace at $(date -u +%FT%TZ)"
}

capture_server() {
    local cell=$1
    curl -sf "$BASE/metrics" >"$cell/metrics-final.json" 2>"$cell/metrics-final.err" || true
    curl -sf "$BASE/health" >"$cell/health-final.json" 2>"$cell/health-final.err" || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        >"$cell/compute-apps-final.txt" 2>&1 || true
    stop_server
    stop_sampler
    grep -n -E '\[worker\].*decode chunk cap|\[admission\]|VRAM defer|session cap|\[tick\]' \
        "$cell/server.log" >"$cell/limiting-lines.txt" || true
    grep -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed|batch step:|FATAL' \
        "$cell/server.log" >"$cell/failure-lines.txt" || true
}

run_cell() {
    local server_dir=$1
    local name=$2
    local concurrency=$3
    local prompt_tokens=$4
    local max_tokens=$5
    local seed=$6
    local cell="$server_dir/$name"
    mkdir -p "$cell"
    set +e
    timeout 3600 python3 research/throughput-20260810/run_workloads.py \
        "$BASE" "$cell/requests.jsonl" \
        --label "$name" --concurrency "$concurrency" --prompt-tokens "$prompt_tokens" \
        --max-tokens "$max_tokens" --seed "$seed" 2>&1 | tee "$cell/client.log"
    local client_rc=${PIPESTATUS[0]}
    set -e
    LAST_CLIENT_RC=$client_rc
    printf '%s\n' "$client_rc" >"$cell/client.rc"
    echo "cell=$name client_rc=$client_rc at $(date -u +%FT%TZ)"
    if [[ $client_rc -ne 0 ]]; then OVERALL_RC=1; fi
}

warm_server() {
    local server_dir=$1
    local seed=$2
    run_cell "$server_dir" warmup 1 128 16 "$seed"
    if [[ $LAST_CLIENT_RC -ne 0 ]]; then return 1; fi
    echo "clearing 30-second step window after warmup"
    sleep 31
}

run_baseline_server() {
    local rep=$1
    local arm=$2
    local grouped=$3
    local server_dir="$OUT/rep$rep-$arm"
    echo "=== baseline rep=$rep arm=$arm grouped=$grouped at $(date -u +%FT%TZ)"
    if ! boot_server "$server_dir" "$grouped" default 0; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    if ! warm_server "$server_dir" "$((rep * 1000 + grouped * 100 + 1))"; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    local c
    for c in 8 16 24 32 48 64; do
        run_cell "$server_dir" "decode-c$c" "$c" 128 256 "$((rep * 1000 + grouped * 100 + c))"
        if [[ $LAST_CLIENT_RC -ne 0 ]]; then
            capture_server "$server_dir"
            OVERALL_RC=1
            return
        fi
    done
    run_cell "$server_dir" mixed-c16 16 2000 256 "$((rep * 1000 + grouped * 100 + 216))"
    capture_server "$server_dir"
    if [[ -s $server_dir/failure-lines.txt ]]; then OVERALL_RC=1; fi
}

run_diagnostic() {
    local server_dir="$OUT/diagnostic-grouped-on-c64"
    echo "=== traced diagnostic grouped=1 c=64 at $(date -u +%FT%TZ)"
    if ! boot_server "$server_dir" 1 default 1; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    if ! warm_server "$server_dir" 99001; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    run_cell "$server_dir" decode-c64-traced 64 128 256 99064
    capture_server "$server_dir"
    if [[ -s $server_dir/failure-lines.txt ]]; then OVERALL_RC=1; fi
}

run_knob_server() {
    local rep=$1
    local arm=$2
    local tick=$3
    local server_dir="$OUT/rep$rep-$arm"
    echo "=== knob rep=$rep arm=$arm tick=$tick grouped=1 at $(date -u +%FT%TZ)"
    if ! boot_server "$server_dir" 1 "$tick" 0; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    if ! warm_server "$server_dir" "$((50000 + rep * 100 + (${tick/default/0})))"; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    run_cell "$server_dir" mixed-c16 16 2000 256 "$((60000 + rep * 100 + (${tick/default/0})))"
    capture_server "$server_dir"
    if [[ -s $server_dir/failure-lines.txt ]]; then OVERALL_RC=1; fi
}

run_pilot() {
    local server_dir="$OUT/prompt-family-pilot"
    echo "=== prompt-family pilot at $(date -u +%FT%TZ)"
    if ! boot_server "$server_dir" 0 default 0; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    if ! warm_server "$server_dir" 701001; then
        capture_server "$server_dir"
        OVERALL_RC=1
        return
    fi
    run_cell "$server_dir" decode-c16 16 128 256 701016
    if [[ $LAST_CLIENT_RC -eq 0 ]]; then
        run_cell "$server_dir" mixed-c16 16 2000 256 702016
    fi
    capture_server "$server_dir"
    if [[ -s $server_dir/failure-lines.txt ]]; then OVERALL_RC=1; fi
}

echo "=== cx-throughput mode=$MODE start $(date -u +%FT%TZ) host=$(hostname)"
printf '%s\n' "$SOURCE_COMMIT" >"$OUT/source-commit.txt"
sha256sum "$BIN" >"$OUT/binary-sha256.txt"
stat -c '%n %s bytes %y' "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifacts.txt"
sha256sum "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifact-sha256.txt"

exec 9>/tmp/memra-gpu.lock
echo "waiting for /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
flock -w "${LOCK_WAIT:-7200}" 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "acquired /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
nvidia-smi >"$OUT/nvidia-smi-before.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-before.txt" 2>&1 || true

if [[ $MODE == baseline ]]; then
    run_baseline_server 1 off 0
    run_baseline_server 1 on 1
    run_baseline_server 2 on 1
    run_baseline_server 2 off 0
    run_baseline_server 3 off 0
    run_baseline_server 3 on 1
    run_diagnostic
elif [[ $MODE == knob ]]; then
    run_knob_server 1 default default
    run_knob_server 1 tick2048 2048
    run_knob_server 2 tick2048 2048
    run_knob_server 2 default default
    run_knob_server 3 default default
    run_knob_server 3 tick2048 2048
else
    run_pilot
fi

nvidia-smi >"$OUT/nvidia-smi-after.txt" 2>&1
echo "=== cx-throughput mode=$MODE done rc=$OVERALL_RC at $(date -u +%FT%TZ)"
exit "$OVERALL_RC"
