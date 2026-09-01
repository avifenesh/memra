#!/usr/bin/env bash
# Capture the owner-shaped accumulation receipt on the live RunPod, then restore its server.
set -uo pipefail

REPO=${REPO:-/workspace/memra-cx-cachespec}
BIN=${BIN:-$REPO/target/release/memra-server}
OWNER_REPO=${OWNER_REPO:-/workspace/memra}
OWNER_BIN=${OWNER_BIN:-$OWNER_REPO/target/release/memra-server}
ENV_FILE=${ENV_FILE:-/root/serve-env.sh}
PID_FILE=${PID_FILE:-/root/memra-server.pid}
OWNER_LOG=${OWNER_LOG:-/var/log/memra-server.log}
KEY_FILE=${KEY_FILE:-/root/OWNER-KEY.txt}
PORT=${PORT:-8002}
BASE=http://127.0.0.1:$PORT
TS=${CACHESPEC_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CACHESPEC_OUT:-$REPO/research/cachespec-20260809/raw/runpod/$TS}
WORKLOAD=$OUT/workload.json
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}
SEQUENTIAL=${SEQUENTIAL:-12}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-768}
BASE_NOTES=${BASE_NOTES:-80}
TEST_PID=
SAMPLE_PID=
ORIGINAL_PID=
ORIGINAL_WAS_RUNNING=0
RESTORED=0

mkdir -p "$OUT/default/responses"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

pid_is_live() {
    local pid=${1:-}
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

wait_down() {
    local pid=$1
    for _ in $(seq 1 90); do
        pid_is_live "$pid" || return 0
        sleep 1
    done
    return 1
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 360); do
        if curl -fsS "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        pid_is_live "$pid" || return 1
        sleep 2
    done
    return 1
}

wait_gpu_quiescent() {
    local limit_mb=${QUIESCENT_GPU_MB:-1024}
    local used_mb
    for _ in $(seq 1 180); do
        used_mb=$(nvidia-smi --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nr | head -1)
        if [[ $used_mb =~ ^[0-9]+$ ]] && (( used_mb <= limit_mb )); then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: GPU memory did not quiesce below ${limit_mb}MiB"
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free \
        --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader || true
    return 1
}

stop_pid() {
    local pid=${1:-}
    pid_is_live "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    if ! wait_down "$pid"; then
        echo "WARN: pid=$pid did not exit after SIGTERM; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        wait_down "$pid" || true
    fi
}

restore_owner() {
    [[ $ORIGINAL_WAS_RUNNING -eq 1 ]] || return 0
    [[ $RESTORED -eq 0 ]] || return 0
    if [[ -n ${TEST_PID:-} ]]; then
        stop_pid "$TEST_PID"
        TEST_PID=
    fi
    wait_gpu_quiescent || return 1
    # Use the owner's documented launch surface verbatim; do not carry probe overrides.
    (
        # The restored long-lived service must not inherit this experiment's flock.
        exec 9>&-
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
        nohup "$OWNER_BIN" >"$OWNER_LOG" 2>&1 &
        echo $! >"$PID_FILE"
    )
    local restored_pid
    restored_pid=$(cat "$PID_FILE")
    if wait_up "$restored_pid"; then
        RESTORED=1
        echo "OWNER_SERVER_RESTORED pid=$restored_pid $(date -u +%FT%TZ)"
        cp "$OWNER_LOG" "$OUT/owner-restored-startup.log"
        curl -fsS "$BASE/metrics" >"$OUT/owner-restored-metrics.json" || true
        return 0
    fi
    echo "FAIL: owner server did not recover; pid=$restored_pid"
    tail -160 "$OWNER_LOG" || true
    return 1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n ${SAMPLE_PID:-} ]]; then
        kill "$SAMPLE_PID" 2>/dev/null || true
        wait "$SAMPLE_PID" 2>/dev/null || true
        SAMPLE_PID=
    fi
    if [[ -n ${TEST_PID:-} ]]; then
        stop_pid "$TEST_PID"
        TEST_PID=
    fi
    restore_owner || rc=70
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "=== cachespec RunPod receipt ts=$TS source_commit=$SOURCE_COMMIT host=$(hostname)"
echo "shape: sequential=$SEQUENTIAL c=$CONCURRENCY max_tokens=$MAX_TOKENS base_notes=$BASE_NOTES"
test -x "$BIN" || { echo "FAIL: missing instrumented binary $BIN"; exit 1; }
test -x "$OWNER_BIN" || { echo "FAIL: missing owner binary $OWNER_BIN"; exit 1; }
test -r "$ENV_FILE" || { echo "FAIL: missing owner env $ENV_FILE"; exit 1; }
test -s "$KEY_FILE" || { echo "FAIL: missing owner key $KEY_FILE"; exit 1; }

sha256sum "$BIN" "$OWNER_BIN" >"$OUT/binary-sha256.txt"
git -C "$OWNER_REPO" rev-parse HEAD >"$OUT/owner-source-commit.txt" 2>&1 || true
git -C "$OWNER_REPO" status --short --branch >"$OUT/owner-source-status.txt" 2>&1 || true
uname -a >"$OUT/host.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q-pre.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader >"$OUT/gpu-processes-pre.txt" 2>&1 || true
curl -fsS "$BASE/metrics" >"$OUT/owner-metrics-pre.json"
cp "$OWNER_LOG" "$OUT/owner-server-pre.log"

ORIGINAL_PID=$(cat "$PID_FILE" 2>/dev/null || true)
if ! pid_is_live "$ORIGINAL_PID"; then
    echo "FAIL: owner server PID is not live: ${ORIGINAL_PID:-missing}"
    exit 1
fi
ORIGINAL_WAS_RUNNING=1

exec 9>/tmp/memra-gpu.lock
flock -w "${LOCK_WAIT:-14400}" 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
echo "=== RunPod lock acquired $(date -u +%FT%TZ)"

stop_pid "$ORIGINAL_PID"
wait_gpu_quiescent || exit 1
if ss -tlnp 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
    echo "FAIL: port $PORT remained occupied after stopping owner pid=$ORIGINAL_PID"
    ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
    exit 1
fi

(
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    export MEMRA_TTFT_TRACE=1
    export MEMRA_SPEC_STATS=1
    exec "$BIN"
) >"$OUT/default/server.log" 2>&1 &
TEST_PID=$!
if ! wait_up "$TEST_PID"; then
    echo "FAIL: instrumented RunPod server did not become ready"
    tail -180 "$OUT/default/server.log" || true
    exit 1
fi
echo "=== instrumented server ready pid=$TEST_PID $(date -u +%FT%TZ)"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,temperature.gpu,pstate,clocks.sm,power.draw,utilization.gpu,utilization.memory,memory.used,memory.free \
            --format=csv,noheader,nounits
        sleep 1
    done
) >"$OUT/default/gpu.csv" 2>&1 &
SAMPLE_PID=$!

timeout 10800 python3 research/cachespec-20260809/replay.py \
    --base "$BASE" \
    --model stepfun/step-3.7-flash \
    --api-key-file "$KEY_FILE" \
    --mode record \
    --workload "$WORKLOAD" \
    --out "$OUT/default/requests.jsonl" \
    --raw-dir "$OUT/default/responses" \
    --sequential "$SEQUENTIAL" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --base-notes "$BASE_NOTES" \
    >"$OUT/default/client.log" 2>&1
client_rc=$?

kill "$SAMPLE_PID" 2>/dev/null || true
wait "$SAMPLE_PID" 2>/dev/null || true
curl -fsS "$BASE/metrics" >"$OUT/default/metrics-final.json" || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader >"$OUT/gpu-processes-post.txt" 2>&1 || true

stop_pid "$TEST_PID"
TEST_PID=
restore_owner || exit 70
echo "=== RunPod lock released $(date -u +%FT%TZ) client_rc=$client_rc"

if [[ $client_rc -ne 0 ]]; then
    echo "--- client tail"
    tail -120 "$OUT/default/client.log" || true
    echo "--- server tail"
    tail -200 "$OUT/default/server.log" || true
    exit "$client_rc"
fi

echo "CACHESPEC_RUNPOD_DONE out=$OUT $(date -u +%FT%TZ)"
