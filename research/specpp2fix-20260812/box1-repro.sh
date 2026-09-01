#!/usr/bin/env bash
# Reproduce the naked dual-active PP-2 + forced-spec failure on box1.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"

REPO=${SPEC_PP2FIX_REPO:-/home/ubuntu/memra-cx-specpp2fix}
OUT=${SPEC_PP2FIX_OUT:-$REPO/research/specpp2fix-20260812/raw/box1/repro-baseline}
MODEL_ROOT=${SPEC_PP2FIX_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SPEC_PP2FIX_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${SPEC_PP2FIX_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
SERVER=$REPO/target/release/memra-server
LOAD=$REPO/tools/load-serve.py
PORT=${SPEC_PP2FIX_PORT:-18481}
BASE=http://127.0.0.1:$PORT
ARM=${SPEC_PP2FIX_ARM:-dual}
K=${SPEC_PP2FIX_K:-1}
CONCURRENCY=${SPEC_PP2FIX_CONCURRENCY:-8}
REQUESTS=${SPEC_PP2FIX_REQUESTS:-16}
MAX_TOKENS=${SPEC_PP2FIX_MAX_TOKENS:-64}
LABEL=${SPEC_PP2FIX_LABEL:-naked-${ARM}-c${CONCURRENCY}-k${K}}
PP_DEVICES=${SPEC_PP2FIX_PP_DEVICES:-0,1}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$SERVER" "$LOAD"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

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
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
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

server_pid=
stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
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
trap 'stop_server || true' EXIT INT TERM

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

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git status --short --branch
sha256sum "$SERVER" "$MODEL" "$DRAFT" "$LOAD" >"$OUT/SHA256SUMS"
stat -c '%n %s bytes %y' "$MODEL" "$DRAFT" >"$OUT/artifacts.txt"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "REPRO_LOCK_ACQUIRED $(date -u +%FT%TZ)"
snapshot "$OUT/nvidia-smi-before.log" repro-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

case "$ARM" in
    dual) dual_policy=() ;;
    serial) dual_policy=(MEMRA_DUAL_PP=0) ;;
    *) echo "FAIL: unknown SPEC_PP2FIX_ARM=$ARM"; exit 1 ;;
esac

started=$(date -u +%FT%TZ)
server_log=$OUT/$LABEL-server.log
env \
    -u MEMRA_DUAL_PP \
    -u MEMRA_PP_OVERLAP \
    -u MEMRA_SPEC_PIPE \
    -u MEMRA_SPEC_NOGRAPH \
    "${dual_policy[@]}" \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="step37=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_SERVE_SPEC=1 \
    MEMRA_SPEC_GATE=0 \
    MEMRA_SPEC_K="$K" \
    MEMRA_SPEC_STATS=1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES="$PP_DEVICES" \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    MEMRA_PREFIX_CACHE_MB=0 \
    MEMRA_MAX_SESSIONS=64 \
    MEMRA_LANE_MAX_JUDGE=64 \
    MEMRA_LANE_MAX_HARVEST=64 \
    MEMRA_SLO_P99_MS=1000000 \
    MEMRA_TAG="specpp2fix-repro-$LABEL" \
    "$SERVER" >"$server_log" 2>&1 &
server_pid=$!
wait_ready "$server_log"
curl -sf "$BASE/metrics" >"$OUT/metrics-before.json"

set +e
python3 "$LOAD" \
    --base "$BASE" \
    --model step37 \
    --concurrency "$CONCURRENCY" \
    --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --greedy \
    --warmup 0 \
    --label "$LABEL" \
    --out "$OUT/points.jsonl" \
    --per-request "$OUT/requests.jsonl" \
    >"$OUT/$LABEL-load.log" 2>&1
load_rc=$?
set -e
echo "$load_rc" >"$OUT/load.exit"
cat "$OUT/$LABEL-load.log"
curl -sf "$BASE/metrics" >"$OUT/metrics-after.json" 2>"$OUT/metrics-after.err" || true
stop_server

journalctl -k --since "$started" --no-pager >"$OUT/kernel-since-start.log" 2>&1 || true
grep -Ein \
    'CUDA_ERROR|illegal memory access|ILLEGAL_ADDRESS|sentinel|step error|alloc failed|worker.*died|server fatal|panicked at' \
    "$server_log" >"$OUT/failure-scan.log" || true
cat "$OUT/failure-scan.log"

snapshot "$OUT/nvidia-smi-after.log" repro-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "REPRO_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
