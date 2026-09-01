#!/usr/bin/env bash
# Block 1: PP-2 plain-affinity validation at the real 262144 serving context.
set -uo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-/opt/scratch/nvme/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18241}
BASE=http://127.0.0.1:$PORT
KEY=${KEY:-val256-affinity-20260809}
STAMP=${VAL256_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${VAL256_OUT:-$REPO/research/val256-20260809/raw/block1-affinity-$STAMP}
WORKLOAD=$OUT/workload.json
CTX=262144
SEQUENTIAL=${SEQUENTIAL:-8}
CONCURRENCY=${CONCURRENCY:-1}
MAX_TOKENS=${MAX_TOKENS:-256}
BASE_NOTES=${BASE_NOTES:-650}
SERVER_PID=
GPU_PID=

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

cleanup() {
    stop_server
    if [[ -n ${GPU_PID:-} ]]; then
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_up() {
    local pid=$1
    for _ in $(seq 1 900); do
        curl -sf -H "Authorization: Bearer $KEY" "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

boot_server() {
    local arm=$1 affinity=$2
    local log=$OUT/$arm/server.log
    mkdir -p "$OUT/$arm/responses"
    env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_AFFINITY="$affinity" \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_CTX="$CTX" \
        MEMRA_MAX_SESSIONS=4 \
        MEMRA_REUSE_POOL=4 \
        MEMRA_PREFIX_CACHE_MB=512 \
        MEMRA_API_KEY="$KEY" \
        MEMRA_TTFT_TRACE=1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: $arm server did not become ready"
        tail -120 "$log" || true
        return 1
    fi
    echo "$arm affinity=$affinity ready pid=$SERVER_PID at $(date -u +%FT%TZ)"
}

run_arm() {
    local arm=$1 affinity=$2 mode=$3
    echo "=== arm=$arm affinity=$affinity mode=$mode at $(date -u +%FT%TZ)"
    boot_server "$arm" "$affinity" || return 1
    set +e
    timeout 14400 python3 research/val256-20260809/deep_affinity_replay.py \
        --base "$BASE" --model step --api-key "$KEY" \
        --mode "$mode" --workload "$WORKLOAD" \
        --out "$OUT/$arm/requests.jsonl" --raw-dir "$OUT/$arm/responses" \
        --sequential "$SEQUENTIAL" --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_TOKENS" --base-notes "$BASE_NOTES" \
        2>&1 | tee "$OUT/$arm/client.log"
    local client_rc=${PIPESTATUS[0]}
    set -e
    curl -sf -H "Authorization: Bearer $KEY" "$BASE/metrics" \
        >"$OUT/$arm/metrics-final.json" 2>"$OUT/$arm/metrics-final.err" || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >"$OUT/$arm/compute-apps-final.txt" 2>&1 || true
    stop_server
    echo "=== arm=$arm client_rc=$client_rc at $(date -u +%FT%TZ)"
    if [[ $client_rc -ne 0 ]]; then
        tail -80 "$OUT/$arm/client.log" || true
        tail -120 "$OUT/$arm/server.log" || true
        return "$client_rc"
    fi
}

echo "=== block1 affinity start at $(date -u +%FT%TZ) host=$(hostname)"
echo "shape: PP=2 devices=0,1 ctx=$CTX sequential=$SEQUENTIAL c=$CONCURRENCY max_tokens=$MAX_TOKENS base_notes=$BASE_NOTES"
git rev-parse HEAD >"$OUT/commit.txt"
git status --short --branch >"$OUT/git-status.txt"
test -f "$MODEL" || { echo "FAIL: missing $MODEL"; exit 2; }
test -f "$DRAFT" || { echo "FAIL: missing $DRAFT"; exit 2; }
stat -c '%n %s bytes' "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifacts.txt"
sha256sum "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifact-sha256.txt"
sha256sum "$OUT/artifact-sha256.txt" >"$OUT/artifact-manifest-sha256.txt"

set +e
cargo build --release -p memra-server 2>&1 | tee "$OUT/build.log"
BUILD_RC=${PIPESTATUS[0]}
set -e
[[ $BUILD_RC -eq 0 ]] || exit "$BUILD_RC"

exec 9>/tmp/memra-gpu.lock
echo "waiting for /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
flock -w "${LOCK_WAIT:-7200}" 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "acquired /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
nvidia-smi >"$OUT/nvidia-smi-before.txt" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-before.txt" 2>&1 || true
nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
    --format=csv,noheader,nounits -l 1 >"$OUT/gpu.csv" 2>&1 &
GPU_PID=$!

run_arm record-on 1 record || exit $?
python3 research/val256-20260809/check_affinity_record.py \
    "$OUT/record-on/requests.jsonl" --out "$OUT/record-check.json" \
    2>&1 | tee "$OUT/record-check.log" || exit "${PIPESTATUS[0]}"
for rep in 1 2 3; do
    run_arm "on-$rep" 1 replay || exit $?
    run_arm "off-$rep" 0 replay || exit $?
done

set +e
python3 research/affinity-20260809/compare_gate.py \
    --on "$OUT/on-1/requests.jsonl" \
    --off "$OUT/off-1/requests.jsonl" \
    --on-runs "$OUT/on-2/requests.jsonl" "$OUT/on-3/requests.jsonl" \
    --on-metrics "$OUT/on-1/metrics-final.json" \
    --max-tokens "$MAX_TOKENS" --out "$OUT/predecessor-gate.json" \
    2>&1 | tee "$OUT/predecessor-gate.log"
COMPARE_RC=${PIPESTATUS[0]}
python3 research/val256-20260809/analyze_affinity.py "$OUT" \
    --out "$OUT/affinity-summary.json" 2>&1 | tee "$OUT/affinity-summary.log"
SUMMARY_RC=${PIPESTATUS[0]}
set -e

grep -h '\[spec-k\]' "$OUT"/on-*/server.log "$OUT"/off-*/server.log \
    >"$OUT/spec-k-policy-lines.txt" || true
grep -h -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed' \
    "$OUT"/*/server.log >"$OUT/failure-lines.txt" || true
nvidia-smi >"$OUT/nvidia-smi-after.txt" 2>&1
echo "=== block1 affinity done compare_rc=$COMPARE_RC summary_rc=$SUMMARY_RC at $(date -u +%FT%TZ)"
[[ $COMPARE_RC -eq 0 && $SUMMARY_RC -eq 0 ]]
