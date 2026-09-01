#!/usr/bin/env bash
# Block 2: request-shaped admission at the real PP-2 262144 serving context.
set -uo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-/opt/dl-image/nvme/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18242}
BASE=http://127.0.0.1:$PORT
STAMP=${VAL256_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${VAL256_OUT:-$REPO/research/val256-20260809/raw/block2-admission-$STAMP}
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
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

boot_server() {
    local order=$1
    mkdir -p "$OUT/$order"
    env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_API_KEY \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_CTX=262144 \
        MEMRA_MAX_SESSIONS=64 \
        MEMRA_REUSE_POOL=2 \
        MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        "$BIN" >"$OUT/$order/server.log" 2>&1 &
    SERVER_PID=$!
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: $order server did not become ready"
        tail -120 "$OUT/$order/server.log" || true
        return 1
    fi
    echo "$order ready pid=$SERVER_PID at $(date -u +%FT%TZ)"
}

run_order() {
    local order=$1
    echo "=== order=$order at $(date -u +%FT%TZ)"
    boot_server "$order" || return 1
    set +e
    timeout 28800 python3 research/val256-20260809/run_admission_workload.py \
        "$BASE" "$OUT/$order/requests.jsonl" "$order" \
        2>&1 | tee "$OUT/$order/client.log"
    local client_rc=${PIPESTATUS[0]}
    set -e
    curl -sf "$BASE/metrics" >"$OUT/$order/metrics-final.json" \
        2>"$OUT/$order/metrics-final.err" || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >"$OUT/$order/compute-apps-final.txt" 2>&1 || true
    stop_server
    grep -n -E '\[admission\]|reclaim-on-defer|VRAM defer' "$OUT/$order/server.log" \
        >"$OUT/$order/admission-lines.txt" || true
    grep -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed' \
        "$OUT/$order/server.log" >"$OUT/$order/failure-lines.txt" || true
    echo "=== order=$order client_rc=$client_rc at $(date -u +%FT%TZ)"
    return "$client_rc"
}

echo "=== block2 admission start at $(date -u +%FT%TZ) host=$(hostname)"
git rev-parse HEAD >"$OUT/commit.txt"
git status --short --branch >"$OUT/git-status.txt"
sha256sum "$BIN" >"$OUT/binary-sha256.txt"
stat -c '%n %s bytes' "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifacts.txt"

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

run_order forward || exit $?
run_order inverse || exit $?

set +e
python3 research/val256-20260809/analyze_admission.py "$OUT" \
    --out "$OUT/admission-summary.json" 2>&1 | tee "$OUT/admission-summary.log"
SUMMARY_RC=${PIPESTATUS[0]}
set -e
nvidia-smi >"$OUT/nvidia-smi-after.txt" 2>&1
echo "=== block2 admission done summary_rc=$SUMMARY_RC at $(date -u +%FT%TZ)"
exit "$SUMMARY_RC"
