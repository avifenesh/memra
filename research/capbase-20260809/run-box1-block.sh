#!/usr/bin/env bash
# Run one bounded capbase measurement block on box1.
set -uo pipefail

MODE=${1:-}
case "$MODE" in
    capacity) PORT=${PORT:-18244} ;;
    bursts) PORT=${PORT:-18245} ;;
    sustained) PORT=${PORT:-18246} ;;
    park) PORT=${PORT:-18247} ;;
    *) echo "usage: $0 capacity|bursts|sustained|park" >&2; exit 2 ;;
esac

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-/opt/scratch/nvme/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
BASE=http://127.0.0.1:$PORT
STAMP=${CAPBASE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CAPBASE_OUT:-$REPO/research/capbase-20260809/raw/block-$MODE-$STAMP}
SERVER_PID=
GPU_PID=
OVERALL_RC=0

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
    local cell=$1
    mkdir -p "$cell"
    nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
        --format=csv,noheader,nounits -l 1 >"$cell/gpu.csv" 2>&1 &
    GPU_PID=$!
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
        "$BIN" >"$cell/server.log" 2>&1 &
    SERVER_PID=$!
    if ! wait_up "$SERVER_PID"; then
        echo "FAIL: server did not become ready for $cell"
        tail -120 "$cell/server.log" || true
        return 1
    fi
    echo "ready cell=$cell pid=$SERVER_PID at $(date -u +%FT%TZ)"
}

capture_cell() {
    local cell=$1
    curl -sf "$BASE/metrics" >"$cell/metrics-final.json" 2>"$cell/metrics-final.err" || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        >"$cell/compute-apps-final.txt" 2>&1 || true
    stop_server
    stop_sampler
    grep -n -E '\[admission\]|reclaim-on-defer|VRAM defer|\[spec-k\]' "$cell/server.log" \
        >"$cell/admission-lines.txt" || true
    grep -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed' "$cell/server.log" \
        >"$cell/failure-lines.txt" || true
}

echo "=== capbase mode=$MODE start $(date -u +%FT%TZ) host=$(hostname)"
git rev-parse HEAD >"$OUT/commit.txt"
git status --short --branch >"$OUT/git-status.txt"
sha256sum "$BIN" >"$OUT/binary-sha256.txt"
stat -c '%n %s bytes %y' "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifacts.txt"
cp research/val256-20260809/raw/block1-affinity-20260809T150220Z/artifact-sha256.txt \
    "$OUT/artifact-sha256.txt"
sha256sum "$OUT/artifact-sha256.txt" >"$OUT/artifact-manifest-sha256.txt"

exec 9>/tmp/memra-gpu.lock
echo "waiting for /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
flock -w "${LOCK_WAIT:-7200}" 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "acquired /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
nvidia-smi >"$OUT/nvidia-smi-before.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-before.txt" 2>&1 || true

if [[ $MODE == capacity ]]; then
    for max_ctx in 8192 32768 131072 262144; do
        cell="$OUT/ctx$max_ctx"
        echo "=== capacity max_ctx=$max_ctx at $(date -u +%FT%TZ)"
        if ! boot_server "$cell"; then
            capture_cell "$cell"
            OVERALL_RC=1
            continue
        fi
        set +e
        timeout 43200 python3 research/capbase-20260809/run_workloads.py capacity \
            "$BASE" "$cell/requests.jsonl" --max-ctx "$max_ctx" --concurrency 24 --max-tokens 64 \
            2>&1 | tee "$cell/client.log"
        client_rc=${PIPESTATUS[0]}
        set -e
        capture_cell "$cell"
        set +e
        python3 research/capbase-20260809/analyze_capacity.py "$cell" \
            --out "$cell/capacity-summary.json" 2>&1 | tee "$cell/capacity-summary.log"
        summary_rc=${PIPESTATUS[0]}
        set -e
        echo "capacity max_ctx=$max_ctx client_rc=$client_rc summary_rc=$summary_rc"
        if [[ $client_rc -ne 0 || $summary_rc -ne 0 ]]; then OVERALL_RC=1; fi
    done
elif [[ $MODE == bursts ]]; then
    for concurrency in 4 8; do
        cell="$OUT/c$concurrency"
        echo "=== burst c=$concurrency at $(date -u +%FT%TZ)"
        if ! boot_server "$cell"; then
            capture_cell "$cell"
            OVERALL_RC=1
            continue
        fi
        set +e
        timeout 14400 python3 research/capbase-20260809/run_workloads.py burst \
            "$BASE" "$cell/requests.jsonl" --concurrency "$concurrency" \
            2>&1 | tee "$cell/client.log"
        client_rc=${PIPESTATUS[0]}
        set -e
        capture_cell "$cell"
        echo "burst c=$concurrency client_rc=$client_rc"
        if [[ $client_rc -ne 0 || -s $cell/failure-lines.txt ]]; then OVERALL_RC=1; fi
    done
elif [[ $MODE == sustained ]]; then
    cell="$OUT/c8-8k-tg128-600s"
    if boot_server "$cell"; then
        set +e
        timeout 45000 python3 research/capbase-20260809/run_workloads.py sustained \
            "$BASE" "$cell/requests.jsonl" --concurrency 8 --duration 600 \
            2>&1 | tee "$cell/client.log"
        client_rc=${PIPESTATUS[0]}
        set -e
        capture_cell "$cell"
        echo "sustained client_rc=$client_rc"
        if [[ $client_rc -ne 0 || -s $cell/failure-lines.txt ]]; then OVERALL_RC=1; fi
    else
        capture_cell "$cell"
        OVERALL_RC=1
    fi
else
    cell="$OUT/park262k-pressure-c4-8k"
    if boot_server "$cell"; then
        set +e
        timeout 28800 python3 research/capbase-20260809/run_workloads.py park \
            "$BASE" "$cell/requests.jsonl" 2>&1 | tee "$cell/client.log"
        client_rc=${PIPESTATUS[0]}
        set -e
        capture_cell "$cell"
        set +e
        python3 research/capbase-20260809/analyze_park.py "$cell" \
            --out "$cell/park-summary.json" 2>&1 | tee "$cell/park-summary.log"
        summary_rc=${PIPESTATUS[0]}
        set -e
        echo "park client_rc=$client_rc summary_rc=$summary_rc"
        if [[ $client_rc -ne 0 || $summary_rc -ne 0 ]]; then OVERALL_RC=1; fi
    else
        capture_cell "$cell"
        OVERALL_RC=1
    fi
fi

nvidia-smi >"$OUT/nvidia-smi-after.txt" 2>&1
echo "=== capbase mode=$MODE done rc=$OVERALL_RC at $(date -u +%FT%TZ)"
exit "$OVERALL_RC"
