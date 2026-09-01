#!/usr/bin/env bash
# Block 3: one honest concurrent requested-128k capacity row at MEMRA_CTX=262144.
set -uo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-/opt/scratch/nvme/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18243}
BASE=http://127.0.0.1:$PORT
CONCURRENCY=${CONCURRENCY:-24}
STAMP=${VAL256_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${VAL256_OUT:-$REPO/research/val256-20260809/raw/block3-capacity-$STAMP}
SERVER_PID=
GPU_PID=

mkdir -p "$OUT"
cd "$REPO" || exit 1
exec > >(tee "$OUT/driver.log") 2>&1

cleanup() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 120); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n ${GPU_PID:-} ]]; then
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== block3 capacity start at $(date -u +%FT%TZ) host=$(hostname)"
git rev-parse HEAD >"$OUT/commit.txt"
git status --short --branch >"$OUT/git-status.txt"
sha256sum "$BIN" >"$OUT/binary-sha256.txt"
stat -c '%n %s bytes' "$MODEL_ROOT"/IQ4_XS/*.gguf "$DRAFT" >"$OUT/artifacts.txt"

exec 9>/tmp/memra-gpu.lock
echo "waiting for /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
flock -w "${LOCK_WAIT:-7200}" 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "acquired /tmp/memra-gpu.lock at $(date -u +%FT%TZ)"
nvidia-smi >"$OUT/nvidia-smi-before.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-before.txt" 2>&1 || true
nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
    --format=csv,noheader,nounits -l 1 >"$OUT/gpu.csv" 2>&1 &
GPU_PID=$!

env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_API_KEY \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="step=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai \
    MEMRA_CTX=262144 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    "$BIN" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 900); do
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
done
if [[ $ready -ne 1 ]]; then
    echo "FAIL: capacity server did not become ready"
    tail -120 "$OUT/server.log" || true
    exit 3
fi

set +e
timeout 43200 python3 research/val256-20260809/run_capacity_workload.py \
    "$BASE" "$OUT/requests.jsonl" --concurrency "$CONCURRENCY" \
    2>&1 | tee "$OUT/client.log"
CLIENT_RC=${PIPESTATUS[0]}
set -e
curl -sf "$BASE/metrics" >"$OUT/metrics-final.json" 2>"$OUT/metrics-final.err" || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-peak-tail.txt" 2>&1 || true
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=

grep -n -E '\[admission\]|reclaim-on-defer|VRAM defer|\[spec-k\]' "$OUT/server.log" \
    >"$OUT/admission-lines.txt" || true
grep -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed' "$OUT/server.log" \
    >"$OUT/failure-lines.txt" || true
set +e
python3 research/val256-20260809/analyze_capacity.py "$OUT" \
    --out "$OUT/capacity-summary.json" 2>&1 | tee "$OUT/capacity-summary.log"
SUMMARY_RC=${PIPESTATUS[0]}
set -e
nvidia-smi >"$OUT/nvidia-smi-after.txt" 2>&1
echo "=== block3 capacity done client_rc=$CLIENT_RC summary_rc=$SUMMARY_RC at $(date -u +%FT%TZ)"
[[ $CLIENT_RC -eq 0 && $SUMMARY_RC -eq 0 ]]
