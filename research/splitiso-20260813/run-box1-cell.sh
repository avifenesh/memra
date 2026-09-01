#!/usr/bin/env bash
# One restartable exactness-only split-map cell on box1 physical GPU 1.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
export TMPDIR=${SPLITISO_TMPDIR:-/home/ubuntu/tmp-lanes}

ROOT=${SPLITISO_ROOT:-/opt/scratch/nvme/cx-splitiso}
REPO=${SPLITISO_REPO:-$ROOT/memra}
SERVER=${SPLITISO_SERVER:-$REPO/target/release/memra-server}
MODEL=${SPLITISO_MODEL:-/opt/scratch/nvme/cx-lcprestore/models/gemma-4-12b-it-qat-q4_0.gguf}
EXPECTED_SOURCE=${SPLITISO_EXPECTED_SOURCE:?set SPLITISO_EXPECTED_SOURCE}
OUT=${SPLITISO_OUT:?set SPLITISO_OUT to a new cell directory}
SPLITS=${SPLITISO_SPLITS:?set SPLITISO_SPLITS}
DETAIL_BOUNDARIES=${SPLITISO_DETAIL_BOUNDARIES:-}
MAP_PROMPTS=${SPLITISO_MAP_PROMPTS:-lcprestore}
PORT=${SPLITISO_PORT:-18832}

GPU_PHYSICAL=${SPLITISO_GPU_PHYSICAL:-1}
GPU_UUID=${SPLITISO_GPU_UUID:-GPU-2b4cf166-fd33-f161-8536-ca04bc72280c}
GPU_LOCK=${SPLITISO_GPU_LOCK:-/tmp/memra-gpu-1.lock}
GLOBAL_LOCK=${SPLITISO_GLOBAL_LOCK-/tmp/memra-gpu.lock}
MODEL_SHA256=93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b
LANE=$REPO/research/lcprestore-20260813
DETAIL_REDUCER=$REPO/research/splitiso-20260813/analyze_split_detail.py
WORKLOAD=$REPO/research/sellgate-20260812/workload.lock.json

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT" "$TMPDIR"
exec > >(tee "$OUT/orchestrator.log") 2>&1

server_pid=
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if test -n "${server_pid:-}"; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "FAIL: owned server did not stop after 60s; sending KILL"
            kill -KILL "$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    nvidia-smi -i "$GPU_PHYSICAL" \
        --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
        2>/dev/null | tee "$OUT/compute-apps-cleanup.log" || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "CELL_PREFLIGHT ts=$(date -u +%FT%TZ) host=$(hostname) splits=$SPLITS detail=$DETAIL_BOUNDARIES map_prompts=$MAP_PROMPTS"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
# Raw evidence is created under this lane before the cell starts. Require the committed source to
# have no tracked/index drift while allowing only untracked receipts to coexist with the run.
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
test -x "$SERVER"
test "$(nvidia-smi -i "$GPU_PHYSICAL" --query-gpu=uuid --format=csv,noheader | tr -d ' ')" = "$GPU_UUID"
if ss -tln 2>/dev/null | grep -q "[:.]$PORT "; then
    echo "FAIL: port $PORT already has a listener"
    exit 1
fi
python3 -m py_compile "$LANE/split_exactness.py" "$LANE/verify_split_receipts.py" "$DETAIL_REDUCER"

# On box1, take the money-lane coordination lock non-blocking as well as card 1. This makes the
# preflight race-free. The local 5090 invocation explicitly sets an empty GLOBAL_LOCK and uses its
# sole shared lock below.
if test -n "$GLOBAL_LOCK"; then
    exec 9>"$GLOBAL_LOCK"
    if ! flock -n 9; then
        echo "YIELD: $GLOBAL_LOCK is held; money-lane campaign outranks split isolation"
        exit 75
    fi
fi
exec 8>"$GPU_LOCK"
if test "${SPLITISO_GPU_LOCK_HELD:-0}" = 1; then
    echo "GPU_LOCK_INHERITED ts=$(date -u +%FT%TZ) gpu=$GPU_LOCK"
elif ! flock -n 8; then
    echo "YIELD: $GPU_LOCK is held; assigned GPU lane is occupied"
    exit 75
fi
echo "LOCKS_ACQUIRED ts=$(date -u +%FT%TZ) global=${GLOBAL_LOCK:-none} gpu=$GPU_LOCK"

apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
if test -n "$apps"; then
    printf '%s\n' "$apps" | tee "$OUT/compute-apps-preflight.log"
    echo "FAIL: compute process present after both locks were acquired"
    exit 1
fi
echo "compute_apps=none" | tee "$OUT/compute-apps-preflight.log"
actual_model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
if test "$actual_model_sha" != "$MODEL_SHA256"; then
    echo "FAIL: model SHA-256 mismatch: expected=$MODEL_SHA256 actual=$actual_model_sha model=$MODEL"
    exit 1
fi
nvidia-smi -i "$GPU_PHYSICAL" \
    --query-gpu=timestamp,index,uuid,name,pstate,temperature.gpu,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUT/gpu-before.log"
git -C "$REPO" show --no-patch --format=fuller HEAD | tee "$OUT/source-commit.log"
{
    printf '%s  %s\n' "$actual_model_sha" "$MODEL"
    sha256sum "$WORKLOAD"
} | tee "$OUT/input-sha256.log"

detail=0
if test -n "$DETAIL_BOUNDARIES"; then detail=1; fi
env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
    -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
    -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SERVE_BATCH \
    -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP \
    -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
    CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" MEMRA_MODELS="g12=$MODEL" \
    MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_CTX=8192 \
    MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=8192 MEMRA_PREFIX_DEDUP=1 \
    MEMRA_PREFIX_PARTIAL_RESTORE=1 MEMRA_PREFIX_SPLIT_TRACE=1 \
    MEMRA_PREFIX_SPLIT_DETAIL="$detail" \
    MEMRA_PREFIX_SPLIT_TRACE_BOUNDARIES="$DETAIL_BOUNDARIES" \
    MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
    "$SERVER" > >(tee "$OUT/candidate-server.log" >/dev/null) 2>&1 &
server_pid=$!
for _ in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "FAIL: server died during boot"
        tail -200 "$OUT/candidate-server.log"
        exit 1
    fi
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/readyz" >/dev/null

set +e
timeout 3600 python3 "$LANE/split_exactness.py" --mode map \
    --map-prompts "$MAP_PROMPTS" \
    --candidate "g12-candidate,http://127.0.0.1:$PORT,g12" \
    --workload-lock "$WORKLOAD" --out "$OUT/requests.jsonl" \
    --namespace cx-splitiso --splits "$SPLITS" --main-split "${SPLITS%%,*}" \
    --repetitions 1 --timeout 1800 --physical-gpu "$GPU_PHYSICAL" --gpu-uuid "$GPU_UUID" \
    2>&1 | tee "$OUT/requests.log"
request_rc=${PIPESTATUS[0]}
set -e
test "$request_rc" -eq 0

kill -TERM "$server_pid"
wait "$server_pid" || true
server_pid=

python3 "$LANE/verify_split_receipts.py" --mode map \
    --candidate-log "$OUT/candidate-server.log" --requests "$OUT/requests.jsonl" \
    --out "$OUT/split-state-receipts.json" --physical-gpu "$GPU_PHYSICAL" \
    --gpu-uuid "$GPU_UUID" --gpu-lock "$GPU_LOCK" \
    2>&1 | tee "$OUT/split-state-verify.log"
if test "$detail" -eq 1; then
    python3 "$DETAIL_REDUCER" --logs "$OUT/candidate-server.log" \
        --out "$OUT/targeted-detail.json" 2>&1 | tee "$OUT/targeted-detail.log"
fi

if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|illegal memory access|ILLEGAL_ADDRESS' \
    "$OUT/candidate-server.log"; then
    echo "FAIL: candidate server emitted a fatal marker"
    exit 1
fi
echo "CELL_COMPLETE ts=$(date -u +%FT%TZ) splits=$SPLITS detail=$DETAIL_BOUNDARIES map_prompts=$MAP_PROMPTS"
