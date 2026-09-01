#!/usr/bin/env bash
# Re-run the frozen Q35 mixed-c4 exact-token cell with grouped dispatch ON and OFF.
set -euo pipefail

cd "$(dirname "$0")/../.."

LABEL=${1:?usage: $0 LABEL}
MODEL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
SERVER=target/release/memra-server
OUT=research/eosclass-20260813/raw/$LABEL
LOCK=/tmp/memra-5090.lock
LOCK_WAIT=${EOSCLASS_LOCK_WAIT_SECONDS:-7200}
PORT=${EOSCLASS_Q35_PORT:-18435}
BASE=http://127.0.0.1:$PORT

test -f "$MODEL"
test -x "$SERVER"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }

exec 9>"$LOCK"
echo "waiting up to ${LOCK_WAIT}s for GPU lease: $LOCK" >&2
if ! flock -w "$LOCK_WAIT" 9; then
    echo "GPU lease busy or wait expired: $LOCK" >&2
    exit 75
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
    echo "port $PORT already has a listener" >&2
    exit 1
fi
mkdir -p "$OUT"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "head=$(git rev-parse HEAD)"
    echo "tag=$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
    echo "branch=$(git branch --show-current)"
    git status --short
    sha256sum "$MODEL" "$SERVER" tools/q35-cold-mixed-gate.py \
        research/sellgate-20260812/sellgate_replay.py \
        research/sellgate-20260812/workload.lock.json
    echo "protocol=frozen Q35 mixed90 c=4; grouped ON then OFF; one GPU-lock hold"
    echo "gpu_lock=$LOCK"
    echo "gpu_lock_wait_seconds=$LOCK_WAIT"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
} >"$OUT/provenance.log" 2>&1

SERVER_PID=
stop_server() {
    test -n "${SERVER_PID:-}" || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=
            return 0
        fi
        sleep 1
    done
    echo "owned server pid=$SERVER_PID did not stop after 60 seconds" >&2
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    return 1
}
trap stop_server EXIT

run_arm() {
    local arm=$1 grouped=$2 arm_out=$OUT/$1
    mkdir -p "$arm_out"
    env \
        -u MEMRA_PP_STAGES \
        -u MEMRA_PP_DEVICES \
        -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP \
        -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_PRIME_PIPE \
        -u MEMRA_PREFILL_TICK \
        -u MEMRA_PRIME_BATCH \
        -u MEMRA_PRIME_BATCH_MAX_T \
        -u MEMRA_PRIME_BATCH_HOLD_MS \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_SERVE_B1FAST \
        -u MEMRA_SERVE_GS \
        -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP \
        -u MEMRA_FAST \
        CUDA_VISIBLE_DEVICES=0 \
        MEMRA_MOE_GROUPED="$grouped" \
        MEMRA_MODELS="q35-eosclass=$MODEL" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR=127.0.0.1:$PORT \
        MEMRA_TAG="cx-eosclass-$LABEL-$arm" \
        MEMRA_SERVE_SPEC=0 \
        MEMRA_CTX=8192 \
        MEMRA_PREFIX_CACHE_MB=4096 \
        MEMRA_PREFIX_DEDUP=1 \
        MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 \
        MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$arm_out/server.log" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" >"$arm_out/server.pid"
    for _ in $(seq 1 900); do
        if curl -sf "$BASE/readyz" >"$arm_out/readyz.json" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server died during $arm boot" >&2
            tail -200 "$arm_out/server.log" >&2
            return 1
        fi
        sleep 1
    done
    curl -sf "$BASE/readyz" >"$arm_out/readyz.json"
    curl -sf "$BASE/metrics" >"$arm_out/metrics-before.json"
    nvidia-smi --query-gpu=index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$arm_out/gpu-ready.csv"

    set +e
    timeout 1800 python3 tools/q35-cold-mixed-gate.py \
        --base "$BASE" --model q35-eosclass \
        --namespace "cx-eosclass-$LABEL-$arm" --timeout 600 \
        2>&1 | tee "$arm_out/client.jsonl"
    local client_rc=${PIPESTATUS[0]}
    set -e
    echo "$client_rc" >"$arm_out/client.exit"

    curl -sf "$BASE/metrics" >"$arm_out/metrics-after.json"
    nvidia-smi --query-gpu=index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$arm_out/gpu-after.csv"
    grep -Ein 'out of memory|CUDA_ERROR|panic|fatal|illegal address|misaligned address' \
        "$arm_out/server.log" >"$arm_out/failure-signature-scan.log" || true
    grep -E '^\[prime-batch\].*carried=[1-9]' "$arm_out/server.log" \
        >"$arm_out/carried-prime-violations.log" || true
    stop_server
    return "$client_rc"
}

overall=0
run_arm grouped-on 1 || overall=1
run_arm grouped-off 0 || overall=1
echo "$overall" >"$OUT/overall.exit"
trap - EXIT
exit "$overall"
