#!/usr/bin/env bash
# Local RTX 5090 PP-2 + prefix-cache exactness/timing gate. No clock changes.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/prefixmoney-20260812
OUT=${PREFIXMONEY_OUT:-$LANE/raw/local5090}
MODEL=${PREFIXMONEY_MODEL:-/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
MODEL_NAME=${PREFIXMONEY_MODEL_NAME:-q27}
PORT=${PREFIXMONEY_PORT:-18512}
BASE=http://127.0.0.1:$PORT
SERVER=$ROOT/target/release/memra-server
SERVER_PID=
SAMPLER_PID=

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

stop_sampler() {
    if [[ -n ${SAMPLER_PID:-} ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=
    fi
}

# shellcheck disable=SC2329  # invoked by trap
cleanup() {
    stop_server
    stop_sampler
}
trap cleanup EXIT INT TERM

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,memory.total,memory.used,memory.free --format=csv,noheader
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader || true
    } >"$path" 2>&1
}

wait_ready() {
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server died before readiness"
            tail -200 "$OUT/server.log"
            return 1
        fi
        sleep 1
    done
    echo "server readiness timeout"
    tail -200 "$OUT/server.log"
    return 1
}

test -f "$MODEL"
test -x "$SERVER"
cd "$ROOT"
exec 9>/tmp/gpu5090.lock
flock -w 14400 9
echo "lock_acquired=$(date -u +%FT%TZ)"
snapshot "$OUT/nvidia-smi-before.log" before
sha256sum "$MODEL" "$SERVER" "$LANE/prefix_gate.py" >"$OUT/SHA256SUMS.input"
{
    echo "source_commit=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "model_name=$MODEL_NAME"
    echo "MEMRA_PP_STAGES=2"
    echo "MEMRA_PP_DEVICES=0,0"
    echo "MEMRA_DUAL_PP=<unset; default Auto>"
    echo "MEMRA_PP_OVERLAP=<unset; follows dual PP>"
    echo "MEMRA_SWA_RING=0"
    echo "MEMRA_PP_HOST_BOUNCE=<unset>"
    echo "MEMRA_PREFIX_CACHE_MB=512"
    echo "MEMRA_SERVE_SPEC=0"
} >"$OUT/provenance.txt"

nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 >"$OUT/gpu.csv" 2>&1 &
SAMPLER_PID=$!

env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
    -u MEMRA_PP_STREAMS -u MEMRA_PP_SHARD -u MEMRA_PEER_PROBE \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="$MODEL_NAME=$MODEL" \
    MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,0 MEMRA_SWA_RING=0 \
    MEMRA_PREFIX_CACHE_MB=512 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
    MEMRA_AFFINITY=0 MEMRA_SERVE_SPEC=0 MEMRA_CTX=2048 MEMRA_MAX_SESSIONS=8 \
    nice -n 10 ionice -c 2 -n 7 "$SERVER" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
wait_ready
curl -sf "$BASE/metrics" >"$OUT/metrics-before.json"

set +e
nice -n 10 ionice -c 2 -n 7 timeout 3600 python3 "$LANE/prefix_gate.py" \
    --base "$BASE" --model "$MODEL_NAME" --out "$OUT/exactness.jsonl" \
    --reps 3 --prefix-tokens 512 --suffix-tokens 16 --max-tokens 32 \
    --concurrency 4 --require-dual 2>&1 | tee "$OUT/gate.log"
gate_rc=${PIPESTATUS[0]}
set -e
curl -sf "$BASE/metrics" >"$OUT/metrics-after.json"
stop_server
stop_sampler
snapshot "$OUT/nvidia-smi-after.log" after

grep -E 'prefix-cache|dual-pp|PP-2|refused|CUDA_ERROR|out of memory|panicked' \
    "$OUT/server.log" >"$OUT/server-markers.log" || true
find "$OUT" -maxdepth 1 -type f ! -name SHA256SUMS ! -name driver.log -print0 \
    | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS"
echo "gate_rc=$gate_rc"
echo "lock_release=$(date -u +%FT%TZ)"
exit "$gate_rc"
