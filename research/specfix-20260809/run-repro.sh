#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

if [ "${MEMRA_GPU_LOCK_HELD:-0}" != 1 ]; then
    exec flock -w "${MEMRA_GPU_LOCK_WAIT:-900}" /tmp/memra-gpu.lock \
        env MEMRA_GPU_LOCK_HELD=1 "$0" "$@"
fi

OUT=${1:?usage: run-repro.sh OUTPUT_DIR}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
DRAFT=${DRAFT:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}
ADDR=${MEMRA_REPRO_ADDR:-127.0.0.1:8177}
BASE=http://$ADDR

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

test -x target/release/memra-server
test -f "$MODEL"
test -f "$DRAFT"
if ss -ltn "sport = :${ADDR##*:}" | grep -q LISTEN; then
    echo "repro port already in use: $ADDR"
    exit 1
fi

echo "commit=$(git rev-parse HEAD)"
sha256sum target/release/memra-server "$MODEL" "$DRAFT"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader \
    > "$OUT/gpu-processes-pre.csv" 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"

SPID=
stop_server() {
    if [ -n "$SPID" ]; then
        kill "$SPID" 2>/dev/null || true
        wait "$SPID" 2>/dev/null || true
        SPID=
    fi
}
trap stop_server EXIT

start_server() {
    local label=$1 models=$2
    env -u MEMRA_COMPAT MEMRA_MODELS="$models" MEMRA_ADDR=$ADDR \
        target/release/memra-server > "$OUT/$label-server.log" 2>&1 &
    SPID=$!
    for _ in $(seq 120); do
        if curl -sf "$BASE/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    tail -20 "$OUT/$label-server.log"
    return 1
}

request() {
    local label=$1
    curl -sf -m 300 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"smoke","messages":[{"role":"user","content":"Explain what a mutex is in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}' \
        -o "$OUT/$label-response.json"
    jq -er '.choices[0].message | (.reasoning // "") + (.content // "")' \
        "$OUT/$label-response.json" > "$OUT/$label-text.txt"
}

request_native_tokens() {
    local label=$1
    curl -sf -m 300 "$BASE/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"smoke","prompt":"Explain what a mutex is in one sentence.","chat":true,"max_tokens":64,"temperature":0,"stream":false}' \
        -o "$OUT/$label-native-response.json"
    jq -er '.text' "$OUT/$label-native-response.json" > "$OUT/$label-native-text.txt"
    jq -ec '.tokens' "$OUT/$label-native-response.json" > "$OUT/$label-tokens.json"
    jq -e '.n_tokens == (.tokens | length) and .n_tokens == 64' \
        "$OUT/$label-native-response.json" >/dev/null
}

echo "== plain =="
start_server plain "smoke=$MODEL"
request plain
request_native_tokens plain
stop_server

echo "== spec =="
start_server spec "smoke=$MODEL+$DRAFT"
request spec
request_native_tokens spec
stop_server

echo "== text diff =="
if cmp -s "$OUT/plain-text.txt" "$OUT/spec-text.txt"; then
    echo "MATCH"
else
    echo "MISMATCH"
    diff -u "$OUT/plain-text.txt" "$OUT/spec-text.txt" || true
fi

echo "== native token receipt =="
cmp "$OUT/plain-text.txt" "$OUT/plain-native-text.txt"
cmp "$OUT/spec-text.txt" "$OUT/spec-native-text.txt"
if cmp -s "$OUT/plain-tokens.json" "$OUT/spec-tokens.json"; then
    echo "TOKEN_IDS_MATCH"
else
    echo "TOKEN_IDS_MISMATCH"
    diff -u "$OUT/plain-tokens.json" "$OUT/spec-tokens.json" || true
fi

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader \
    > "$OUT/gpu-processes-post.csv" 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post.csv"
