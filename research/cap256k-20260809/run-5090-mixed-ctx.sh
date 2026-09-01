#!/usr/bin/env bash
# Bounded local-5090 receipt for the cap256k admission lane. The build runs outside the lock;
# the live server, GPU sampling, and requests run under /tmp/memra-gpu.lock as one block.
set -euo pipefail

cd "$(dirname "$0")/../.."

ARM=${1:-before}
case "$ARM" in
    before|after) ;;
    *) echo "usage: $0 before|after" >&2; exit 2 ;;
esac

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
PORT=8187
BASE=http://127.0.0.1:$PORT
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_ROOT=${MEMRA_RECEIPT_ROOT:-research/ctxcharge-20260809/raw}
OUT=$OUT_ROOT/${STAMP}-${ARM}-mixed-ctx

test -f "$MODEL"
test -f "$DRAFT"
mkdir -p "$OUT"

git rev-parse HEAD > "$OUT/commit.txt"
stat -c '%n %s bytes' "$MODEL" "$DRAFT" > "$OUT/models.txt"
sha256sum "$MODEL" "$DRAFT" >> "$OUT/models.txt"

cargo build --release -p memra-server > "$OUT/build.log" 2>&1

exec 9>/tmp/memra-gpu.lock
flock 9

SERVER_PID=
GPU_PID=
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "${GPU_PID:-}" ]]; then
        kill "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$OUT/compute-apps-before.txt" 2>&1 || true
nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
    --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" 2>&1 &
GPU_PID=$!

CUDA_VISIBLE_DEVICES=0 \
MEMRA_COMPAT=openai \
MEMRA_MODELS="cap=$MODEL+$DRAFT" \
MEMRA_ADDR=127.0.0.1:$PORT \
MEMRA_CTX=262144 \
MEMRA_PREFIX_CACHE_MB=0 \
MEMRA_REUSE_POOL=2 \
target/release/memra-server > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 180); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    tail -n 30 "$OUT/server.log"
    exit 3
fi

set +e
timeout 900 python3 research/cap256k-20260809/run_mixed_ctx.py \
    "$BASE" "$OUT/requests.jsonl" "$ARM" "$OUT/server.log" 2>&1 | tee "$OUT/client.log"
CLIENT_RC=${PIPESTATUS[0]}
set -e

curl -sf "$BASE/metrics" > "$OUT/final-metrics.json" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$OUT/compute-apps-after.txt" 2>&1 || true
rg -n "\[admission\] request cost:|reclaim-on-defer|reclaim-on-alloc-oom|VRAM defer" \
    "$OUT/server.log" > "$OUT/admission-lines.txt" || true
rg -n "CUDA_ERROR|out of memory|panicked at|memory allocation.*failed" "$OUT/server.log" \
    > "$OUT/failure-lines.txt" || true

exit "$CLIENT_RC"
