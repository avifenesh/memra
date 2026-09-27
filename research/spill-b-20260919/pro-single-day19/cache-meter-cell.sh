#!/usr/bin/env bash
# Day 19 cache-metering arm on the fix binary: native /v1/completions, plain path, collector-held lock.
set -uo pipefail
R=/root/spill-receipts/b-day19
BIN=$R/bins/fix/memra-server
PORT=18191
cd /root/wt-b
CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS=meter=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=8192 MEMRA_SERVE_SPEC=0 "$BIN" > $R/cache-meter-server.log 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; wait $SPID 2>/dev/null" EXIT
for _ in $(seq 150); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -5 $R/cache-meter-server.log; exit 1; }
  sleep 2
done
sha256sum "$BIN"
python3 tools/cache-meter-gate.py http://127.0.0.1:$PORT meter --n 5 --k 256
