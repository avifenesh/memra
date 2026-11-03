#!/usr/bin/env bash
# Local RTX 5090: boot memra-server on Qwen3.5-9B with 32 sessions, record the wave-probe line,
# stop the server.
set -uo pipefail
H=$HOME/.cache/memra-batchspec-harness
OUT=$H/rig5090-probe
mkdir -p "$OUT"
export MEMRA_MODELS="q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
export MEMRA_ADDR=127.0.0.1:8093 MEMRA_MAX_SESSIONS=32 MEMRA_CTX=32768 MEMRA_COMPAT=openai
"$HOME/.cache/memra-batchspec-target/release/memra-server" > "$OUT/serve.log" 2>&1 &
pid=$!
waitfor --file "$OUT/serve.log" --grep 'decode wave cap|\[wave-probe\]' --timeout 15m --quiet
waitfor --file "$OUT/serve.log" --grep 'decode wave cap' --timeout 5m --quiet
kill $pid
waitfor --gone $pid --timeout 2m --quiet
grep -E '^\[wave-probe\]|decode wave cap' "$OUT/serve.log" > "$OUT/boot-lines.txt"
echo "PROBE_DONE"
