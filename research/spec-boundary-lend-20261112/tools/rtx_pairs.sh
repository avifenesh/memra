#!/usr/bin/env bash
# Rig (RTX 5090 Laptop GPU, under the caller's gpulock lease), memra#1221 part 2: warm == cold pair
# cells (box/pairs_1221b.py) on one binary. usage: rtx_pairs.sh OUT_NAME CASE...
set -u
D=$HOME/.cache/memra-1221b; NAME=$1; shift
OUT=$D/runs/pairs5090/$NAME; mkdir -p "$OUT"
{ date -Is; nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,temperature.gpu --format=csv,noheader; } > "$OUT/guard.log"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
PAIR_MODEL=/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf python3 "$D/box/pairs_1221b.py" "$OUT" "$@" > "$OUT/run.log" 2>&1
rc=$?
for f in "$OUT"/*/*.log; do [ -f "$f" ] && gzip -f "$f"; done
exit $rc
