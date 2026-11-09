#!/usr/bin/env bash
# Rig: the dspark ladder panic's reference. The same cell (dspark_q38_gate, DSpark agentic
# drafter, MEMRA_DSPARK_VT=ladder, eager, backtrace) on the stack parent 9390bd3755's binary.
set -u
export PATH=/usr/bin:$PATH
S=$HOME/.cache/memra-gb10-r5/seam
B=$S/bins/9390bd3755-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DRAFT=/data/ai-ml/models/q38-dspark-agentic
CD=$S/cells-2
L=$CD/driver.log
waitfor --quiet --until "! gpulock status | grep -qi landing" --every 30s --timeout 60m >> "$L" 2>&1
waitfor --quiet --until "bash $HOME/.cache/memra-gb10-r5/wt-seam/research/gb10-decode-20261109/tools/gdn_seam_cool.sh" --every 15s --timeout 30m >> "$L" 2>&1
echo "$(date -Is) card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$L"
name="dspark_q38_gate at the stack parent 9390bd3755, DSpark agentic drafter, MEMRA_DSPARK_VT=ladder, eager, backtrace"
echo "$(date -Is) $name start; $(gpulock status 2>&1 | head -n 3 | tr '\n' ' ')" >> "$L"
gpulock run --ttl 20m --for "memra GDN packed seam removal: $name (one cell per lease)" --wait 90m -- \
  env RUST_BACKTRACE=1 MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_FULLG_DEBUG=1 \
  "$B/dspark_q38_gate" "$Q38" "$DRAFT" > "$CD/dspark-q38-base-eager-ladder-bt.log" 2>&1
echo "$(date -Is) $name rc=$? last: $(tail -n 1 "$CD/dspark-q38-base-eager-ladder-bt.log")" >> "$L"
