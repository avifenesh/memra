#!/usr/bin/env bash
# Rig: the review's mixed-width verify-graph cell. dspark_q38_gate's one-shot replays verify
# graphs only on its deferred path (a device draft chain plus the resident embed table), which a
# DFlash2 drafter never takes. The DSpark agentic drafter with MEMRA_DSPARK_VT=ladder does: the
# reactive accepted+1 ladder puts mixed widths through one pool. ALL EXACT required;
# MEMRA_DSPARK_FULLG_DEBUG=1 prints the first graph-walk round (proof the pool was armed).
set -u
export PATH=/usr/bin:$PATH
S=$HOME/.cache/memra-gb10-r5/seam
W=$HOME/.cache/memra-gb10-r5/wt-seam
T=$S/bins/0a7d83bf65-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DRAFT=/data/ai-ml/models/q38-dspark-agentic
cd "$W" || exit 2
CD=$S/cells-2
L=$CD/driver.log
waitfor --quiet --until "! gpulock status | grep -qi landing" --every 30s --timeout 60m >> "$L" 2>&1
waitfor --quiet --until "bash $HOME/.cache/memra-gb10-r5/wt-seam/research/gb10-decode-20261109/tools/gdn_seam_cool.sh" --every 15s --timeout 30m >> "$L" 2>&1
echo "$(date -Is) card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$L"
name="dspark_q38_gate, DSpark agentic drafter, MEMRA_DSPARK_VT=ladder, MEMRA_DSPARK_VERIFY_GRAPH=1"
echo "$(date -Is) $name start; $(gpulock status 2>&1 | head -n 3 | tr '\n' ' ')" >> "$L"
gpulock run --ttl 20m --for "memra GDN packed seam removal: $name (one cell per lease)" --wait 90m -- \
  env RUST_BACKTRACE=1 MEMRA_DSPARK_VERIFY_GRAPH=1 MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_FULLG_DEBUG=1 \
  "$T/dspark_q38_gate" "$Q38" "$DRAFT" > "$CD/dspark-q38-vg-ladder-bt.log" 2>&1
echo "$(date -Is) $name rc=$? last: $(tail -n 1 "$CD/dspark-q38-vg-ladder-bt.log")" >> "$L"
