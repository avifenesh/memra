#!/usr/bin/env bash
# Rig: the served DFlash route's verify graphs (default ON on the serve route), where the window
# ladder puts mixed widths through one pool: tools/dspark-serve-smoke.sh on Qwen3.8-27B with the
# DFlash2 drafter, this branch's memra-server (0a7d83bf65). ALL GREEN required, and the server
# log's `[dspark-vg] serve pool ENGAGED` line proves the pool ran.
set -u
export PATH=/usr/bin:$PATH
S=$HOME/.cache/memra-gb10-r5/seam
W=$HOME/.cache/memra-gb10-r5/wt-seam
T=$S/bins/0a7d83bf65-120a
Q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DRAFT=/data/ai-ml/models/q38-dflash2
cd "$W" || exit 2
CD=$S/cells-2
L=$CD/driver.log
waitfor --quiet --until "! gpulock status | grep -qi landing" --every 30s --timeout 60m >> "$L" 2>&1
waitfor --quiet --until "bash $HOME/.cache/memra-gb10-r5/wt-seam/research/gb10-decode-20261109/tools/gdn_seam_cool.sh" --every 15s --timeout 30m >> "$L" 2>&1
echo "$(date -Is) card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$L"
name="dspark-serve-smoke on Qwen3.8-27B + DFlash2, serve-route verify graphs (default ON)"
echo "$(date -Is) $name start; $(gpulock status 2>&1 | head -n 3 | tr '\n' ' ')" >> "$L"
gpulock run --ttl 25m --for "memra GDN packed seam removal: $name (one cell per lease)" --wait 90m -- \
  bash "$W/tools/dspark-serve-smoke.sh" "$Q38" "$DRAFT" "$T/memra-server" "$CD/serve-smoke" > "$CD/serve-smoke.log" 2>&1
echo "$(date -Is) $name rc=$? last: $(tail -n 1 "$CD/serve-smoke.log")" >> "$L"
