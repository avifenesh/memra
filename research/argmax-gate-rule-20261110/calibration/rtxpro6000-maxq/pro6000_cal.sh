#!/usr/bin/env bash
# Runs ON <box>: the argmax margin gate's calibration cells for memra#1199 on one RTX PRO 6000
# (main 031f146690's unmodified argmax-margin-probe, the whole board-2048 prompt, eager and batch,
# both roster models). Untimed. One memra-gpu-run lease per cell with the card's box lock, cores
# 24-31/56-63, the power limit untouched. usage: pro6000_cal.sh GPU_UUID BOX_LOCK
set -u
W=$HOME/.cache/memra-cal-1199
UUID=$1; LOCK=$2
M=<models>
cd "$W" || exit 1
{
  echo "start $(date -Is)"
  nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
  sha256sum argmax-margin-probe board-2048.txt
  stat -c '%s %n' "$M/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf" "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
} > meta.txt 2>&1
run() { # NAME MODEL [ARGS...]
  local name=$1 model=$2; shift 2
  echo "$(date -Is) $name start" >> runs.log
  taskset -c 24-31,56-63 python3 "$W/memra-gpu-run" --gpus "$UUID" --receipt "$W/lease-$name" --box-lock "$LOCK" \
    --timeout 7200 --wait 600 -- "$W/argmax-margin-probe" "$M/$model" "$W/board-2048.txt" 2045 "$@" > "cal-$name.log" 2>&1
  echo "$(date -Is) $name rc=$? last: $(tail -n 1 "cal-$name.log" | cut -c1-120)" >> runs.log
}
run q38-eager Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run q38-batch Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --decode batch
run ornith-eager Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
run ornith-batch Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf --decode batch
sha256sum argmax-margin-probe >> meta.txt
echo "end $(date -Is)" >> meta.txt
echo CAL-DONE
