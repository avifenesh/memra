#!/usr/bin/env bash
# Day 18 local RTX 5090 arms: hashlock, serverdoor, hashmicro through the collector (/tmp/memra-5090.lock).
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
export D18_RIG=rtx5090 D18_R=$L/rtx5090-day18 D18_TREE=/home/avifenesh/projects/wt-spill-c
export D18_CELL_SCRIPT=$L/day18-cell.sh D18_BINS=/home/avifenesh/projects/wt-spill-c/target/release
export D18_LOCK=/tmp/memra-5090.lock D18_PORT=18141 D18_MOE_ENV=""
export D18_ART=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
export D18_ART_OTHER=$D18_ART
mkdir -p "$D18_R"
echo "local start $(date -u +%FT%TZ) tree $(git -C "$D18_TREE" rev-parse HEAD)"
for cell in hashlock hashmicro serverdoor; do
  bash "$L/day18-run-cell.sh" "$cell" 1500; echo "$cell rc=$? $(date -u +%FT%TZ)"
  python3 "$D18_TREE/tools/tier-battery.py" --rig rtx5090 --validate "$D18_R/$cell" > "$D18_R/$cell-validate.log" 2>&1; echo "$cell validate rc=$?"
done
echo "local done $(date -u +%FT%TZ)"
