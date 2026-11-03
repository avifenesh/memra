#!/usr/bin/env bash
# Local RTX 5090, Qwen3.5-9B NVFP4 GGUF: lever 2 (one 32-row wave vs two 16-row waves, same
# binary, 32 sessions depth 2048) and lever 3 (batched-tick FA deep walk: rpsc build vs lever-3
# build, 16 sessions depth 12288).
# Interleaved both orders, 3 runs per arm; GPU temperature and clock logged per run.
set -uo pipefail
H=$HOME/.cache/memra-batchspec-harness
OUT=$H/rig5090-levers
mkdir -p "$OUT"
# the 27B GGUF runs out of memory on 24 GB at 24+ sessions and at 8 sessions x depth 12288
G=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
run() { # tag bin seqs chunk depth
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1500 "$2" "$G" \
    --seqs $3 --chunk $4 --ctx $5 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "$1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed 's/.*\[//' | md5sum | cut -c1-12) | $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)" >> "$OUT/ab.txt"
}
i=0
for arm in c16 c32 c32 c16 c16 c32; do
  i=$((i + 1)); ch=16; [ $arm = c32 ] && ch=32
  run l2-r$i-$arm $H/rig-bins/dbb-l3 32 $ch 2048
done
i=0
for arm in old new new old old new; do
  i=$((i + 1)); bin=$H/rig-bins/dbb-l3; [ $arm = old ] && bin=$H/rig-bins/dbb-rpsc
  run l3-r$i-$arm $bin 16 16 12288
done
echo "AB_DONE $OUT"
