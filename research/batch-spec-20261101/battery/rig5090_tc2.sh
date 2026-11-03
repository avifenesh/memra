#!/usr/bin/env bash
# Local RTX 5090, Qwen3.5-9B: exact tensor-core b32 tier, forced on/off on one binary.
# kernel-check rp gates with the tier on, then decode-batch-bench 32 sessions depth 2048 3+3.
set -uo pipefail
H=$HOME/.cache/memra-batchspec-harness
OUT=$H/rig5090-tc2
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
G=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
MEMRA_MMVQ_TC=1 MEMRA_KC_ONLY=a6-split-plane,nvfp4-fused4 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
i=0
for arm in dp4a tc tc dp4a dp4a tc; do
  i=$((i + 1)); t=0; [ $arm = tc ] && t=1
  MEMRA_MMVQ_TC=$t MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1500 "$T/decode-batch-bench" "$G" \
    --seqs 32 --chunk 32 --ctx 2048 --steps 16 --reps 3 > "$OUT/r$i-$arm.log" 2>&1
  echo "r$i $arm: $(grep CHUNKSWEEP "$OUT/r$i-$arm.log") $(grep TOKFP "$OUT/r$i-$arm.log" | sed 's/.*\[//' | md5sum | cut -c1-12) | $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)" >> "$OUT/ab.txt"
done
echo "AB_DONE $OUT"
