#!/usr/bin/env bash
# Local RTX 5090, Qwen3.5-9B: tensor-core decode arm (MEMRA_DECODE_TC=1) vs the dp4a exact tier,
# same binary, 16 and 32 sessions at depth 2048, interleaved both orders. Per-session token
# fingerprints dumped for the divergence count.
set -uo pipefail
H=$HOME/.cache/memra-batchspec-harness
OUT=$H/rig5090-tc
mkdir -p "$OUT"
B=$HOME/.cache/memra-decodetc-target/release/decode-batch-bench
G=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
run() { # tag tc seqs samp
  MEMRA_DECODE_TC=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=$4 timeout 1500 "$B" "$G" \
    --seqs $3 --chunk $3 --ctx 2048 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "$1: $(grep CHUNKSWEEP "$OUT/$1.log") | $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)" >> "$OUT/ab.txt"
}
for n in 16 32; do
  i=0
  for arm in dp4a tc tc dp4a dp4a tc; do
    i=$((i + 1)); t=0; [ $arm = tc ] && t=1
    run b$n-r$i-$arm $t $n greedy
  done
done
run b32-temp-dp4a 0 32 temp
run b32-temp-tc 1 32 temp
echo "AB_DONE $OUT"
