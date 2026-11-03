#!/usr/bin/env bash
# Runs ON the second GB10 box: build the combined research tree, then
#  (1) gdn-state-bench: batched GDN scan with fp32 vs bf16-stored state, B = 1/8/16/32;
#  (2) tensor-core GEMM decode tier (MEMRA_DECODE_TC=1) vs the dp4a exact tier, 16 and 32
#      sessions at depth 2048, interleaved both orders, per-session token fingerprints.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=claude-code-20261103T022643-3047127
OUT=$HOME/memra-run/research
mkdir -p "$OUT"
T=$HOME/.cache/memra-research-target/release
cd ~/projects/memra-research
CARGO_TARGET_DIR=$HOME/.cache/memra-research-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin gdn-state-bench > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
"$T/gdn-state-bench" > "$OUT/gdn-state-bench.txt" 2>&1
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag tc seqs samp
  MEMRA_DECODE_TC=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=$4 timeout 1800 "$T/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx 2048 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "$1: $(grep CHUNKSWEEP "$OUT/$1.log")" >> "$OUT/tc-ab.txt"
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
echo "RESEARCH_DONE $OUT"
