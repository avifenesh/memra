#!/usr/bin/env bash
# Runs ON the GB10 box: option C (non-exact attention partition), the split-size knob
# MEMRA_FA_SPLIT forced on the batched tick vs the default ladder (64 keys at depth <=16384, 128
# above). Ticks + per-session token fingerprints at 16 x 12288, 8 x 32768, 32 x 12288.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-fasplit
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
cp "$T/decode-batch-bench" "$OUT/dbb"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag split seqs depth
  local E="MEMRA_X=0"; [ "$2" != def ] && E="MEMRA_FA_SPLIT=$2"
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 env $E "$OUT/dbb" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for s in def 128 256 512 512 256 128 def; do i=$((i+1)); run b16d12k-r$i-$s $s 16 12288; done
for s in def 256 512 512 256 def; do i=$((i+1)); run b8d32k-r$i-$s $s 8 32768; done
for s in def 256; do i=$((i+1)); run b32d12k-r$i-$s $s 32 12288; done
rm -f "$OUT/dbb"
echo "STEP_DONE $OUT"
