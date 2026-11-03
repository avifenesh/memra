#!/usr/bin/env bash
# Runs ON the GB10 box: raw-V attention twin A/B (MEMRA_FA_RAWV=0/1, same binary).
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-rawv
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MEMRA_FA_RAWV=1 MEMRA_KC_FAST=1 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
cp "$T/decode-batch-bench" "$OUT/dbb"
run() { # tag rawv seqs depth
  MEMRA_FA_RAWV=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$OUT/dbb" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for arm in deep rawv rawv deep deep rawv; do
  i=$((i + 1)); v=0; [ $arm = rawv ] && v=1
  run r$i-$arm $v 16 12288
done
run b8d32k-rawv 1 8 32768
run b8d32k-deep 0 8 32768
echo "STEP_DONE $OUT"
