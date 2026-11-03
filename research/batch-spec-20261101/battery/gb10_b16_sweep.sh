#!/usr/bin/env bash
# Runs ON the GB10 box: b16 schedule sweep (mmvq-b16-bench), variants interleaved, 3 rounds.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/sweep
mkdir -p "$OUT"
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin mmvq-b16-bench > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
B=$HOME/.cache/memra-batchspec-target/release/mmvq-b16-bench
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for round in 1 2 3; do
  for v in rp rpsc rpscp rpsc4 rpscs rpcs8 rpcs4; do
    MEMRA_BVAR=$v timeout 600 "$B" "$G" --layers 4 --m 16 --iters 20 >> "$OUT/sweep.txt" 2>> "$OUT/err.txt"
  done
done
for v in rp rpsc rpscp rpsc4 rpscs rpcs8 rpcs4; do
  MEMRA_BVAR=$v timeout 600 "$B" "$G" --layers 4 --m 12 --iters 20 >> "$OUT/sweep-m12.txt" 2>> "$OUT/err.txt"
done
echo "SWEEP_DONE"
