#!/usr/bin/env bash
# Runs ON the GB10 box: build the bf16-GDN-state research tree (server + bench), then a bench
# identity check with the arm OFF (TOKFP vs the lever-3 bench) and the arm ON (tick + TOKFP).
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-gdn
mkdir -p "$OUT" "$HOME/memra-run/bin-gdnbf16"
T=$HOME/.cache/memra-gdnbf16-target/release
cd ~/projects/memra-gdnbf16
CARGO_TARGET_DIR=$HOME/.cache/memra-gdnbf16-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
cp "$T/memra-server" "$HOME/memra-run/bin-gdnbf16/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag bin bf16 seqs depth samp
  MEMRA_GDN_STATE_BF16=$3 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=$6 timeout 1800 "$2" "$G" \
    --seqs $4 --chunk $4 --ctx $5 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "$1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
run ref-l3 $HOME/memra-run/step-l3/dbb 0 16 2048 greedy
run off $T/decode-batch-bench 0 16 2048 greedy
run on $T/decode-batch-bench 1 16 2048 greedy
run on-temp $T/decode-batch-bench 1 16 2048 temp
run off-temp $T/decode-batch-bench 0 16 2048 temp
echo "GDN_BUILD_DONE $OUT"
