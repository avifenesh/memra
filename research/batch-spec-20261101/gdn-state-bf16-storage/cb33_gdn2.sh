#!/usr/bin/env bash
# Runs ON the GB10 box: bf16-stored GDN state arm. Build bench + server from the research tree
# (own target), then:
#  - identity: 16 sessions depth 2048 greedy, flag OFF must match fp32 (ffd722c3ca34) and flag ON
#    must match the earlier bf16-rounding emulation (73fe26285da0): same numerics, real bytes;
#  - speed: flag OFF/ON interleaved both orders at 32 sessions depth 2048 and 16 sessions
#    depth 12288.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-gdn2
mkdir -p "$OUT" "$HOME/memra-run/bin-gdnbf16"
T=$HOME/.cache/memra-gdnbf16-target/release
cd ~/projects/memra-gdnbf16
CARGO_TARGET_DIR=$HOME/.cache/memra-gdnbf16-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
cp "$T/memra-server" "$HOME/memra-run/bin-gdnbf16/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag bf16 seqs depth samp
  MEMRA_GDN_STATE_BF16=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=$5 timeout 1800 "$T/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
run id-off 0 16 2048 greedy
run id-on 1 16 2048 greedy
i=0
for arm in off on on off off on; do
  i=$((i + 1)); v=0; [ $arm = on ] && v=1
  run b32-r$i-$arm $v 32 2048 greedy
done
for arm in on off; do
  v=0; [ $arm = on ] && v=1
  run b16d12k-$arm $v 16 12288 greedy
done
echo "STEP_DONE $OUT"
