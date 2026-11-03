#!/usr/bin/env bash
# Runs ON the GB10 box: exact tensor-core b16/b32 tier. Build, kernel-check with the tier ON
# (RP BATCHED m=12/16/24/32 vs per-m MMVQ), mmvq-b16-bench hash + time ON/OFF at m=16/32,
# decode-batch-bench A/B (MEMRA_MMVQ_TC=0/1, same binary) both orders at 16 and 32 sessions.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-tc3
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check --bin mmvq-b16-bench > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MEMRA_MMVQ_TC=1 MEMRA_KC_ONLY=a6-split-plane,nvfp4-fused4 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
for m in 32 24; do
  for tc in 0 1; do
    MEMRA_MMVQ_TC=$tc timeout 600 "$T/mmvq-b16-bench" "$G" --layers 4 --m $m --iters 20 | sed "s/^/tc=$tc /" >> "$OUT/bench.txt" 2>&1
  done
done
run() { # tag tc seqs
  MEMRA_MMVQ_TC=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$T/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx ${4:-2048} --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "$1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
for n in 32; do
  i=0
  for arm in dp4a tc tc dp4a dp4a tc; do
    i=$((i + 1)); t=0; [ $arm = tc ] && t=1
    run b$n-r$i-$arm $t $n
  done
done
run d12k-tc 1 32 12288
run d12k-dp4a 0 32 12288
echo "STEP_DONE $OUT"
