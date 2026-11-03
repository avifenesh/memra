#!/usr/bin/env bash
# Runs ON the GB10 box: lever 3 (batched-tick FA on the deep walk: padded K rows + tile+2 L2
# prefetch). Build bench + gates + server; A = the lever-2 bench (step-l2b/dbb), B = this build.
# Interleaved both orders (3+3) at 16 sessions depth 12288, then pairs at 8 sessions depth
# 32768 and 32 sessions depth 12288, nsys kernel sum at 16 sessions depth 12288.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-l3
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
BS=$(cat ~/projects/memra-batchspec/BATCHSPEC_SHA)
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check \
  -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
mkdir -p "$HOME/memra-run/bin-batchspec-$BS"
cp "$T/memra-server" "$HOME/memra-run/bin-batchspec-$BS/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MEMRA_KC_ONLY=a6-split-plane,nvfp4-fused4 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
OLD=$HOME/memra-run/step-l2b/dbb
NEW=$OUT/dbb
cp "$T/decode-batch-bench" "$NEW"
run() { # tag bin seqs depth
  local ch=$3; [ "$ch" -gt 32 ] && ch=32
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$2" "$G" \
    --seqs $3 --chunk $ch --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for arm in old new new old old new; do
  i=$((i + 1)); bin=$OLD; [ $arm = new ] && bin=$NEW
  run r$i-$arm $bin 16 12288
done
run b8d32k-new $NEW 8 32768
run b8d32k-old $OLD 8 32768
run b32d12k-old $OLD 32 12288
run b32d12k-new $NEW 32 12288
MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/fa" -f true -t cuda \
  --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false "$NEW" "$G" \
  --seqs 16 --chunk 16 --ctx 12288 --steps 8 --reps 1 > "$OUT/fa-nsys.out" 2>&1
nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/fa.nsys-rep" > "$OUT/fa-kern.csv" 2>/dev/null
rm -f "$OUT/fa.nsys-rep" "$OUT/fa.sqlite"
echo "STEP_DONE $OUT bs=$BS"
