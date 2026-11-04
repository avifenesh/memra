#!/usr/bin/env bash
# Runs ON the GB10 box: dead partial-buffer memset removal in fa_decode_batch_seqs_v4.
# 1. build OLD (batch-spec 6ad5af8032 lib.rs) bench, then NEW (c86ae1ac6e) bench + kernel-check + server;
# 2. kernel-check (seqs-vs-loop + fused cells) on NEW;
# 3. 3+3 interleaved both orders at 16 sessions depth 12288, pairs at B8 d12288 and B32 d12288;
# 4. nsys kernel + memset sums at B16 d12288 on both binaries.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-memset
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
BS=$(cat ~/projects/memra-batchspec/BATCHSPEC_SHA)
cd ~/projects/memra-batchspec
L=crates/memra-engine/src/lib.rs
cp "$L" "$OUT/lib.rs.new"
cp "$HOME/memra-run/lib.rs.old" "$L"
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench > "$OUT/build-old.log" 2>&1
echo "BUILD=$?" >> "$OUT/build-old.log"
cp "$T/decode-batch-bench" "$OUT/dbb-old"
cp "$OUT/lib.rs.new" "$L"
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check \
  -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
cp "$T/decode-batch-bench" "$OUT/dbb-new"
mkdir -p "$HOME/memra-run/bin-batchspec-$BS"
cp "$T/memra-server" "$HOME/memra-run/bin-batchspec-$BS/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MEMRA_KC_FAST=1 timeout 1200 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
run() { # tag bin seqs depth
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$2" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for arm in old new new old old new; do
  i=$((i + 1)); run r$i-$arm "$OUT/dbb-$arm" 16 12288
done
run b8-old "$OUT/dbb-old" 8 12288
run b8-new "$OUT/dbb-new" 8 12288
run b8-new2 "$OUT/dbb-new" 8 12288
run b8-old2 "$OUT/dbb-old" 8 12288
run b32-new "$OUT/dbb-new" 32 12288
run b32-old "$OUT/dbb-old" 32 12288
for arm in old new; do
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/p-$arm" -f true -t cuda \
    --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false "$OUT/dbb-$arm" "$G" \
    --seqs 16 --chunk 16 --ctx 12288 --steps 8 --reps 1 > "$OUT/p-$arm-nsys.out" 2>&1
  nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/p-$arm.nsys-rep" > "$OUT/p-$arm-kern.csv" 2>/dev/null
  nsys stats -q -r cuda_gpu_mem_time_sum -f csv "$OUT/p-$arm.nsys-rep" > "$OUT/p-$arm-mem.csv" 2>/dev/null
  rm -f "$OUT/p-$arm.nsys-rep" "$OUT/p-$arm.sqlite"
done
echo "STEP_DONE $OUT bs=$BS"
