#!/usr/bin/env bash
# Runs ON the GB10 box: combined research build (exact TC b32 tier + bf16-stored GDN state flag).
# 1. build bench + server into memra-batchspec-target;
# 2. identity: flag off at 16 sessions depth 2048 must match fp32 (ffd722c3ca34);
# 3. profile the 8- and 16-row ticks at depth 12288 (the served cell's median turn) on the
#    default (fp32) path and with the flag on: tick + nsys kernel summary.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-combo
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
BS=$(cat ~/projects/memra-batchspec/BATCHSPEC_SHA)
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
mkdir -p "$HOME/memra-run/bin-batchspec-$BS"
cp "$T/memra-server" "$HOME/memra-run/bin-batchspec-$BS/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag bf16 seqs depth
  MEMRA_GDN_STATE_BF16=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$T/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b$3 d$4 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
prof() { # tag bf16 seqs depth
  MEMRA_GDN_STATE_BF16=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/$1" -f true -t cuda \
    --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false "$T/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 8 --reps 1 > "$OUT/$1-nsys.out" 2>&1
  nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/$1.nsys-rep" > "$OUT/$1-kern.csv" 2>/dev/null
  rm -f "$OUT/$1.nsys-rep" "$OUT/$1.sqlite"
}
run id-off 0 16 2048
for n in 8 16; do
  run d12288-b$n 0 $n 12288
  prof d12288-b$n 0 $n 12288
  run d12288-b$n-bf16 1 $n 12288
  prof d12288-b$n-bf16 1 $n 12288
done
echo "STEP_DONE $OUT bs=$BS"
