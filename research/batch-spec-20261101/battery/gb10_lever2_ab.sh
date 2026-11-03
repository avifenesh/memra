#!/usr/bin/env bash
# Runs ON the GB10 box: lever 2 final. Build bench + gates + server, gates, then decode-batch-bench
# at 32 sessions: chunk 16 vs chunk 32 interleaved both orders (3+3) at depth 2048, one pair at
# depth 12288, nsys kernel sum at chunk 32. Then BUILD_DONE for the agentic cell.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-l2b
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
BS=$(cat ~/projects/memra-batchspec/BATCHSPEC_SHA)
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check --bin mmvq-b16-bench \
  -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
mkdir -p "$HOME/memra-run/bin-batchspec-$BS"
cp "$T/memra-server" "$HOME/memra-run/bin-batchspec-$BS/"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MEMRA_KC_ONLY=a6-split-plane,nvfp4-fused4 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc.log" 2>&1
echo "KC rc=$?" >> "$OUT/kc.log"
cp "$T/decode-batch-bench" "$OUT/dbb"
run() { # tag depth chunk
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$OUT/dbb" "$G" \
    --seqs 32 --chunk $3 --ctx $2 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "d$2 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for arm in c16 c32 c32 c16 c16 c32; do
  i=$((i + 1)); ch=16; [ $arm = c32 ] && ch=32
  run r$i-$arm 2048 $ch
done
run d12288-c32 12288 32
run d12288-c16 12288 16
MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/b32" -f true -t cuda \
  --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false "$OUT/dbb" "$G" \
  --seqs 32 --chunk 32 --ctx 2048 --steps 8 --reps 1 > "$OUT/b32-nsys.out" 2>&1
nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/b32.nsys-rep" > "$OUT/b32-kern.csv" 2>/dev/null
rm -f "$OUT/b32.nsys-rep" "$OUT/b32.sqlite"
echo "STEP_DONE $OUT bs=$BS"
