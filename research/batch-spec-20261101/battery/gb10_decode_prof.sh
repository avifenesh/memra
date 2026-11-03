#!/usr/bin/env bash
# Runs ON the GB10 box: plain batched decode profile for the GB10 plateau (owner GGUF 27B).
# 1. build decode-batch-bench from the synced batch-spec tree;
# 2. per (depth, B): steady tick time (CHUNKSWEEP, chunk = B up to the 16-row exact tier,
#    so B=32 runs as 2 x 16 like serving) plus MEMRA_BATCH_PHASE decomposition;
# 3. nsys kernel summary per (depth, B) for the per-kernel breakdown;
# 4. bw-roofline for the part's own streaming ceiling.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/decode-prof-$(date +%H%M)
mkdir -p "$OUT"
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin bw-roofline > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
B=$HOME/.cache/memra-batchspec-target/release
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
"$B/bw-roofline" 0 8 > "$OUT/bw-roofline.log" 2>&1
for depth in 2048 12288 32768; do
  for n in 1 4 8 16 32; do
    [ "$depth" = 32768 ] && [ "$n" = 32 ] && continue
    c=$n; [ "$c" -gt 16 ] && c=16
    tag="d${depth}-b${n}"
    MEMRA_DECODE_BATCH_CAP=16 MEMRA_BATCH_PHASE=1 MEMRA_DBB_SAMP=greedy \
      timeout 1800 "$B/decode-batch-bench" "$G" --seqs $n --chunk $c --ctx $depth --steps 16 --reps 3 \
      > "$OUT/$tag.log" 2>&1
    echo "$tag rc=$?" >> "$OUT/runs.txt"
    MEMRA_DECODE_BATCH_CAP=16 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/$tag" -f true -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
      --stats=false "$B/decode-batch-bench" "$G" --seqs $n --chunk $c --ctx $depth --steps 8 --reps 1 \
      > "$OUT/$tag-nsys.out" 2>&1
    nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/$tag.nsys-rep" > "$OUT/$tag-kern.csv" 2>/dev/null
    rm -f "$OUT/$tag.nsys-rep" "$OUT/$tag.sqlite"
  done
done
echo "PROF_DONE $OUT"
