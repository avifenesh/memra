#!/usr/bin/env bash
# Runs ON the GB10 box: group4 exact-width twin prototype (MEMRA_G4X=1/2/4) vs group4_b16_rp.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-g4x
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin decode-batch-bench --bin kernel-check > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for x in 2 4 1; do
  MEMRA_G4X=$x MEMRA_KC_ONLY=nvfp4-fused4 timeout 900 "$T/kernel-check" "$G" > "$OUT/kc-g4x$x.log" 2>&1
  echo "KC g4x$x rc=$?" >> "$OUT/kc-g4x$x.log"
done
cp "$T/decode-batch-bench" "$OUT/dbb"
run() { # tag g4x
  if [ "$2" = 0 ]; then unset MEMRA_G4X; else export MEMRA_G4X=$2; fi
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$OUT/dbb" "$G" \
    --seqs 16 --chunk 16 --ctx 12288 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b16 d12288 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
  unset MEMRA_G4X
}
i=0
for x in 0 2 4 1 1 4 2 0; do
  i=$((i + 1)); run r$i-g$x $x
done
for x in 0 2; do
  if [ $x = 0 ]; then unset MEMRA_G4X; else export MEMRA_G4X=$x; fi
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 nsys profile -o "$OUT/p-g$x" -f true -t cuda \
    --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false "$OUT/dbb" "$G" \
    --seqs 16 --chunk 16 --ctx 12288 --steps 8 --reps 1 > "$OUT/p-g$x-nsys.out" 2>&1
  nsys stats -q -r cuda_gpu_kern_sum -f csv "$OUT/p-g$x.nsys-rep" > "$OUT/p-g$x-kern.csv" 2>/dev/null
  rm -f "$OUT/p-g$x.nsys-rep" "$OUT/p-g$x.sqlite"
  unset MEMRA_G4X
done
rm -f "$OUT/dbb"
echo "STEP_DONE $OUT"
