#!/usr/bin/env bash
# Runs on the Spark under a sparklock lease. Review of memra #1312, phase B: the teeth check.
# The head's tree with the ring kernel templated on a mutation id (REVIEW_MUT, 0 = the head's
# kernel text) and four more ring shapes in the gate (K 768, 1024, 1280 at (2, 30); (8, 30, 5120)).
# One build, then one gate run per mutation. Writes only under ~/.cache/review-1312.
# Done-file out/b.txt, marker B-DONE.
set -u
R=$HOME/.cache/review-1312
D=$HOME/.cache/dsv41-speed/data
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out
mkdir -p "$O" "$R/bins/mut"
: > "$O/b.txt"
done_() { echo "B-DONE $(date +%T)" >> "$O/b.txt"; exit "$1"; }
cd "$R/src" || { echo "no src" >> "$O/b.txt"; done_ 1; }
cp "$R/mut/dsv41_gpu.cu" crates/memra-engine/cu/dsv41_gpu.cu
cp "$R/mut/dsv41_gate.rs" crates/memra-engine/src/bin/dsv41_gate.rs
echo "mut build start $(date +%T)" >> "$O/b.txt"
nice -n 19 timeout 20m cargo build --release --locked -j 16 -p memra-engine --bin dsv41-gate > "$O/build-mut.log" 2>&1
rc=$?
echo "mut build rc $rc $(date +%T)" >> "$O/b.txt"
[ $rc = 0 ] || { grep -E "error" -B2 -A12 "$O/build-mut.log" | head -60 >> "$O/b.txt"; done_ 1; }
cp -a "$R/target/release/dsv41-gate" "$R/bins/mut/"
export MEMRA_FULL_PREC=1
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> "$O/b.txt" 2>&1
run() { # mut, pass
  local tag=mut$1-$2 rc
  REVIEW_MUT=$1 REVIEW_TRACE=${3:-0} timeout 5m "$R/bins/mut/dsv41-gate" kernels "$D/fixtures-gpu/rand" --out "$O/kernels-$tag.json" > "$O/kernels-$tag.log" 2>&1
  rc=$?
  echo "== REVIEW_MUT=$1 pass $2 rc $rc: $(grep -ao '[0-9]* checks, [0-9]* failed' "$O/kernels-$tag.log" | tail -1), $(grep -ac '^FAIL' "$O/kernels-$tag.log") FAIL lines" >> "$O/b.txt"
  grep -a "^FAIL" "$O/kernels-$tag.log" | cut -c1-150 >> "$O/b.txt"
}
run 0 trace 1
grep -a "^REVIEW" "$O/kernels-mut0-trace.log" | sort | uniq -c >> "$O/b.txt"
for pass in 1 2 3; do
  for m in 0 1 2 3 4 5 6; do run $m $pass; done
done
done_ 0
