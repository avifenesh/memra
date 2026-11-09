#!/bin/bash
# After-fix verification on <host>: repeated darklane runs (stress included), the full
# memra-server suite twice, and release all-target clippy with warnings denied.
set -u
D=$WORK/darklane
export RUSTUP_HOME=$WORK/rust/rustup CARGO_HOME=$WORK/rust/cargo
export PATH=$WORK/rust/cargo/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$WORK/target-darklane TMPDIR=$WORK/tmp
PIN="taskset -c 24-31,56-63 nice -n 10"
cd $WORK/src-darklane || exit 1
mkdir -p "$D/verify"
V=$D/verify
$PIN cargo test -p memra-server -j 16 --lib --no-run > "$V/build.log" 2>&1
echo "build rc=$?" > "$V/summary.txt"
for i in $(seq 1 20); do
  $PIN cargo test -p memra-server -j 16 --lib -- darklane:: --include-ignored --nocapture > "$V/darklane-$i.log" 2>&1
  rc=$?
  echo "darklane iter $i rc=$rc $(grep -h 'test result' "$V/darklane-$i.log") $(grep -h 'parked in vfork:' "$V/darklane-$i.log")" >> "$V/summary.txt"
done
for i in 1 2; do
  $PIN cargo test -p memra-server -j 16 > "$V/full-$i.log" 2>&1
  echo "full suite $i rc=$? $(grep -h 'test result' "$V/full-$i.log" | tr '\n' ' ')" >> "$V/summary.txt"
done
$PIN cargo clippy --release --all-targets -p memra-server -j 16 -- -D warnings > "$V/clippy.log" 2>&1
echo "clippy rc=$?" >> "$V/summary.txt"
echo done > "$V/DONE"
