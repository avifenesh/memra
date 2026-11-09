#!/bin/bash
# Red control bound by sha256: the base runner with this change's tests, then the fix again.
# Expects $D/red/before-darklane.rs and $D/red/final-darklane.rs.
set -u
D=$WORK/darklane
F=$WORK/src-darklane/crates/memra-server/src/darklane.rs
export RUSTUP_HOME=$WORK/rust/rustup CARGO_HOME=$WORK/rust/cargo
export PATH=$WORK/rust/cargo/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$WORK/target-darklane TMPDIR=$WORK/tmp
PIN="taskset -c 24-31,56-63 nice -n 10"
cd $WORK/src-darklane || exit 1
R=$D/red
for arm in before final; do
  cp "$R/$arm-darklane.rs" "$F"
  { echo "arm=$arm darklane.rs sha256 $(sha256sum "$F" | cut -d' ' -f1)"; } > "$R/$arm.txt"
  $PIN cargo test -p memra-server -j 16 --lib -- darklane:: --include-ignored --nocapture > "$R/$arm-darklane.log" 2>&1
  echo "rc=$? $(grep -h 'test result' "$R/$arm-darklane.log")" >> "$R/$arm.txt"
  grep -h 'no acknowledgement\|owned runner assertion\|parked in vfork: [0-9]' "$R/$arm-darklane.log" >> "$R/$arm.txt"
done
cmp "$R/final-darklane.rs" "$F" && echo restored-final >> "$R/final.txt"
echo done > "$R/DONE"
