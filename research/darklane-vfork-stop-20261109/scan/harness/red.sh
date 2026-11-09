#!/bin/bash
# Scan-round red control bound by sha256: the previous runner (8795e8d8c5) with this
# change's tests, then the fix. Expects $R/before-darklane.rs and $R/final-darklane.rs.
set -u
W=$WORK
R=$W/darklane/round4/red
F=$W/src-darklane/crates/memra-server/src/darklane.rs
export RUSTUP_HOME=$W/rust/rustup CARGO_HOME=$W/rust/cargo
export PATH=$W/rust/cargo/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$W/target-darklane TMPDIR=$W/tmp
PIN="taskset -c 24-31,56-63 nice -n 10"
cd $W/src-darklane || exit 1
rm -f "$R/DONE"
# Every file under crates/, by path and sha256: the code the test binary is built from.
crates_digest() { find crates -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }
for arm in before final; do
  cp "$R/$arm-darklane.rs" "$F"
  echo "arm=$arm darklane.rs sha256 $(sha256sum "$F" | cut -d' ' -f1) crates/ digest $(crates_digest)" > "$R/$arm.txt"
  $PIN cargo test -p memra-server -j 16 --lib -- darklane:: --include-ignored --nocapture > "$R/$arm-darklane.log" 2>&1
  echo "rc=$? $(grep -h 'test result' "$R/$arm-darklane.log")" >> "$R/$arm.txt"
  grep -h ' \.\.\. FAILED$\|no acknowledgement within\|job code ran after\|running at the yield\|owned runner assertion\|parked in vfork: [0-9]\|failed the job.s stop' "$R/$arm-darklane.log" >> "$R/$arm.txt"
done
cmp "$R/final-darklane.rs" "$F" && echo restored-final >> "$R/final.txt"
echo done > "$R/DONE"
