#!/bin/bash
# Late-member round fix, verification: 20 runs of darklane:: with the stress test, the full
# memra-server suite twice, rustfmt check and release all-target clippy with warnings denied.
set -u
W=$WORK
V=$W/darklane/round5/verify
F=$W/src-darklane/crates/memra-server/src/darklane.rs
export RUSTUP_HOME=$W/rust/rustup CARGO_HOME=$W/rust/cargo
export PATH=$W/rust/cargo/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$W/target-darklane TMPDIR=$W/tmp
PIN="taskset -c 24-31,56-63 nice -n 10"
cd $W/src-darklane || exit 1
mkdir -p "$V"
rm -f "$V/DONE"
crates_digest() { find crates -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }
echo "darklane.rs sha256 $(sha256sum "$F" | cut -d' ' -f1) crates/ digest $(crates_digest)" > "$V/summary.txt"
$PIN cargo test -p memra-server -j 16 --lib --no-run > "$V/build.log" 2>&1
echo "build rc=$?" >> "$V/summary.txt"
for i in $(seq 1 20); do
  $PIN cargo test -p memra-server -j 16 --lib -- darklane:: --include-ignored --nocapture > "$V/darklane-$i.log" 2>&1
  rc=$?
  echo "darklane iter $i rc=$rc $(grep -h 'test result' "$V/darklane-$i.log") $(grep -h 'parked in vfork: [0-9]' "$V/darklane-$i.log")" >> "$V/summary.txt"
done
for i in 1 2; do
  $PIN cargo test -p memra-server -j 16 > "$V/full-$i.log" 2>&1
  echo "full suite $i rc=$? $(grep -h 'test result' "$V/full-$i.log" | tr '\n' ' ')" >> "$V/summary.txt"
done
$PIN cargo fmt --all -- --check > "$V/fmt.log" 2>&1
echo "fmt rc=$?" >> "$V/summary.txt"
$PIN cargo clippy --release --all-targets -p memra-server -j 16 -- -D warnings > "$V/clippy.log" 2>&1
echo "clippy rc=$?" >> "$V/summary.txt"
echo "darklane.rs sha256 after $(sha256sum "$F" | cut -d' ' -f1)" >> "$V/summary.txt"
echo done > "$V/DONE"
