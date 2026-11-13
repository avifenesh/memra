#!/usr/bin/env bash
# GB10 (gb10, under a sparklock lease): build a449d3183c (o2) for 121a, then overlay k3 (the bf16 family in the
# K-segment order: dsv41_gpu.cu + dsv41_gate.rs) on the same tree and build it, then pc1 (o2 + the transport fix).
# nice 19, 10 jobs (half the cores).
set -u
R=$HOME/.cache/dsv41-speed
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
mkdir -p "$R/logs" "$R/bins/o2" "$R/bins/k3" "$R/src"
rm -f "$R/logs/build.done"
B="--bin dsv41-gate --bin dsv41-run --bin dsv41-tpbench --bin dsv41-bench"
cd "$R/src" || exit 1
tar -x -f "$R/o2-src.tar"
echo "o2 start $(date +%T)" >> "$R/logs/build.times"
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine $B > "$R/logs/build-o2.log" 2>&1; then
  echo BUILD-FAIL-o2 > "$R/logs/build.done"; exit 1
fi
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/o2/"; done
echo "o2 end $(date +%T)" >> "$R/logs/build.times"
tar -x -f "$R/k3-overlay.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/bin/dsv41_gate.rs
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine $B > "$R/logs/build-k3.log" 2>&1; then
  echo BUILD-FAIL-k3 > "$R/logs/build.done"; exit 1
fi
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/k3/"; done
echo "k3 end $(date +%T)" >> "$R/logs/build.times"
# pc1 = o2 + the transport fix (tp.rs only): back to o2's two k3 files, then the overlay
mkdir -p "$R/bins/pc1"
tar -x -f "$R/o2-src.tar" crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/bin/dsv41_gate.rs
tar -x -f "$R/pc1-overlay.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/bin/dsv41_gate.rs crates/memra-engine/src/dsv41_gpu/tp.rs
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine $B > "$R/logs/build-pc1.log" 2>&1; then
  echo BUILD-FAIL-pc1 > "$R/logs/build.done"; exit 1
fi
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/pc1/"; done
echo "pc1 end $(date +%T)" >> "$R/logs/build.times"
sha256sum "$R"/bins/*/* > "$R/bins/SHA256SUMS"
echo BUILD-DONE > "$R/logs/build.done"
