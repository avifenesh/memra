#!/usr/bin/env bash
# CPU build host (CPU build host): the patched session gate (MEMRA_GATE_LEND_ABOVE) for the RTX 5090 at
# the review head (src) and at the first head 0952fd4474 (src-old, the PR's first production code
# with the same gate patch). -> bins5090/gate-{R,old}/session-gate, gate-builds.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
L=$R/gate-builds.log; : > "$L"
b() { ( cd "$1" && CARGO_TARGET_DIR=$2 taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 12 -p memra-engine --bin session-gate ) >> "$L" 2>&1 || { echo "BUILD-FAIL $3" >> "$L"; exit 1; }
  mkdir -p "$R/bins5090/gate-$3"; cp -a "$2/release/session-gate" "$R/bins5090/gate-$3/"; sha256sum "$R/bins5090/gate-$3/session-gate" >> "$L"; }
b "$R/src" "$R/target" R
rm -rf "$R/src-old"; mkdir -p "$R/src-old" && tar -xzf "$R/old-0952.tar.gz" -C "$R/src-old" || exit 1
cp "$R/src/crates/memra-engine/src/bin/session_gate.rs" "$R/src-old/crates/memra-engine/src/bin/session_gate.rs"
b "$R/src-old" "$R/target-old" old
echo BUILD-DONE >> "$L"
