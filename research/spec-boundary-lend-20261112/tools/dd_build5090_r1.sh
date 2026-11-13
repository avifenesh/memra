#!/usr/bin/env bash
# CPU build host (CPU build host, no GPU), memra#1221 part 2 review round 1: the RTX 5090 (sm_120a) ST
# server of the review head (src, as synced at the review head) with worker.ST-R.rs (that head's
# worker.rs plus the leak-check trim probe), after the review's red arms. Own target dir (target-st),
# cores 20-31, low weight.  -> ~/.cache/memra-1221b/bins5090/ST-R/memra-server, build5090-r1.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
L=$R/build5090-r1.log; : > "$L"
waitfor --grep 'RED-R1-DONE' "$R/red-r1.txt" --timeout 180m --quiet || { echo "BUILD-FAIL (red arms not done)" >> "$L"; exit 1; }
W=$R/src/crates/memra-server/src/worker.rs
cp "$W" "$R/worker.head-r1.keep.rs"; cp "$R/worker.ST-R.rs" "$W"
( cd "$R/src" && CARGO_TARGET_DIR=$R/target-st taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 12 -p memra-server --bin memra-server ) >> "$L" 2>&1
rc=$?
cp "$R/worker.head-r1.keep.rs" "$W"
[ $rc = 0 ] || { echo "BUILD-FAIL ST-R" >> "$L"; exit 1; }
mkdir -p "$R/bins5090/ST-R"; cp -a "$R/target-st/release/memra-server" "$R/bins5090/ST-R/"
sha256sum "$R/bins5090/ST-R/memra-server" >> "$L"
echo BUILD-DONE >> "$L"
