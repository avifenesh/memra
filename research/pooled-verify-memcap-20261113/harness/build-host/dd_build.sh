#!/usr/bin/env bash
# CPU build host (CPU build host, no GPU), memra#1266: build the RTX 5090 (sm_120a) memra-server from
# <lane>/src as it stands, into <lane>/target, and copy it to
# <lane>/bins5090/ARM. Cores 20-31, nice 15, idle-class IO weight.
# usage: dd_build.sh ARM   -> build-ARM.log, build-ARM.rc
set -u
ARM=$1
R=<lane>
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
L=$R/build-$ARM.log
rm -f "$R/build-$ARM.rc"
{ date -Is; echo "arm $ARM"; } > "$L"
( cd "$R/src" && CARGO_TARGET_DIR=$R/target taskset -c 20-31 nice -n 15 ionice -c2 -n7 \
    cargo build --release --locked -j 12 -p memra-server --bin memra-server ) >> "$L" 2>&1
rc=$?
if [ $rc = 0 ]; then
  mkdir -p "$R/bins5090/$ARM"
  cp -a "$R/target/release/memra-server" "$R/bins5090/$ARM/"
  sha256sum "$R/bins5090/$ARM/memra-server" >> "$L"
fi
date -Is >> "$L"
echo $rc > "$R/build-$ARM.rc"
