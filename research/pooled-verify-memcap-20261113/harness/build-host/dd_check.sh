#!/usr/bin/env bash
# CPU build host (CPU host, no GPU), memra#1266: one cargo command on <lane>/src at low
# weight on cores 20-31, own target dir (target-test, apart from the release arms' target).
# usage: dd_check.sh NAME CARGO_ARGS...   -> <lane>/NAME.log, NAME.rc
set -u
R=<lane>
NAME=$1; shift
export PATH="$HOME/.cargo/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$R/target-test}
cd "$R/src" || exit 1
rm -f "$R/$NAME.rc"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo "$@" > "$R/$NAME.log" 2>&1
echo "rc=$?" > "$R/$NAME.rc"
