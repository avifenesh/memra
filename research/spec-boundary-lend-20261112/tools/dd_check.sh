#!/usr/bin/env bash
# CPU build host (CPU build host, sm_120a, no GPU), memra#1221 part 2: cargo check (or another cargo
# command) on the synced tree, at low weight on cores 20-31 (the CPU build host is shared).
# usage: dd_check.sh NAME CARGO_ARGS...   -> ~/.cache/memra-1221b/NAME.log, NAME.rc
set -u
R=$HOME/.cache/memra-1221b
NAME=$1; shift
export PATH="$HOME/.cargo/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
cd "$R/src" || exit 1
rm -f "$R/$NAME.rc"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo "$@" > "$R/$NAME.log" 2>&1
echo "rc=$?" > "$R/$NAME.rc"
