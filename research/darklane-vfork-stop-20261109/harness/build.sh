#!/bin/bash
# Build (and optionally run) memra-server tests on <host> with the private toolchain.
# Usage: build.sh LABEL [cargo test args after --]
set -u
LABEL=$1; shift
export RUSTUP_HOME=$WORK/rust/rustup CARGO_HOME=$WORK/rust/cargo
export PATH=$WORK/rust/cargo/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$WORK/target-darklane TMPDIR=$WORK/tmp
cd $WORK/src-darklane || exit 1
LOG=$WORK/darklane/$LABEL.log
echo "start $(date -Is) args: $*" > "$LOG"
taskset -c 24-31,56-63 nice -n 10 cargo test -p memra-server -j 16 "$@" >> "$LOG" 2>&1
rc=$?
echo "exit $rc $(date -Is)" >> "$LOG"
echo "$rc" > $WORK/darklane/$LABEL.rc
