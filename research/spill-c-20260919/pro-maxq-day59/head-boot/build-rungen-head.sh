#!/bin/bash
# Build run-gen from the moe-prefetch lane tip (the Max-Q arm commit) for the HEAD boot-line check.
# Copies the binary out inside the lock hold, because the target dir is shared with other trees.
set -u
WT=/home/<user>/.cache/memra-maxq/prereg-wt/moe-prefetch
O=/home/<user>/.cache/memra-maxq/receipts/head-boot
mkdir -p $O
cd $WT || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/home/<user>/.cache/memra-fix/target-cuda CARGO_BUILD_JOBS=2
git diff --quiet HEAD || { echo "tree dirty" > $O/build-status; exit 3; }
find crates/memra-engine/src -name '*.rs' -exec touch {} +
flock /home/<user>/.cache/memra-rig-build.lock bash -c '
  nice -n 15 cargo build --release -p memra-engine --bin run-gen > '$O'/build.log 2>&1 || exit $?
  cp $CARGO_TARGET_DIR/release/run-gen '$O'/run-gen-head'
rc=$?
{
  echo "rc=$rc tree=$(git rev-parse HEAD)"
  echo "compiled-from-lane=$(grep -c 'Compiling memra-engine .*prereg-wt/moe-prefetch' $O/build.log)"
  sha256sum $O/run-gen-head 2>/dev/null
  echo "maxq-arm-string=$(grep -a -c 'pro6000-maxq-receipt' $O/run-gen-head 2>/dev/null)"
} > $O/build-status
