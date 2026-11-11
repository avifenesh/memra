#!/usr/bin/env bash
# CPU build host (the cpu_checks.sh tree): the memra-engine lib suite again at c232c9132a with the argmax floors table's
# receipt logs in the tree (the trimmed source tree left them out; one lib test reads them).
set -u
R=$HOME/.cache/memra-specpub
export PATH="$HOME/.cargo/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
cd "$R/src" || exit 1
tar -xzf "$R/in/floor-receipts.tar.gz" -C "$R/src" || exit 1
O=$R/logs/engine-lib-rerun.txt
{
  echo "# the memra-engine lib suite at c232c9132a, with the floors table's receipts in the tree"
  nice -n 10 cargo test --release --locked -p memra-engine --lib > "$R/logs/engine-lib-rerun.log" 2>&1; rc=$?
  grep -E '^test result|FAILED|panicked' "$R/logs/engine-lib-rerun.log" | head -20
  echo "engine lib c232c9132a rc=$rc"
  echo ENGINE-LIB-DONE
} > "$O" 2>&1
