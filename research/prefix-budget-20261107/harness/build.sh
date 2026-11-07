#!/usr/bin/env bash
# Build memra-server from the lane worktree at its current HEAD and stage the binary by short SHA.
# usage: build.sh [label]
set -o pipefail
cd /home/avifenesh/.cache/memra-prefix-budget/wt || exit 2
export CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-prefix-budget/target MEMRA_CUDA_ARCH=120a
L=/home/avifenesh/.cache/memra-rig-build.lock
O=/home/avifenesh/.cache/memra-prefix-budget/bins; mkdir -p "$O"
sha=$(git rev-parse --short=10 HEAD)
label=${1:-$sha}
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "build $label sha=$sha dirty_files=$dirty $(date -Is)"
flock "$L" nice -n 15 cargo build --release -j 2 -p memra-server --bin memra-server 2>&1 | grep -E '^(error|warning: unused)|Finished|nvcc from' | tail -5
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "BUILD FAILED rc=$rc"; exit "$rc"; }
cp "$CARGO_TARGET_DIR/release/memra-server" "$O/memra-server-$label"
sha256sum "$O/memra-server-$label"
echo BUILD_DONE "$label"
