#!/usr/bin/env bash
# Build memra-server from a worktree at its current HEAD (plus any working-tree patch) and stage it.
# usage: build.sh WORKTREE LABEL
set -o pipefail
WT=${1:?worktree}; label=${2:?label}
cd "$WT" || exit 2
export CARGO_TARGET_DIR=${TARGET:-<lane>/target} MEMRA_CUDA_ARCH=120a
L=~/.cache/memra-rig-build.lock
O=<bins>; mkdir -p "$O"
sha=$(git rev-parse --short=10 HEAD)
dirty=$(git status --porcelain --untracked-files=no | wc -l)
echo "build $label sha=$sha dirty_files=$dirty $(date -Is)"
git diff --stat | tail -n 3
flock "$L" nice -n 15 cargo build --release -j 2 -p memra-server --bin memra-server 2>&1 | grep -E '^(error|warning)|Finished|^\s+-->' | tail -40
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "BUILD FAILED rc=$rc"; exit "$rc"; }
cp "$CARGO_TARGET_DIR/release/memra-server" "$O/memra-server-$label"
sha256sum "$O/memra-server-$label"
echo "BUILD_DONE $label $(date -Is)"
