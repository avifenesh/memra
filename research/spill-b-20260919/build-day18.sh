#!/usr/bin/env bash
# Day 18 native build receipt on the local RTX 5090: memra-server release at one source ref, in one
# worktree, with that worktree's own target dir. No GPU cell, no lock. The owner's CPU quota keeps the
# desktop responsive; the base build (a fresh target dir, so every crate and the CUDA kernels compile)
# runs at 800% beside the fix build's 800%, together under the 24-core rig's 2400%.
# usage: build-day18.sh <arm base|fix> <worktree> <receipt-dir> [cpu-quota-pct]
set -uo pipefail
arm=${1:?arm}; wt=${2:?worktree}; out=${3:?receipt-dir}; quota=${4:-800}
run() { systemd-run --user --scope -q -p "CPUQuota=${quota}%" -p MemoryMax=28G "$@"; }
mkdir -p "$out"
cd "$wt" || { echo worktree-missing > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git rev-parse origin/main > "$out/origin-main.txt" 2>/dev/null
git status --porcelain --untracked-files=no > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
CARGO_TARGET_DIR="$wt/target" run cargo build --release --offline -p memra-server -j 12 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
mkdir -p "$HOME/projects/wt-spill-b/target/bins/$arm"
cp "$wt/target/release/memra-server" "$HOME/projects/wt-spill-b/target/bins/$arm/memra-server" 2>/dev/null
sha256sum "$HOME/projects/wt-spill-b/target/bins/$arm/memra-server" > "$out/binary.sha256" 2>/dev/null || echo missing > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
