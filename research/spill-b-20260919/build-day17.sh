#!/usr/bin/env bash
# Day 17 native build receipt on the local RTX 5090: memra-server release at the lane tip (the merge
# of origin/main; the lane differs from main only under research/), built in the lane worktree with
# its own target dir. No GPU cell, no lock. The owner's CPU quota keeps the desktop responsive.
# usage: build-day17.sh [worktree] [receipt-dir]
set -uo pipefail
wt=${1:-$HOME/projects/wt-spill-b}
out=${2:-$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day17/build-tip}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
mkdir -p "$out"
cd "$wt" || { echo worktree-missing > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git rev-parse origin/main > "$out/origin-main.txt"
git status --porcelain --untracked-files=no > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
CARGO_TARGET_DIR="$wt/target" run cargo build --release --offline -p memra-server -j 12 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
sha256sum "$wt/target/release/memra-server" > "$out/binary.sha256" 2>/dev/null || echo missing > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
