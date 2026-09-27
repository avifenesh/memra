#!/usr/bin/env bash
# Day 22 FIX build receipt on the local RTX 5090: the prefill-rows scope (memra#427) in the lane tree; the tightened
# continuation gate, the two-scope width walk, kernel-check and memra-server (for the hit gate). No GPU cell, no lock.
# usage: build-day22.sh [worktree] [receipt-dir]
set -uo pipefail
wt=${1:-$HOME/projects/wt-spill-b}
out=${2:-$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day22/build-fix}
# Owner rule: no local CPU saturation; the quota keeps the desktop responsive.
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
mkdir -p "$out"
cd "$wt" || { echo worktree-missing > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
CARGO_TARGET_DIR="$wt/target" run cargo build --release --offline -p memra-engine \
  -p memra-server --bin qwen-a4-continuation-gate --bin qwen-a4-width-walk --bin kernel-check --bin memra-server -j 12 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
sha256sum "$wt/target/release/qwen-a4-continuation-gate" "$wt/target/release/qwen-a4-width-walk" "$wt/target/release/kernel-check" "$wt/target/release/memra-server" > "$out/binary.sha256" 2>/dev/null || echo missing > "$out/binary.sha256"
cat "$out/exit"
