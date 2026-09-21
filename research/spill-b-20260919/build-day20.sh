#!/usr/bin/env bash
# Day 20 native build receipt on the local RTX 5090: memra-server release in the DETACHED worktree at
# the pre-registration ref 9466b8912 (the two-arm binary and the A/B harness exist only there) with
# the day-18 and day-19 capture-alignment commits cherry-picked on top (269178070, 49d79b946,
# fa5d7a014; DAY20.md records the picks and the two tool-file conflict resolutions). Own target dir,
# never on the lane. No GPU cell, no lock.
# usage: build-day20.sh [worktree] [receipt-dir]
set -uo pipefail
wt=${1:-$HOME/projects/wt-spill-b-ab}
out=${2:-$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day20/build-tip}
# Owner rule: no local CPU saturation; the quota keeps the desktop responsive.
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
mkdir -p "$out"
cd "$wt" || { echo worktree-missing > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git log --format='%H %s' -4 > "$out/source-log.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
CARGO_TARGET_DIR="$wt/target" run cargo build --release --offline -p memra-server -j 12 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
sha256sum "$wt/target/release/memra-server" > "$out/binary.sha256" 2>/dev/null || echo missing > "$out/binary.sha256"
cat "$out/exit"
