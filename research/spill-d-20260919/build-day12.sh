#!/usr/bin/env bash
# Day 12 native build receipt for lane D on the single-PRO clone: release gate binaries
# (kv-tier-gate and tier-transfer-gate), scoped clippy, kv/tier CPU tests. No GPU cell; the
# collector runs the arms and the transfer gate afterwards, bound to these binaries.
# usage: build-day12.sh [worktree] [receipt-dir]
set -uo pipefail
wt=${1:-/root/wt-d}
out=${2:-/root/spill-receipts/d-day12/build}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd "$wt" || exit 2
mkdir -p "$out"
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
nice -n 10 cargo build --release -p memra-engine --bin kv-tier-gate --bin tier-transfer-gate --offline -j 8 > "$out/build.log" 2>&1
echo $? > "$out/exit"
sha256sum target/release/kv-tier-gate > "$out/binary.sha256" 2>/dev/null || echo "missing" > "$out/binary.sha256"
sha256sum target/release/tier-transfer-gate > "$out/transfer-binary.sha256" 2>/dev/null || echo "missing" > "$out/transfer-binary.sha256"
nice -n 10 cargo clippy --release -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -j 8 -- -D warnings > "$out/clippy.log" 2>&1
echo $? > "$out/clippy.exit"
nice -n 10 cargo test --release -p memra-kv -p memra-tier --offline --no-fail-fast -j 8 > "$out/tests.log" 2>&1
echo $? > "$out/tests.exit"
cat "$out/exit" "$out/clippy.exit" "$out/tests.exit" | tr '\n' ' '; echo
