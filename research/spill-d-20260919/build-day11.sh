#!/usr/bin/env bash
# Day 11 native build receipt for lane D on the single-PRO clone: release gate binary, scoped
# clippy, kv/tier CPU tests. No GPU cell; the collector runs the arms afterwards.
# usage: build-day11.sh [worktree] [receipt-dir]
set -uo pipefail
wt=${1:-/root/wt-d}
out=${2:-/root/spill-receipts/d-day11/build}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd "$wt" || exit 2
mkdir -p "$out"
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
cargo build --release -p memra-engine --bin kv-tier-gate > "$out/build.log" 2>&1
echo $? > "$out/exit"
sha256sum target/release/kv-tier-gate > "$out/binary.sha256" 2>/dev/null || echo "missing" > "$out/binary.sha256"
cargo clippy --release -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings > "$out/clippy.log" 2>&1
echo $? > "$out/clippy.exit"
cargo test --release -p memra-kv -p memra-tier --offline --no-fail-fast > "$out/tests.log" 2>&1
echo $? > "$out/tests.exit"
cat "$out/exit" "$out/clippy.exit" "$out/tests.exit" | tr '\n' ' '; echo
