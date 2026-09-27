#!/usr/bin/env bash
# Day 13 second build receipt: the flags check tests the write-combined bit (refs/bundle/a13b,
# branch lane-a-day13). No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/a-day13
out=$R/build2
mkdir -p "$out" "$R/bins"
cd /root/wt-a
git checkout -q lane-a-day13
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
nice -n 19 cargo build --release -p memra-engine --bin tier-transfer-gate --offline -j 16 > "$out/build-gate.log" 2>&1
echo $? > "$out/exit-gate"
cp target/release/tier-transfer-gate "$R/bins/tier-transfer-gate"
sha256sum "$R/bins/tier-transfer-gate" > "$out/binary-gate.sha256"
nice -n 19 cargo test --release -p memra-engine --offline --lib tier_transfer --no-run > "$out/build-test.log" 2>&1
echo $? > "$out/exit-test"
date -u +%FT%TZ > "$out/finished.txt"
echo "gate exit $(cat "$out/exit-gate") test-build exit $(cat "$out/exit-test")"
