#!/usr/bin/env bash
# Day 15 native build receipt: memra-server release at the lane tip (lane/spill-c-20260919, refs/bundle/c15). No GPU cell.
# Niced and job-capped: lane B's scored A/B holds the card while this builds; the window is recorded.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/c-day15/build
mkdir -p "$out" /root/spill-receipts/c-day15/bins
cd /root/wt-c
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 19 env CARGO_BUILD_JOBS=6 cargo build --release -p memra-server > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/c-day15/bins/memra-server
sha256sum /root/spill-receipts/c-day15/bins/memra-server > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
