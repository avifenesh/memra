#!/usr/bin/env bash
# Day 14 native build receipt: memra-server release at the lane tip (lane/spill-c-20260919, refs/bundle/c14). No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/c-day14/build
mkdir -p "$out" /root/spill-receipts/c-day14/bins
cd /root/wt-c
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
cargo build --release -p memra-server > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/c-day14/bins/memra-server
sha256sum /root/spill-receipts/c-day14/bins/memra-server > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
