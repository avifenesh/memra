#!/usr/bin/env bash
# Day 15 review build receipt: memra-server release at the lane tip after the PR #599 review fixes (refs/bundle/c15r). No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/c-day15-review/build
mkdir -p "$out" /root/spill-receipts/c-day15-review/bins
cd /root/wt-c
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 19 env CARGO_BUILD_JOBS=8 cargo build --release -p memra-server > "$out/build.log" 2>&1
rc=$?
# The two GPU unit cells run under the rig lock; their test binary compiles here, outside it.
nice -n 19 env CARGO_BUILD_JOBS=8 cargo test --release -p memra-server --offline --no-run > "$out/test-build.log" 2>&1 || rc=$?
echo $rc > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/c-day15-review/bins/memra-server
sha256sum /root/spill-receipts/c-day15-review/bins/memra-server > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
