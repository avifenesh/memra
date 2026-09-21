#!/usr/bin/env bash
# Day 16 native build receipt: memra-server release plus the memra-server test binary at the lane tip
# (lane/spill-c-20260919, refs/bundle/c16r, Option C review fixes). No GPU cell. Niced and job-capped: another lane
# may hold the card while this builds.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/c-day16-review/build
mkdir -p "$out" /root/spill-receipts/c-day16-review/bins
cd /root/wt-c
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 19 env CARGO_BUILD_JOBS=8 cargo build --release -p memra-server > "$out/build.log" 2>&1
rc=$?
nice -n 19 env CARGO_BUILD_JOBS=8 cargo test --release -p memra-server --offline --no-run > "$out/test-build.log" 2>&1 || rc=$?
echo $rc > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/c-day16-review/bins/memra-server
sha256sum /root/spill-receipts/c-day16-review/bins/memra-server > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
