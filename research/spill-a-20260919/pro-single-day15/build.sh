#!/usr/bin/env bash
# Day 15 native build receipt: memra-server release at the lane tree (lane/spill-a-20260919 after the merge
# of origin/main a51e29abb; refs/bundle/a15), for the door's cached-destination pair (task 2, lane C's
# harness). No GPU cell, no lock. Niced and job-capped: another lane may hold the card while this builds.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/a-day15
out=$R/build
mkdir -p "$out" "$R/bins"
cd /root/wt-a
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 19 env CARGO_BUILD_JOBS=8 cargo build --release -p memra-server > "$out/build.log" 2>&1
rc=$?
echo $rc > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
[ $rc = 0 ] && cp target/release/memra-server "$R/bins/memra-server" && sha256sum "$R/bins/memra-server" > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
