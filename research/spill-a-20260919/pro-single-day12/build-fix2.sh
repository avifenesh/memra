#!/usr/bin/env bash
# Integ15 review rerun: memra-server release at the lane tip after the deferred-reclaim change
# (branch lane-a-day12, refs/bundle/a12e). The base binary (origin/main be07f2d36) is unchanged
# and is not rebuilt. No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/a-day12
out=$R/build
cd /root/wt-a
git checkout -q lane-a-day12
git rev-parse HEAD > "$out/source-fix2.txt"
git status --porcelain > "$out/dirty-fix2.txt"
date -u +%FT%TZ > "$out/started-fix2.txt"
nice -n 19 cargo build --release -p memra-server --offline -j 16 > "$out/build-fix2.log" 2>&1
echo $? > "$out/exit-fix2"
cp target/release/memra-server "$R/bins/memra-server-fix2"
sha256sum "$R/bins/memra-server-fix2" > "$out/binary-fix2.sha256"
date -u +%FT%TZ > "$out/finished-fix2.txt"
echo "fix2 exit $(cat "$out/exit-fix2")"
