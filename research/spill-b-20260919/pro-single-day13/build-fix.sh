#!/usr/bin/env bash
# Day 13 native build receipt: memra-server release at the fix tip (lane/spill-b-20260919). No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/b-day13/build-fix
cd /root/wt-b
git checkout -q --detach refs/bundle/b13fix || { echo checkout-failed > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
cargo build --release -p memra-server > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/b-day13/bins/fix/memra-server
sha256sum /root/spill-receipts/b-day13/bins/fix/memra-server > "$out/binary.sha256"
cat "$out/exit"
