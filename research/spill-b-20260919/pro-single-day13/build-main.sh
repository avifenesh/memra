#!/usr/bin/env bash
# Day 13 native build receipt: memra-server release at main ea08bc7f8 (the red base). No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/root/spill-receipts/b-day13/build-main
cd /root/wt-b
git checkout -q --detach ea08bc7f8 || { echo checkout-failed > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
cargo build --release -p memra-server > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server /root/spill-receipts/b-day13/bins/main/memra-server
sha256sum /root/spill-receipts/b-day13/bins/main/memra-server > "$out/binary.sha256"
cat "$out/exit"
