#!/usr/bin/env bash
# Day 14 native build receipt: memra-server release at one arm's source ref. No GPU cell, no lock.
# usage: build-arm.sh <arm main|fix> <ref>
set -uo pipefail
arm=${1:?arm}; ref=${2:?ref}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/b-day14
out=$R/build-$arm
mkdir -p "$out" "$R/bins/$arm"
cd /root/wt-b
git checkout -q --detach "$ref" || { echo checkout-failed > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 10 cargo build --release -p memra-server -j 16 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server "$R/bins/$arm/memra-server"
sha256sum "$R/bins/$arm/memra-server" > "$out/binary.sha256"
cat "$out/exit"
