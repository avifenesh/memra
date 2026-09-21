#!/usr/bin/env bash
# Day 19 native build receipt on the target card: memra-server release at one source ref, fetched from
# the pushed lane branch (the checkout is a public-repo clone; no token). No GPU cell, no lock.
# usage: build-arm.sh <arm> <ref>
set -uo pipefail
arm=${1:?arm}; ref=${2:?ref}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/b-day19
out=$R/build-$arm
mkdir -p "$out" "$R/bins/$arm"
cd /root/wt-b
git fetch -q origin "+refs/heads/lane/spill-b-20260919:refs/remotes/origin/lane/spill-b-20260919" || { echo fetch-failed > "$out/exit"; exit 1; }
git checkout -q --detach "$ref" || { echo checkout-failed > "$out/exit"; exit 1; }
git rev-parse HEAD > "$out/source.txt"
git status --porcelain --untracked-files=no > "$out/dirty.txt"
date -u +%FT%TZ > "$out/started.txt"
nice -n 10 cargo build --release -p memra-server -j 16 > "$out/build.log" 2>&1
echo $? > "$out/exit"
date -u +%FT%TZ > "$out/finished.txt"
cp target/release/memra-server "$R/bins/$arm/memra-server"
sha256sum "$R/bins/$arm/memra-server" > "$out/binary.sha256"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
cat "$out/exit"
