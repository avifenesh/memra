#!/usr/bin/env bash
# Runs ON spark-b23d: build memra-server for #1031 (restorable-hit preference + cold-drafter door)
# into the existing research target (no new target dir).
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-b23d-lease.txt 2>/dev/null)
OUT=$HOME/memra-run/step-1031
mkdir -p "$OUT"
SHA=$(cat ~/projects/memra-1031/SHA)
cd ~/projects/memra-1031
CARGO_TARGET_DIR=$HOME/.cache/memra-research-target nice -n 5 cargo build --release -j 16 \
  -p memra-server --bin memra-server > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
mkdir -p "$HOME/memra-run/bin-1031-$SHA"
cp "$HOME/.cache/memra-research-target/release/memra-server" "$HOME/memra-run/bin-1031-$SHA/"
echo "BUILD_DONE $SHA"
