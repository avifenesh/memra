#!/usr/bin/env bash
# Day 18 local build (CPU-capped scope): run-gen, run-spec, hash-micro, memra-server from this tree.
set -uo pipefail
cd /home/avifenesh/projects/wt-spill-c
export CARGO_TARGET_DIR=/home/avifenesh/projects/wt-spill-c/target
echo "build start $(date -u +%FT%TZ) tree $(git rev-parse HEAD)"
cargo build --release --keep-going -p memra-engine --bin run-gen --bin run-spec --bin hash-micro; rc1=$?
echo "engine bins rc=$rc1 $(date -u +%FT%TZ)"
cargo build --release -p memra-server --bin memra-server; rc2=$?
echo "server rc=$rc2 $(date -u +%FT%TZ)"
sha256sum target/release/run-gen target/release/run-spec target/release/hash-micro target/release/memra-server 2>&1
echo "build done rc=$((rc1|rc2)) $(date -u +%FT%TZ)"
