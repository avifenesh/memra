#!/usr/bin/env bash
# Day 18 CPU proofs (no CUDA): item 1's WrongOwner refusal from a spawned thread (memra-tier bank tests,
# owner_proxy) and item 3's scale-plane and catalog refusals (memra-gguf expert_banks unit tests).
set -uo pipefail
cd /home/avifenesh/projects/wt-spill-c
export CARGO_TARGET_DIR=/home/avifenesh/projects/wt-spill-c/target-cpu
echo "cpu proofs start $(date -u +%FT%TZ) tree $(git rev-parse HEAD)"
cargo test --release -p memra-tier --offline --test bank owner_proxy -- --nocapture 2>&1; echo "memra-tier owner_proxy rc=$?"
cargo test --release -p memra-tier --offline --lib bank::owner_proxy 2>&1 | tail -30; echo "memra-tier lib owner_proxy rc=$?"
cargo test --release -p memra-gguf --offline --lib expert_banks 2>&1 | tail -30; echo "memra-gguf expert_banks rc=$?"
echo "cpu proofs done $(date -u +%FT%TZ)"
