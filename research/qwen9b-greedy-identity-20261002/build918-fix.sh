#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a RUST_TEST_THREADS=2
out=$PWD/research/qwen9b-greedy-identity-20261002/fix-build
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff HEAD --binary > "$out/source.diff"
cargo build --release -p memra-server --bin memra-server -p memra-engine --bin run-gen --bin run-spec --bin spec-serve-gate 2>&1 | tee "$out/build.log"
sha256sum target/release/{memra-server,run-gen,run-spec,spec-serve-gate} > "$out/binaries.sha256"
cargo test --release -p memra-engine --lib spec::prime::tests -- --test-threads=2 2>&1 | tee "$out/prime-tests.log"
cargo fmt -p memra-engine -- --check
git diff --check
