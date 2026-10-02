#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a RUST_TEST_THREADS=2
out=$PWD/research/cache-metrics-20261002/rig-subset
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff HEAD --binary > "$out/source.diff"
cargo test --release -p memra-server --lib metrics -- --test-threads=2 2>&1 | tee "$out/metrics-tests.log"
cargo clippy --release -p memra-server --lib -- -D warnings 2>&1 | tee "$out/clippy.log"
cargo build --release -p memra-server --bin memra-server 2>&1 | tee "$out/build.log"
sha256sum target/release/memra-server > "$out/binary.sha256"
cargo fmt --all -- --check
git diff --check
python3 -m unittest discover -s tools -p test_prometheus_metrics.py 2>&1 | tee "$out/parser-tests.log"
python3 -m unittest discover -s tools -p test_cache_meter_gate.py 2>&1 | tee "$out/cache-gate-tests.log"
