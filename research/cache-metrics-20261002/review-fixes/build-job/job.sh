#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a RUST_TEST_THREADS=2
out=$PWD/research/cache-metrics-20261002/review-fixes
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff HEAD --binary > "$out/source.diff"
cargo test --release -p memra-engine --lib spec::prime::tests -- --test-threads=2 2>&1 | tee "$out/prime-tests.log"
cargo test --release -p memra-server --lib http_metrics_use_the_authenticated_batch_lane -- --test-threads=1 2>&1 | tee "$out/batch-lane-test.log"
cargo test --release -p memra-server --lib background_http_ack_is_separate -- --test-threads=1 2>&1 | tee "$out/background-test.log"
python3 tools/validation_plan.py cargo clippy --packages memra-server 2>&1 | tee "$out/clippy.log"
cargo build --release --workspace --bin memra-server --bin run-spec --example background_accounting_gate 2>&1 | tee "$out/build.log"
sha256sum target/release/memra-server target/release/run-spec target/release/examples/background_accounting_gate > "$out/binaries.sha256"
cargo fmt --all -- --check
git diff --check
