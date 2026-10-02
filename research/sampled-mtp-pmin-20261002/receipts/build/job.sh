#!/usr/bin/env bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target"
export MEMRA_CUDA_ARCH=120a
export RUSTC_WRAPPER=
export SCCACHE_DISABLE=1
EV="$PWD/research/sampled-mtp-pmin-20261002/receipts/build"
mkdir -p "$EV"
cargo build --release -j2 -p memra-engine --bin sample_check
sha256sum target/release/sample_check target/release/run-spec target/release/memra-server > "$EV/binaries.sha256"
git rev-parse HEAD > "$EV/collector-source.txt"
git diff 2873dd4ca37faa15cd4261e7b398926cced30106 -- Cargo.toml Cargo.lock .cargo crates > "$EV/runtime-source.patch"
test ! -s "$EV/runtime-source.patch"
