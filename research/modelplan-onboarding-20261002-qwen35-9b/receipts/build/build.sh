#!/usr/bin/env bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target"
export MEMRA_CUDA_ARCH=120a
export RUSTC_WRAPPER=
export SCCACHE_DISABLE=1
EV="$PWD/research/modelplan-onboarding-20261002-qwen35-9b/receipts/build"
mkdir -p "$EV"
git rev-parse HEAD > "$EV/source.txt"
git diff --binary > "$EV/source.patch"
rustc -Vv > "$EV/rustc.txt"
cargo build --release -j2 -p memra-cli --bin memra -p memra-server --bin memra-server -p memra-engine --bin run-spec --bin run-gen
sha256sum target/release/memra target/release/memra-server target/release/run-spec target/release/run-gen > "$EV/binaries.sha256"
sha256sum /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf > "$EV/model.sha256"
