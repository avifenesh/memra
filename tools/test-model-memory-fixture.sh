#!/bin/sh
# Compile the actual fixture/contract and arithmetic modules without the CUDA engine crate.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$repo/target"
scratch=$(mktemp -d "$repo/target/model-memory-fixture-src.XXXXXX")
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
cat > "$scratch/Cargo.toml" <<TOML
[package]
name = "memra-memory-fixture-cpu"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
memra-gguf = { path = "$repo/crates/memra-gguf" }
memra-reference = { path = "$repo/crates/memra-reference" }
sha2 = "0.10"
[lib]
path = "lib.rs"
TOML
cat > "$scratch/lib.rs" <<RS
#[cfg(test)]
#[path = "$repo/crates/memra-engine/src/model_memory_plan.rs"]
mod model_memory_plan;
#[cfg(test)]
#[path = "$repo/crates/memra-engine/src/model_memory_fixture.rs"]
mod model_memory_fixture;
RS
cp "$repo/Cargo.lock" "$scratch/Cargo.lock"
CARGO_TARGET_DIR="$repo/target/model-memory-fixture-cpu" cargo test \
  --manifest-path "$scratch/Cargo.toml" --offline --lib -- --test-threads=1 --nocapture
