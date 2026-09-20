#!/bin/sh
# CPU-only tests of the production ownership arithmetic and reclaim ordering.
# Neither a CUDA build nor model/hardware qualification.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
cat > "$scratch/runner.rs" <<RS
#[path = "$repo/crates/memra-engine/src/model_memory_plan.rs"]
mod model_memory_plan;
#[path = "$repo/crates/memra-server/src/worker/device_memory.rs"]
mod device_memory;
RS
rustc --version
shasum -a 256 "$repo/crates/memra-engine/src/model_memory_plan.rs" "$repo/crates/memra-server/src/worker/device_memory.rs"
rustc --edition 2024 --test "$scratch/runner.rs" -o "$scratch/tests"
shasum -a 256 "$scratch/tests"
"$scratch/tests" --nocapture
