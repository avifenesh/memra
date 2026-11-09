#!/usr/bin/env bash
# Dev desk: the GDN packed seam removal's CPU checks at commit ed2e0c85ac's tree.
#   1. clippy -D warnings, all targets, memra-engine + memra-server (DOCS_RS stub build)
#   2. the memra-server suite (real nvcc 13.1 build, MEMRA_CUDA_ARCH=120a, no GPU)
#   3. the engine's env_audit unit tests (the retired-name refusal test), same build
set -u
D=$HOME/.cache/memra-gb10-r5-seam
cd "$D/src" || exit 2
rm -f "$D/job.DONE"
export PATH=$HOME/.cargo/bin:$PATH
touch crates/memra-engine/build.rs
DOCS_RS=1 CARGO_TARGET_DIR=$HOME/.cache/memra-gb10-r4-tf/target \
  nice -n 10 cargo clippy --release --all-targets -p memra-engine -p memra-server -- -D warnings > "$D/clippy.log" 2>&1
echo "clippy rc=$?" >> "$D/status"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$D/target
nice -n 10 cargo test --release --locked -p memra-server --no-fail-fast > "$D/server-suite.log" 2>&1
echo "suite rc=$?" >> "$D/status"
nice -n 10 cargo test --release --locked -p memra-engine --lib env_audit > "$D/engine-env-audit.log" 2>&1
echo "env_audit rc=$?" >> "$D/status"
echo done > "$D/job.DONE"
