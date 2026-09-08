#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/root/target MEMRA_CUDA_ARCH=120a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd /root/memra-release-v01340
run() {
  local label=$1; shift
  {
    git -c safe.directory=/root/memra-release-v01340 rev-parse HEAD
    git -c safe.directory=/root/memra-release-v01340 status --short | head
    sha256sum /root/target/release/run-spec /root/target/release/kernel-check /root/target/release/argmax-margin-probe /root/target/release/memra-server
    sha256sum crates/memra-reference/src/lib.rs
    printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'
    "$@"
    echo 'EXIT=0'
  } > "/root/release-v01340-work/$label.log" 2>&1
  echo "$label PASS"
}
run corrected-format cargo fmt --all -- --check
run corrected-model-plan-tests env MEMRA_GGUF_SKIP_BUDGET=12 python3 tools/skip-census.py run --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 130 -- cargo test --release -p memra-gguf -p memra-tokenizer -p memra-validate -p memra-sampling -p memra-reference --lib
run corrected-tanh-control env LD_PRELOAD=/root/release-v01340-work/tanh-control.so /root/target/release/deps/memra_reference-6c4392a5799681e5 --exact tests::dense_gemma_executes_scaled_parallel_residual_and_k_as_v --nocapture
run corrected-clippy cargo clippy --release --all-targets -- -D warnings
