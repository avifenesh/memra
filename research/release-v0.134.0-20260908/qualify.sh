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
    printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'
    "$@"
    echo 'EXIT=0'
  } > "/root/release-v01340-work/$label.log" 2>&1
  echo "$label PASS"
}
run format cargo fmt --all -- --check
run release-guard tools/release-guard.sh v0.134.0
run changelog bash tools/changelog.sh v0.133.0
run battery tools/release-battery.sh
run squeeze bash /root/release-v01340-work/squeeze.sh
run model-hashes sha256sum /data/qual/models/orn-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf /data/qual/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run engine-skip-census python3 tools/skip-census.py verify --crate memra-engine
run engine-tests env MEMRA_ENGINE_SKIP_BUDGET=0 python3 tools/skip-census.py run --budget-var MEMRA_ENGINE_SKIP_BUDGET --min-passed 300 -- cargo test --release -p memra-engine --lib
run server-tests cargo test --release -p memra-server
run model-plan-tests env MEMRA_GGUF_SKIP_BUDGET=12 python3 tools/skip-census.py run --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 130 -- cargo test --release -p memra-gguf -p memra-tokenizer -p memra-validate -p memra-sampling -p memra-reference --lib
run dsv4-gate-tests cargo test --release -p memra-engine --bin dsv4_tp_ep_sampled_perf_gate
run clippy cargo clippy --release --all-targets -- -D warnings
