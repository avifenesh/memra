#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a
out=$PWD/research/cache-metrics-20261002/issue777
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
cargo --version > "$out/toolchain.txt"
rustc --version >> "$out/toolchain.txt"
nvcc --version >> "$out/toolchain.txt"
sha256sum /data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf > "$out/model.sha256"
python3 - "$out/model.sha256" <<'PY'
import sys
assert open(sys.argv[1]).read().split()[0] == 'df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf'
PY
cargo build --release -p memra-server 2>&1 | tee "$out/build.log"
sha256sum target/release/memra-server > "$out/binary.sha256"
