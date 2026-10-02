#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a RUST_TEST_THREADS=2
out=$PWD/research/cache-metrics-20261002/issue522
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff HEAD --binary > "$out/source.diff"
sha256sum /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf > "$out/model.sha256"
cargo build --release -p memra-server --bin memra-server 2>&1 | tee "$out/build.log"
sha256sum target/release/memra-server > "$out/binary.sha256"
for filter in histogram prometheus emitted_gap emission_metrics_tests stream_worker_error premature_stream_end streaming_client_disconnect committed_prefill_timeout interactive_admission_error metrics_; do
    cargo test --release -p memra-server --lib "$filter" -- --test-threads=2 2>&1 | tee "$out/cpu-$filter.log"
done
python3 -m unittest discover -s tools -p test_prometheus_metrics.py 2>&1 | tee "$out/parser-tests.log"
python3 -m py_compile tools/prometheus_metrics.py tools/cache-meter-gate.py tools/metrics-live-gate.py
cargo fmt -p memra-server -- --check
git diff --check
