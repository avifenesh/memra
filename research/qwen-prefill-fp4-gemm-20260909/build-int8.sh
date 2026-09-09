#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/root/target-fp4gemm
mkdir -p "$CARGO_TARGET_DIR"
lane=research/qwen-prefill-fp4-gemm-20260909
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -Xptxas=-v "$lane/int8-roof.cu" -o "$CARGO_TARGET_DIR/int8-roof"
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v "$lane/int8-tiles.cu" crates/memra-engine/cu/mmq_nvfp4_f8f4.cu -o "$CARGO_TARGET_DIR/int8-tiles"
sha256sum "$CARGO_TARGET_DIR/int8-roof" "$CARGO_TARGET_DIR/int8-tiles"
