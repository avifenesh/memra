#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/root/target-fp4gemm
mkdir -p "$CARGO_TARGET_DIR"
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v -c crates/memra-engine/cu/mmq_nvfp4_w4a8.cu -o "$CARGO_TARGET_DIR/w4a8.o"
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v research/qwen-prefill-fp4-gemm-20260909/microbench.cu crates/memra-engine/cu/mmq_nvfp4_f8f4.cu "$CARGO_TARGET_DIR/w4a8.o" -o "$CARGO_TARGET_DIR/microbench"
sha256sum "$CARGO_TARGET_DIR/microbench"
cuobjdump -symbols "$CARGO_TARGET_DIR/microbench" | rg 'fp4gemm_quant|mul_mat_q_nvfp4' > "$CARGO_TARGET_DIR/symbols.txt"
cuobjdump -sass "$CARGO_TARGET_DIR/microbench" | rg 'OMMA|IMMA' > "$CARGO_TARGET_DIR/mma-sass.txt"
