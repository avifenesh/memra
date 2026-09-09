#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/root/target-fp4gemm
lane=research/qwen-prefill-fp4-gemm-20260909
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v \
 -Dmemra_mmq_nvfp4_w4a8=exact_gemm -Dmemra_mmq_nvfp4_w4a8_act_bytes=exact_bytes -Dmemra_mmq_nvfp4_f8f4=exact_f8f4 -Dkvalues_mxfp4=exact_lut -Dmul_mat_q_nvfp4_w4a8=exact_kernel \
 -c "$lane/exact-kernel.cu" -o "$CARGO_TARGET_DIR/exact.o"
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 "$lane/exact-bench.cu" crates/memra-engine/cu/mmq_nvfp4_f8f4.cu "$CARGO_TARGET_DIR/"{base,exact}.o -o "$CARGO_TARGET_DIR/exact-bench"
