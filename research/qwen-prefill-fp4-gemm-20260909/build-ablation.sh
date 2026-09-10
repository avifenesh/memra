#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/root/target-fp4gemm
lane=research/qwen-prefill-fp4-gemm-20260909
for entry in '0 base 0' '1 unpack 0' '2 fold 1' '3 act 0'; do
 read -r arm label fold <<< "$entry"
 nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v -DABLATION="$arm" -DMEMRA_MMQ_FOLD_CEILING="$fold" \
 -Dmemra_mmq_nvfp4_w4a8="${label}_gemm" -Dmemra_mmq_nvfp4_w4a8_act_bytes="${label}_bytes" -Dmemra_mmq_nvfp4_f8f4="${label}_f8f4" -Dkvalues_mxfp4="${label}_lut" \
 -Dmul_mat_q_nvfp4_w4a8="${label}_kernel" -c "$lane/ablation-kernel.cu" -o "$CARGO_TARGET_DIR/$label.o"
done
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 "$lane/ablation-bench.cu" crates/memra-engine/cu/mmq_nvfp4_f8f4.cu "$CARGO_TARGET_DIR/"{base,unpack,fold,act}.o -o "$CARGO_TARGET_DIR/ablation-bench"
