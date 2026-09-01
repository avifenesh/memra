#!/usr/bin/env bash
# sm_100 compile census for the B200 prep lane (glm5-b200-prep-20260901).
#
# Mirrors crates/memra-engine/build.rs at f98cfbf17 exactly (same nvcc argument
# construction per translation unit) and compiles every TU for the B200 arch,
# WITHOUT a GPU and WITHOUT cargo, so one ptxas failure cannot hide the others
# (build.rs asserts on the first failure; a census must see them all).
#
# Three arms:
#   A  build.rs-faithful sm_100a: stub substitutions honored (mmq_fp4 ->
#      mmq_fp4_stub, mmq_nvfp4_w4a8 -> stub, mmq_fp8_blk -> stub), fa3_prefill
#      gets -DMEMRA_FA3_STUB, qmatvec_gemm gets -DMEMRA_DISABLE_NATIVE_FP4=1,
#      dsv4_gpu gets -fmad=false. This is what MEMRA_CUDA_ARCH=100a cargo build
#      would attempt, TU by TU.
#   B  real-file sm_100a: the three stubbed TUs and fa3_prefill compiled from
#      their REAL sources (no stub, no -DMEMRA_FA3_STUB). This is the
#      needs-port catalog, not a build config.
#   C  compute_100 (non-a) fatbin probe, informational: build.rs does NOT
#      accept "100" (assert allows 120a|100a|90a|89), but the family arch tells
#      us which failures are `a`-suffix-feature-specific vs generic.
#
# Sequential on purpose: one nvcc at a time keeps the rig responsive (owner
# CPU-quota law) and keeps error logs unambiguous.
set -uo pipefail

cd "$(dirname "$0")/../../crates/memra-engine"
NVCC=${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}
OUTDIR=${1:-/tmp/b200-census}
mkdir -p "$OUTDIR"/{A,B,C,logs,fatbins}
TSV="$OUTDIR/census.tsv"
echo -e "arm\ttu\tkind\tstatus\tfirst_error" > "$TSV"

run() { # run <arm> <tu-label> <kind> <output> <args...>
  local arm=$1 label=$2 kind=$3 out=$4; shift 4
  local log="$OUTDIR/logs/${arm}_${label//\//_}.log"
  if "$NVCC" "$@" -o "$out" > "$log" 2>&1; then
    echo -e "${arm}\t${label}\t${kind}\tOK\t-" >> "$TSV"
  else
    local err
    err=$(grep -m1 -E 'error|Error|fatal' "$log" | tr '\t' ' ' | cut -c1-220)
    echo -e "${arm}\t${label}\t${kind}\tFAIL\t${err:-see-log}" >> "$TSV"
  fi
}

fatbins=(kernels hybrid qmatvec flash_attn qmatvec_gemm moe_router spec_sample kda)

for arch in 100a 100; do
  arm=A; [ "$arch" = 100 ] && arm=C
  GC="arch=compute_${arch},code=sm_${arch}"
  # ---- the 8 fatbins (build.rs first loop; no portable/hopper flags on 100a) ----
  for stem in "${fatbins[@]}"; do
    args=(-gencode "$GC" -O3 --fatbin)
    [ "$stem" = qmatvec_gemm ] && args+=(-DMEMRA_DISABLE_NATIVE_FP4=1)
    run "$arm" "cu/${stem}.cu" fatbin "$OUTDIR/$arm/${stem}.fatbin" "${args[@]}" "cu/${stem}.cu"
  done
  # ---- the 5 flash_attn KV-format variants ----
  for v in VQ4:0:1 VF8:0:2 KF8:1:0 KF8VQ4:1:1 KF8VF8:1:2; do
    IFS=: read -r suf k vv <<< "$v"
    run "$arm" "cu/flash_attn.cu[$suf]" fatbin "$OUTDIR/$arm/flash_attn_${suf,,}.fatbin" \
      -gencode "$GC" -O3 --fatbin -DMEMRA_KV_KFMT="$k" -DMEMRA_KV_VFMT="$vv" cu/flash_attn.cu
  done
done

# ---- static-lib TUs, arm A (build.rs-faithful, 100a) ----
GC="arch=compute_100a,code=sm_100a"
declare -A SUB=( [cu/mmq_fp4.cu]=cu/mmq_fp4_stub.cu
                 [cu/mmq_nvfp4_w4a8.cu]=cu/mmq_nvfp4_w4a8_stub.cu
                 [cu/mmq_fp8_blk.cu]=cu/mmq_fp8_blk_stub.cu )
statics=(cu/mmq_fp4.cu cu/mmq_q45k.cu cu/mmq_nvfp4_w4a8.cu cu/mmq_iq_experts.cu
         cu/mmq_q8_0.cu cu/mmq_q4_0.cu cu/fp8_prefill.cu cu/f16_prefill.cu
         cu/mmq_nvfp4_f8f4.cu cu/fa3_prefill.cu cu/moe_f16_grouped.cu
         cu/fp8_blk_dequant.cu cu/mmq_fp8_blk.cu cu/mmq_q8_0_f32acc.cu
         cu/mla_attn.cu cu/dsv4_gpu.cu)
for src in "${statics[@]}"; do
  real=$src; comp=${SUB[$src]:-$src}
  args=(-gencode "$GC" -O3 -std=c++17 --expt-relaxed-constexpr)
  [ "$src" = cu/fa3_prefill.cu ] && args+=(-DMEMRA_FA3_STUB)   # cuda_arch != 90a
  [ "$src" = cu/dsv4_gpu.cu ] && args+=(-fmad=false)
  stem=$(basename "$src" .cu)
  run A "$src" static-obj "$OUTDIR/A/${stem}.o" "${args[@]}" -c "$comp"
  # ---- arm B: the real file wherever arm A substituted or stubbed ----
  if [ "$comp" != "$src" ] || [ "$src" = cu/fa3_prefill.cu ]; then
    run B "$src" static-obj "$OUTDIR/B/${stem}.o" \
      -gencode "$GC" -O3 -std=c++17 --expt-relaxed-constexpr -c "$src"
  fi
done
# arm B extra: qmatvec_gemm WITHOUT -DMEMRA_DISABLE_NATIVE_FP4 (the real sm_120a kernel set)
run B "cu/qmatvec_gemm.cu[native-fp4]" fatbin "$OUTDIR/B/qmatvec_gemm_native.fatbin" \
  -gencode "$GC" -O3 --fatbin cu/qmatvec_gemm.cu

echo "=== census done ==="
column -t -s $'\t' "$TSV"
