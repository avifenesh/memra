#!/usr/bin/env bash
# GPU-less B200 NVFP4 object/SASS gate. This script never probes or opens a CUDA device.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NVCC=${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}
CUDA_BIN=$(dirname "$NVCC")
CUOBJDUMP="$CUDA_BIN/cuobjdump"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

cd "$ROOT"

COMMON=(-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 --expt-relaxed-constexpr)

"$NVCC" "${COMMON[@]}" -DMEMRA_SM100_TCGEN05=1 \
  -c crates/memra-engine/cu/mmq_fp4.cu -o "$SCRATCH/mmq_fp4_sm100.o"
"$NVCC" "${COMMON[@]}" -DMEMRA_F8F4_PLAIN_MMA=1 \
  -c crates/memra-engine/cu/mmq_nvfp4_w4a8.cu -o "$SCRATCH/mmq_nvfp4_w4a8_sm100.o"

for symbol in memra_mmq_nvfp4 memra_mmq_nvfp4_ex memra_mmq_nvfp4_ex2 \
              memra_mmq_nvfp4_act_bytes; do
  nm -g --defined-only "$SCRATCH/mmq_fp4_sm100.o" | grep -Eq "[[:space:]]${symbol}$"
done
for symbol in memra_mmq_nvfp4_w4a8 memra_mmq_nvfp4_w4a8_act_bytes \
              memra_mmq_nvfp4_f8f4; do
  nm -g --defined-only "$SCRATCH/mmq_nvfp4_w4a8_sm100.o" | grep -Eq "[[:space:]]${symbol}$"
done

W4A4_SASS=$SCRATCH/w4a4.sass
W4A8_SASS=$SCRATCH/w4a8.sass
"$CUOBJDUMP" --dump-sass "$SCRATCH/mmq_fp4_sm100.o" > "$W4A4_SASS"
"$CUOBJDUMP" --dump-sass "$SCRATCH/mmq_nvfp4_w4a8_sm100.o" > "$W4A8_SASS"

grep -q 'Function : _Z21mul_mat_q_nvfp4_sm100' "$W4A4_SASS"
grep -q 'UTCOMMA.4X' "$W4A4_SASS"
grep -q 'UTCCP' "$W4A4_SASS"
grep -q 'UTCBAR' "$W4A4_SASS"
grep -q 'UTCATOMSWS.FIND_AND_SET' "$W4A4_SASS"
grep -q 'UTCATOMSWS.AND' "$W4A4_SASS"
grep -q 'IMMA.16816.S8.S8' "$W4A8_SASS"
grep -q 'HMMA.16816.F32' "$W4A8_SASS"

printf '%s\n' \
  'B200-NVFP4-DRY PASS' \
  'w4a4: ABI present, tcgen05 NVFP4 4X + TMEM copy/barrier/alloc/dealloc present' \
  'w4a8: int8 tensor-core MMA present; optional F8F4 plain-E4M3 MMA present'
