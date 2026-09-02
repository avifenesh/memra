#!/usr/bin/env bash
# GPU-less B200 block-FP8 object/SASS gate. This script never probes or opens a CUDA device.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NVCC=${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}
CUDA_BIN=$(dirname "$NVCC")
CUOBJDUMP="$CUDA_BIN/cuobjdump"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

cd "$ROOT"

"$NVCC" -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 \
  --expt-relaxed-constexpr -DMEMRA_FP8BLK_PLAIN_MMA=1 -DMEMRA_SM100_TCGEN05=1 \
  -c crates/memra-engine/cu/mmq_fp8_blk.cu -o "$SCRATCH/mmq_fp8_blk_sm100.o"

for symbol in memra_mmq_fp8_blk memra_mmq_fp8_blk_act_bytes \
              memra_mmq_fp8_blk_quantize_act memra_mmq_fp8_blk_grouped \
              memra_mmq_fp8_blk_scale_rows memra_mmq_fp8_blk_scale_cols; do
  nm -g --defined-only "$SCRATCH/mmq_fp8_blk_sm100.o" | grep -Eq "[[:space:]]${symbol}$"
done

SASS=$SCRATCH/fp8.sass
"$CUOBJDUMP" --dump-sass "$SCRATCH/mmq_fp8_blk_sm100.o" > "$SASS"
grep -q 'Function : _Z23mul_mat_q_fp8_blk_sm100' "$SASS"
grep -q 'UTCQMMA' "$SASS"
grep -q 'UTCCP' "$SASS"
grep -q 'UTCBAR' "$SASS"
grep -q 'UTCATOMSWS.FIND_AND_SET' "$SASS"
grep -q 'UTCATOMSWS.AND' "$SASS"
# The expert-grouped fallback stays present on SM100 through the legal plain E4M3 MMA form.
grep -q 'HMMA.16816.F32' "$SASS"

printf '%s\n' \
  'B200-FP8-DRY PASS' \
  'dense: ABI present, tcgen05 FP8 + TMEM copy/barrier/alloc/dealloc present' \
  'grouped: ABI present, plain E4M3 tensor-core MMA fallback present' \
  'runtime: explicit MEMRA_FP8_MMQ=1 retained; B200 hardware state is NativeReference, not tuned'
