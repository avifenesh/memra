#!/usr/bin/env bash
# GB10, one check lease (<= 30 min): compute-sanitizer on bs1f, the block-scaled GEMM kernels only (the pipelined
# tile, the tile list, the small-row and fused SwiGLU bodies, the reference tile):
#   1. initcheck on the kernel gate (uninitialized global / shared reads), with the partial-N cases
#   2. racecheck on the kernel gate (shared-memory hazards)
#   3. initcheck on the failing configuration at reduced size: tp, layer 1 only, 4 ranks on the card, peer-pull
# Each step has its own time cap; the kernel gate's own verdicts are kept beside the sanitizer's.
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
CS=/usr/local/cuda/bin/compute-sanitizer
B=$R/bins/bs1f/dsv41-gate
O=$R/out/san-bs1f; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1
F="--kernel-name kns=v41_gemm_f8 --kernel-name kns=v41_gemv_bs --kernel-name kns=v41_swiglu_bs"
step() { tag=$1; cap=$2; shift 2
  echo "$tag start $(date +%T)" >> $O/runs.txt
  timeout $cap "$@" > $O/$tag.log 2>&1
  echo "$tag rc $? $(date +%T) | $(grep -aE 'ERROR SUMMARY|RACECHECK SUMMARY' $O/$tag.log | tail -1) | $(grep -ao '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1)" >> $O/runs.txt; }
step kernels-plain 4m $B kernels $D/fixtures-gpu/rand --out $O/kernels-plain.json
step kernels-initcheck 9m $CS --tool initcheck --track-unused-memory no $F --print-limit 50 $B kernels $D/fixtures-gpu/rand --out $O/kernels-initcheck.json
step kernels-racecheck 7m $CS --tool racecheck --racecheck-report hazard $F --print-limit 50 $B kernels $D/fixtures-gpu/rand --out $O/kernels-racecheck.json
MEMRA_TP_TRANSPORT=peer-pull step tp4-initcheck 7m $CS --tool initcheck --track-unused-memory no --kernel-name kns=v41_gemm_f8_pipe --print-limit 50 \
  $B tp $C $D/captures-v2/needle --layers 1 --devices 0,0,0,0 --token-map $TM --out $O/tp4-initcheck.json
echo "SAN-BS1F-DONE $(date +%T)" >> $O/runs.txt
