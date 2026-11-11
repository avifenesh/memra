#!/usr/bin/env bash
# GB10 sanitizer lease (cutlass branch): compute-sanitizer racecheck and initcheck (unfiltered) on the kernel gate,
# initcheck on a 4-rank peer-pull layer-1 tp gate and on a one-rank decode-graph tp gate.
# Done-file out/gates-$TAG/san.txt, marker GATES-SAN2-DONE.
set -u
TAG=$(cat $HOME/.cache/dsv41-cutlass/build-tag)
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
CS=/usr/local/cuda/bin/compute-sanitizer
B=$R/bins/$TAG/dsv41-gate
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/san2.txt
export MEMRA_FULL_PREC=1
step() { tag=$1; cap=$2; shift 2
  echo "$tag start $(date +%T)" >> $O/san2.txt
  timeout $cap "$@" > $O/$tag.log 2>&1
  echo "$tag rc $? $(date +%T) | $(grep -aE 'ERROR SUMMARY|RACECHECK SUMMARY' $O/$tag.log | tail -1) | $(grep -ao '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1)" >> $O/san2.txt; }
step kernels-racecheck 17m $CS --tool racecheck --print-limit 50 $B kernels $D/fixtures-gpu/rand --out $O/kernels-racecheck.json
MEMRA_TP_TRANSPORT=peer-pull step tp4-initcheck2 12m $CS --tool initcheck --print-limit 50 \
  $B tp $C $D/captures-v2/needle --layers 1 --devices 0,0,0,0 --token-map $TM --out $O/tp4-initcheck2.json
echo "GATES-SAN2-DONE $(date +%T)" >> $O/san2.txt
