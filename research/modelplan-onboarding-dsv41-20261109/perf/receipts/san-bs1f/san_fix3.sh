#!/usr/bin/env bash
# Same lease as san_bs1f.sh: initcheck rerun UNFILTERED (a kernel filter hides the writes of the kernels it leaves out,
# so every read of act_quant's codes showed as uninitialized) and without --track-unused-memory (not an option of
# this sanitizer). initcheck covers global memory only; shared-memory hazards are racecheck's (0 hazards).
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
CS=/usr/local/cuda/bin/compute-sanitizer
B=$R/bins/bs1f/dsv41-gate
O=$R/out/san-bs1f
export MEMRA_FULL_PREC=1
: > $O/fix.txt
step() { tag=$1; cap=$2; shift 2
  echo "$tag start $(date +%T)" >> $O/fix.txt
  timeout $cap "$@" > $O/$tag.log 2>&1
  echo "$tag rc $? $(date +%T) | $(grep -aE 'ERROR SUMMARY' $O/$tag.log | tail -1) | $(grep -ao '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1)" >> $O/fix.txt; }
step kernels-initcheck3 8m $CS --tool initcheck --print-limit 50 $B kernels $D/fixtures-gpu/rand --out $O/kernels-initcheck3.json
MEMRA_TP_TRANSPORT=peer-pull step tp4-initcheck3 6m $CS --tool initcheck --print-limit 50 \
  $B tp $C $D/captures-v2/needle --layers 1 --devices 0,0,0,0 --token-map $TM --out $O/tp4-initcheck3.json
echo "SANFIX-DONE $(date +%T)" >> $O/fix.txt
