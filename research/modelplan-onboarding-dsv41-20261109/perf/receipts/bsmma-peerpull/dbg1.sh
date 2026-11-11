#!/usr/bin/env bash
# GB10, inside the bs1 gates lease: locate the 4-rank layer-1 prefill MoE mismatch. Layer 1 only:
# 4 ranks twice (determinism), 4 ranks with --chunk 1024, 2 ranks, 4 ranks host-canonical.
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/dbg1
mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
B=$R/bins/bs1
run() { tag=$1; shift
  $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1 --token-map $TM "$@" --out $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1) | $(grep -a '^FAIL.*moe_out' $O/$tag.log | head -1 | cut -c1-120)" >> $O/runs.txt; }
run n4a --devices 0,0,0,0
run n4b --devices 0,0,0,0
run n4c1024 --devices 0,0,0,0 --chunk 1024
run n2 --devices 0,0
echo "DBG1-DONE $(date +%T)" >> $O/runs.txt
