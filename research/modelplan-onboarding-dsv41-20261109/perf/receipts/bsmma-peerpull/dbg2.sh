#!/usr/bin/env bash
# GB10, one lease (<= 30 min): the exact failing configuration of the bs1 gates (tp, 6 layers, 4 ranks on the
# card, peer-pull) twice with bs1 and once with k3's binary: is the layer-1 MoE mismatch intermittent, and can k3
# hit it too?
set -u
R=$HOME/.cache/dsv41-bsmma
S=$HOME/.cache/dsv41-speed
D=$S/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/dbg2
mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
run() { tag=$1; bin=$2
  timeout 9m $bin/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices 0,0,0,0 --decode 2 \
    --token-map $TM --out $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1) | $(grep -a '^FAIL' $O/$tag.log | head -1 | cut -c1-110) $(date +%T)" >> $O/runs.txt; }
run bs1-a $R/bins/bs1
run k3-a $S/bins/k3
run bs1-b $R/bins/bs1
echo "DBG2-DONE $(date +%T)" >> $O/runs.txt
