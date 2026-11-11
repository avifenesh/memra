#!/usr/bin/env bash
# GB10 debug lease: the peer-pull rows-16 failures of the tp gate. 2 ranks on peer-pull, layers 2 and 24: c1 x3,
# interleaved k3 / bs1f / c1, four runs each (the failure rate of each program on the same configuration).
# Done-file out/dbg-pp/runs.txt, marker DBG-PP2-DONE.
set -u
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
O=$R/out/dbg-pp2; mkdir -p $O; : > $O/runs.txt
run() { tag=$1; bin=$2; shift 2
  timeout 6m env "$@" $bin/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 --devices 0,0 --decode 2 --token-map $TM \
    --out $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1) $(date +%T)" >> $O/runs.txt
  grep -a '^FAIL' $O/$tag.log | head -6 >> $O/runs.txt; }
run k3-0 $HOME/.cache/dsv41-speed/bins/k3 X=1
run bs1f-0 $HOME/.cache/dsv41-bsmma/bins/bs1f X=1
run c1-0 $R/bins/c1 X=1
run k3-1 $HOME/.cache/dsv41-speed/bins/k3 X=1
run bs1f-1 $HOME/.cache/dsv41-bsmma/bins/bs1f X=1
run c1-1 $R/bins/c1 X=1
run k3-2 $HOME/.cache/dsv41-speed/bins/k3 X=1
run bs1f-2 $HOME/.cache/dsv41-bsmma/bins/bs1f X=1
run c1-2 $R/bins/c1 X=1
run k3-3 $HOME/.cache/dsv41-speed/bins/k3 X=1
run bs1f-3 $HOME/.cache/dsv41-bsmma/bins/bs1f X=1
run c1-3 $R/bins/c1 X=1
echo "DBG-PP2-DONE $(date +%T)" >> $O/runs.txt
