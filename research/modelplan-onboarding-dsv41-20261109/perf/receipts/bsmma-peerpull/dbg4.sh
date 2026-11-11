#!/usr/bin/env bash
# GB10, one lease (<= 30 min): the failing configuration (tp, 6 layers, 4 ranks on the card, peer-pull),
# k3d and bs1d interleaved (ORDER), each run logging the hashes of the single-card reference's and every
# rank's moe_out / out_stream, so a failing run shows whether the reference or the ranks moved.
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/${TAG:-dbg4}
ORDER=${ORDER:-"k3d bs1d bs1d k3d k3d"}
mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
i=0
for arm in $ORDER; do
  i=$((i+1))
  timeout 6m $R/bins/$arm/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices 0,0,0,0 \
    --decode 2 --token-map $TM --out $O/run$i-$arm.json > $O/run$i-$arm.log 2>&1
  echo "run$i-$arm rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/run$i-$arm.log | tail -1) $(date +%T)" >> $O/runs.txt
  grep -a "^HASH layer1/prefill/.*moe_out" $O/run$i-$arm.log | tr '\n' ' ' >> $O/runs.txt; echo >> $O/runs.txt
done
echo "DBG4-DONE $(date +%T)" >> $O/runs.txt
