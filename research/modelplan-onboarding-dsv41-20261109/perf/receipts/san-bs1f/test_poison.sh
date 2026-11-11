#!/usr/bin/env bash
# GB10 lease (<= 30 min): bs1p (dynamic shared memory poisoned with 0xFF in every GEMM kernel) through the kernel gate
# (byte-exact against the unpoisoned reference tile) and the failing configuration (tp, 6 layers, 4 ranks, peer-pull)
# twice, with the hash log: an output that read unwritten shared memory would move.
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$R/bins/bs1p/dsv41-gate
O=$R/out/poison; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1
timeout 5m $B kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
echo "kernels rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels.log | tail -1) $(date +%T)" >> $O/runs.txt
grep -a "^FAIL" $O/kernels.log | head -10 >> $O/runs.txt
for i in 1 2; do
  MEMRA_TP_TRANSPORT=peer-pull timeout 7m $B tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices 0,0,0,0 \
    --decode 2 --token-map $TM --out $O/tp4-$i.json > $O/tp4-$i.log 2>&1
  echo "tp4 run $i rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/tp4-$i.log | tail -1) $(date +%T)" >> $O/runs.txt
  grep -a "^HASH layer1/prefill/.*moe_out" $O/tp4-$i.log | tr '\n' ' ' >> $O/runs.txt; echo >> $O/runs.txt
done
echo "POISON-DONE $(date +%T)" >> $O/runs.txt
