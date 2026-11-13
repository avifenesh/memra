#!/bin/bash
set -u
DL=$(( $(date +%s) + 840 - 30 ))
for b in seam-kpc seam-kpcR; do
  for t in default peer-pull; do
    [ $(( $(date +%s) + 150 )) -le $DL ] || { echo "SKIP $b $t (cap)" >> /home/access/dsv41/receipts-perf/seam1113/runs.txt; continue; }
    if [ $t = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$t; fi
    echo "$b $t start $(date +%T)" >> /home/access/dsv41/receipts-perf/seam1113/runs.txt
    env MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel /home/access/dsv41/perf-bins/$b/dsv41-gate tp /home/access/dsv41/DeepSeek-V4.1-Flash-1017 /home/access/dsv41/captures-v2/needle       --layers 1,2,14,20,21,24 --devices 0,1,2,3 --decode 2 --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32 --out /home/access/dsv41/receipts-perf/seam1113/gate4-$b-$t.json       > /home/access/dsv41/receipts-perf/seam1113/gate4-$b-$t.log 2>&1
    echo "$b $t rc $? end $(date +%T): $(grep -o '[0-9]* checks, [0-9]* failed' /home/access/dsv41/receipts-perf/seam1113/gate4-$b-$t.log | tail -1) $(grep -ac '^FAIL' /home/access/dsv41/receipts-perf/seam1113/gate4-$b-$t.log) FAIL lines" >> /home/access/dsv41/receipts-perf/seam1113/runs.txt
  done
done
