#!/bin/bash
# One more pair of runs of a two-build served A/B (B then A), numbered 5 and 6.
# usage: abs_rerun.sh TAG_A TAG_B CARD
set -u
A=$1; B=$2; CARD=$3
S=$LANE/spec
i=4
for arm in $B $A; do
  i=$((i + 1))
  TBIN=$S/bin/$arm EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh abs-r$i 900 $CARD bash $S/scripts/serve_bench.sh \
    abs-$A-$B/r$i-$arm decode_c1,conc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 > $S/logs/abs-$A-$B-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i $arm rc $? $(grep -h "page-cache drops" $S/logs/abs-$A-$B-r$i-$arm.log | tail -n 1 | cut -c10-200)"
done
python3 $S/scripts/ab_summary.py $S/out/abs-$A-$B
echo "ABS-RERUN-DONE"
