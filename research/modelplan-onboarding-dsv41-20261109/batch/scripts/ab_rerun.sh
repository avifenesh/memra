#!/bin/bash
# Two more runs of a served A/B directory (one alone, one batched), numbered after the existing six.
# usage: ab_rerun.sh TAG CARD
set -u
TAG=$1; CARD=$2
S=$LANE/spec
export TBIN=$S/bin/$TAG
i=6
for arm in A B; do
  i=$((i + 1))
  n=8; [ $arm = A ] && n=1
  EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh ab-$TAG-r$i 900 $CARD bash $S/scripts/serve_bench.sh \
    ab-$TAG/r$i-$arm decode_c1,conc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 MEMRA_DSV41_BATCH_ROWS=$n \
    > $S/logs/ab-$TAG-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i arm $arm rc $?"
done
python3 $S/scripts/ab_summary.py $S/out/ab-$TAG
echo "AB-RERUN-DONE $TAG"
