#!/bin/bash
# The served batched-step A/B on a 9-layer resident trunk: one short lease per run, arms interleaved A B B A A B
# (A: every call alone, MEMRA_DSV41_BATCH_ROWS=1; B: batched, 8), the lane's shared client, decode_c1 + conc8.
# usage: ab9.sh CARD [SAMPLES]
set -u
CARD=$1; SAMPLES=${2:-3}
S=$LANE/spec
i=0
for arm in A B B A A B; do
  i=$((i + 1))
  n=8; [ $arm = A ] && n=1
  EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh ab9-r$i 900 $CARD bash $S/scripts/serve_bench.sh ab9/r$i-$arm \
    decode_c1,conc8 $SAMPLES MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 MEMRA_DSV41_BATCH_ROWS=$n \
    > $S/logs/ab9-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i arm $arm rc $? $(tail -n 1 $S/logs/ab9-r$i-$arm.log | cut -c1-80)"
done
echo AB9-DONE
