#!/bin/bash
# One card, back to back short leases on the binaries kept under spec/bin/srv13: the served chat gate on the 9-layer
# trunk, the draft tapes (the untraced draft's device chain against the traced one), then the served A/B
# (MEMRA_DSV41_BATCH_ROWS 1 vs 8, A B B A A B) through the shared client.
# usage: chain13.sh CARD
set -u
CARD=$1
S=$LANE/spec
export TBIN=$S/bin/srv13
bash $S/scripts/lease.sh gate9 600 $CARD bash $S/scripts/serve_gate9.sh gate9-srv13 > $S/logs/gate9-srv13.log 2>&1
echo "$(date +%H:%M:%S) gate rc $? $(grep -c PASS $S/logs/gate9-srv13.log) pass lines"
BIN=$TBIN/dsv41-spec bash $S/scripts/lease.sh dtape 300 $CARD bash $S/scripts/spec_run.sh dtape dtape-srv13 --layers 9 --resident > $S/logs/dtape-srv13.log 2>&1
echo "$(date +%H:%M:%S) dtape rc $?"
i=0
for arm in A B B A A B; do
  i=$((i + 1))
  n=8; [ $arm = A ] && n=1
  EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh ab13-r$i 900 $CARD bash $S/scripts/serve_bench.sh ab13/r$i-$arm \
    decode_c1,conc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 MEMRA_DSV41_BATCH_ROWS=$n > $S/logs/ab13-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i arm $arm rc $?"
done
echo CHAIN13-DONE
