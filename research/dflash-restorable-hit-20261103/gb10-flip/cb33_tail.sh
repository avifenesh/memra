#!/usr/bin/env bash
# Runs ON spark-cb33 after arm 4 (dflash-1031r-off, full) finishes: the c1 repeats that make the c1
# cell a same-unit ABBA with arm 4's c1 as the first A: on, on, off (on = flip build, doors on by
# device default; off = #1031 head, all doors off; both STABLE_BOUNDARY=0). b23d went back to the
# landing agent after arm 3, so the repeats run here.
#   LEASE=<lease> bash cb33_tail.sh
set -u
LEASE=${LEASE:?lease}
D=$HOME/memra-run
L=$D/arms-local/chain.log
for i in $(seq 1 1440); do grep -q 'ARMS_LOCAL2_DONE' "$L" && break; grep -q -E 'FAILED|refusing' "$L" && exit 3; sleep 5; done
grep -q 'ARMS_LOCAL2_DONE' "$L" || exit 2
LEASE=$LEASE UNIT=spark-cb33 SERVE=$D/serve-batchspec-novision.sh bash $D/arms_local2.sh $D/arms-local \
  dflash-flip-default-c1r1:bin-flip-6561499516:1:DEF \
  dflash-flip-default-c1r2:bin-flip-6561499516:1:DEF \
  dflash-1031r-off-c1r:bin-1031r-ea5ef13e8b:1:OFF
echo CB33_TAIL_DONE >> "$L"
