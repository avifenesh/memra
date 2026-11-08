#!/usr/bin/env bash
# Phase SM (scored): the mixed shape (MEMRA_MAX_SESSIONS=12), main (A) against the candidate (B),
# ABBAAB, then one cache-off run per arm as the greedy reference.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-${CAND:-e0565d0f20}
O=<runs>/phaseSM
mkdir -p "$O"
n=0
for arm in A B B A A B; do
  n=$((n + 1)); if [ $arm = A ]; then bin=$A; else bin=$B; fi
  bash $H/arm2.sh "$(printf %02d $n)-$arm" "$bin" "$O" mixed MEMRA_MAX_SESSIONS=12
done
bash $H/arm2.sh off-A "$A" "$O" mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
bash $H/arm2.sh off-B "$B" "$O" mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
echo PHASE_SM_DONE
