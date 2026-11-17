#!/usr/bin/env bash
# memra#1258 review round 1 (laptop, one gpulock lease on the RTX 5090): ONE prompt as the first
# placed row of a boot, N fresh boots. Each boot learns its `required` with one unplaced prime,
# starts the keeper and runs one cold prime of seed SEED placed at required + MARGIN (b-sweep.py,
# one margin, one rep), on a ring build with the gap dumps on (MEMRA_DIAG_RING=1).
# usage: b-first.sh ARM LEN MARGIN SEED N [TAG] [FIRST_K]   (LEN 5.7k|8k; output
#        sweep/cap-ARM-first-LEN-mMARGIN[-TAG]/sSEED-rK for K = FIRST_K (default 1) .. FIRST_K+N-1; TAG "-" = none)
set -u
ARM=$1; LEN=$2; MARGIN=$3; SEED=$4; N=$5; TAG=${6:--}; K0=${7:-1}
[ "$TAG" = - ] && TAG="" || TAG="-$TAG"
B=$HOME/.cache/memra-1258
W=()
[ "$LEN" = 8k ] && W=(--words 5112)
rc=0
for k in $(seq "$K0" $((K0 + N - 1))); do
  name="cap-$ARM-first-$LEN-m$MARGIN$TAG/s$SEED-r$k"
  bash "$B/b-sweep.sh" "$name" "$ARM" --margins "$MARGIN" --seed0 $((SEED - 1)) --ring "${W[@]}" > /dev/null 2>&1 || rc=1
  grep -E '^rep ' "$B/sweep/$name/sweep.log" | sed "s/^/boot $k seed $SEED: /"
done
exit $rc
