#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): margin rows on the boot's own charge,
# one row per fresh boot. Each boot learns its `required` with one unplaced prime, starts the
# keeper and runs ONE cold prime placed at required + MARGIN (b-sweep.py, one margin, one rep),
# so no row follows a plain replay that raised the gate's fixed residual. One prompt per boot:
# seed S gives the same prompt in every arm.
# usage: b-fresh.sh ARM LEN MARGIN SEED...   (LEN 5.7k|8k; output sweep/cap-ARM-fresh-LEN-mMARGIN/sSEED)
set -u
ARM=$1; LEN=$2; MARGIN=$3; shift 3
B=$HOME/.cache/memra-1258
W=()
[ "$LEN" = 8k ] && W=(--words 5112)
rc=0
for seed in "$@"; do
  bash "$B/b-sweep.sh" "cap-$ARM-fresh-$LEN-m$MARGIN/s$seed" "$ARM" --margins "$MARGIN" --seed0 $((seed - 1)) "${W[@]}" > /dev/null 2>&1 || rc=1
  grep -E '^rep ' "$B/sweep/cap-$ARM-fresh-$LEN-m$MARGIN/s$seed/sweep.log" | sed "s/^/seed $seed: /"
done
exit $rc
