#!/usr/bin/env bash
# Day 18 local chain, parts 2 and 3: wait for part 1 (the fix cells) to finish, then the base cells, then
# the #379 gate on the fix. One GPU cell at a time on this card.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day18
cd "$WT" || exit 1
for _ in $(seq 1 240); do grep -q "part1 done" $R/chain.log 2>/dev/null && break; sleep 15; done
bash research/spill-b-20260919/chain-day18-base.sh
bash research/spill-b-20260919/run-day18-hitgate.sh
echo "part3 done $(date -u +%FT%TZ)" >> $R/chain.log
