#!/usr/bin/env bash
# Day 18 local chain, parts 2b and 3. The first restore cells (restore-fix, restore-base) ran the gate's
# mistaken default of the chain's turn 12 (12,650 ids); they are kept as valid extra receipts on a prompt
# that is not the day-17 near-tie. The day-17 shape is turn 10 (12,350 ids): wait for the running base
# restore cell, then the restore gate at --turns 10 on base (red expected on the off-grid points) and on
# the fix, then the #379 gate on the fix. One GPU cell at a time.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day18
cd "$WT" || exit 1
for _ in $(seq 1 240); do [ -f $R/restore-base.exit ] && break; sleep 15; done
echo "restore-base(t12) rc=$(cat $R/restore-base.exit)" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-base-restore-t10.csv
bash research/spill-b-20260919/run-day18-restore.sh base restore-base-t10 1200 --turns 10; echo "restore-base-t10 rc=$?" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-fix-restore-t10.csv
bash research/spill-b-20260919/run-day18-restore.sh fix restore-fix-t10 1200 --turns 10; echo "restore-fix-t10 rc=$?" >> $R/chain.log
bash research/spill-b-20260919/run-day18-hitgate.sh
echo "part3 done $(date -u +%FT%TZ)" >> $R/chain.log
