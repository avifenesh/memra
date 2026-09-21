#!/usr/bin/env bash
# Day 18 local chain, part 2: the base binary (origin/main 1b354be59, built fresh in its own worktree,
# rtx5090-day18/build-base/) on the same two shapes: red expected on V5 (identity 11/12, turn 10) and V6
# (every seed off the grid) for the twin gate, and on the off-grid points for the restore gate.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day18
cd "$WT" || exit 1
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-base-gate.csv
bash research/spill-b-20260919/run-day18-gate.sh base gate-base-day16-shape 1800 \
  --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150; echo "gate-base rc=$?" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-base-restore.csv
bash research/spill-b-20260919/run-day18-restore.sh base restore-base 1200; echo "restore-base rc=$?" >> $R/chain.log
echo "part2 done $(date -u +%FT%TZ)" >> $R/chain.log
