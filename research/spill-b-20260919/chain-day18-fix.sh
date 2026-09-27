#!/usr/bin/env bash
# Day 18 local chain, part 1: the fix binary on the day-17 shapes. Cell B shape (the day-16 lru loop:
# cohort 1250/1350/1450/1550, 12 turns of 11,000 + 150) on the twin gate, then the five restore points.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day18
cd "$WT" || exit 1
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-fix-gate.csv
bash research/spill-b-20260919/run-day18-gate.sh fix gate-fix-day16-shape 1800 \
  --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150; echo "gate-fix rc=$?" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-fix-restore.csv
bash research/spill-b-20260919/run-day18-restore.sh fix restore-fix 1200; echo "restore-fix rc=$?" >> $R/chain.log
echo "part1 done $(date -u +%FT%TZ)" >> $R/chain.log
