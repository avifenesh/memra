#!/usr/bin/env bash
# Day 19 local chain, one GPU cell at a time: the #379 gate on base then fix (own flock, as local-ci
# runs it), the twin gate on the day-16 shape (fix), the restore gate at turn 10 (fix), then the full
# local-ci correctness stage once (it takes the rig lock itself and runs the hit gate again, both
# postures). Every cell records its card state before it starts.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day19
cd "$WT" || exit 1
bash research/spill-b-20260919/run-day19-hitgate-base.sh
bash research/spill-b-20260919/run-day19-hitgate.sh
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-fix-gate.csv
bash research/spill-b-20260919/run-day19-gate.sh fix gate-fix-day16-shape 1800 \
  --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150; echo "gate-fix rc=$?" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-fix-restore.csv
bash research/spill-b-20260919/run-day19-restore.sh fix restore-fix-t10 1200 --turns 10; echo "restore-fix-t10 rc=$?" >> $R/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-localci.csv
mkdir -p $R/local-ci
MEMRA_CI_HITGATE_EV=$R/local-ci/hitgate systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G \
  bash tools/local-ci.sh > $R/local-ci/local-ci.log 2>&1; rc=$?; echo $rc > $R/local-ci.exit; echo "local-ci rc=$rc" >> $R/chain.log
echo "chain done $(date -u +%FT%TZ)" >> $R/chain.log
