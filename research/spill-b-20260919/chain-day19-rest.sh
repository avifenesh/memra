#!/usr/bin/env bash
# Day 19 local chain, part 2: wait for part 1, then the #379 gate on base and fix with the corrected
# gate (the day-19 site list), then tools/local-ci.sh --perf (the pre-push hook's perf-ci freshness
# gate names it for engine-file changes; spec.rs moved today), then the push attempt through the hook.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day19
cd "$WT" || exit 1
for _ in $(seq 1 480); do grep -q "chain done" $R/chain.log 2>/dev/null && break; sleep 15; done
bash research/spill-b-20260919/run-day19-hitgate-base-2.sh
bash research/spill-b-20260919/run-day19-hitgate-2.sh
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-localci-perf.csv
mkdir -p $R/local-ci-perf
MEMRA_CI_HITGATE_EV=$R/local-ci-perf/hitgate systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G \
  bash tools/local-ci.sh --perf > $R/local-ci-perf/local-ci.log 2>&1; rc=$?; echo $rc > $R/local-ci-perf.exit; echo "local-ci --perf rc=$rc" >> $R/chain.log
echo "part2 done $(date -u +%FT%TZ)" >> $R/chain.log
