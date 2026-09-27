#!/usr/bin/env bash
# Day 14 final gate pair on the corrected gate (calibration boot + footprint-corrected V3): base then fix,
# same gate source, same binaries as the first round, same card, back to back; then the collector validates.
set -uo pipefail
R=/root/spill-receipts/b-day14
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
git rev-parse HEAD > $R/gate-source-final.txt
bash $R/run-gate.sh main -final; echo "gate-main-final rc=$?" >> $R/finals.log
bash $R/run-gate.sh fix -final; echo "gate-fix-final rc=$?" >> $R/finals.log
for c in gate-main-final gate-fix-final; do
  [ -d $R/$c ] && python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/finals.log
done
echo done >> $R/finals.log
