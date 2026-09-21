#!/usr/bin/env bash
# Day 14 round 3 on the state-form V3 gate: base then fix, same gate source, same binaries as rounds 1 and 2,
# same card, back to back; then the collector validates both cells.
set -uo pipefail
R=/root/spill-receipts/b-day14
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
git rev-parse HEAD > $R/gate-source-r3.txt
bash $R/run-gate.sh main -r3; echo "gate-main-r3 rc=$?" >> $R/r3.log
bash $R/run-gate.sh fix -r3; echo "gate-fix-r3 rc=$?" >> $R/r3.log
for c in gate-main-r3 gate-fix-r3; do
  [ -d $R/$c ] && python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/r3.log
done
echo done >> $R/r3.log
