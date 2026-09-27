#!/usr/bin/env bash
R=/root/spill-receipts/a-day38
cd /root/wt-a || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
bash research/spill-a-20260919/pro-single-day38/diag6-build.sh
echo "$(date -u +%FT%TZ) diag6-build $(tail -1 $R/diag6-build.log)" >> $R/progress.log
tail -1 $R/diag6-build.log | grep -q "^rc=0$" || exit 1
python3 tools/tier-battery.py --rig pro-single --timeout 7200 --out $R/diag6-cell --external-lock --execute bash research/spill-a-20260919/pro-single-day38/diag6.sh @COLLECTOR_LOCK_FD@ > $R/diag6-cell.collector.log 2>&1
echo "$(date -u +%FT%TZ) diag6-cell rc=$?" >> $R/progress.log
