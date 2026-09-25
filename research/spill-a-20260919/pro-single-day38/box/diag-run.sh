#!/usr/bin/env bash
set -uo pipefail
cd /root/wt-a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
R=/root/spill-receipts/a-day38
bash research/spill-a-20260919/pro-single-day38/diag-build.sh
grep -q "^rc=0" $R/diag-build.log || { echo "diag build failed"; exit 1; }
python3 tools/tier-battery.py --rig pro-single --timeout 7200 --out $R/diag-cell --external-lock --execute bash research/spill-a-20260919/pro-single-day38/diag.sh @COLLECTOR_LOCK_FD@ > $R/diag-cell.collector.log 2>&1
echo "$(date -u +%FT%TZ) diag-cell rc=$?" | tee -a $R/progress.log
