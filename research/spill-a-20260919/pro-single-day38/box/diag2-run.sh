#!/usr/bin/env bash
set -uo pipefail
cd /root/wt-a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
R=/root/spill-receipts/a-day38
python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out $R/diag2-cell --external-lock --execute bash research/spill-a-20260919/pro-single-day38/diag2.sh @COLLECTOR_LOCK_FD@ > $R/diag2-cell.collector.log 2>&1
echo "$(date -u +%FT%TZ) diag2-cell rc=$?" | tee -a $R/progress.log
