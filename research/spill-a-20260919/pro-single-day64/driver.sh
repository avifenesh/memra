#!/usr/bin/env bash
# DAY64's target-card driver (DAY64.md section 2): waits for the build receipt, then the placing cell (the promote
# mode, 20 boots, the prefix cache at 256 MB) under ONE collector hold, then day64-reading.py. A busy collector
# lock: bounded retries (60 x 120 s). Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-d64
D=research/spill-a-20260919/pro-single-day64
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
for try in $(seq 1 60); do
  python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out "$R/ab-promote-cell" --external-lock --execute bash $D/ab-mode.sh @COLLECTOR_LOCK_FD@ promote promote promote 256 > "$R/ab-promote-cell.collector.log" 2>&1
  rc=$?
  if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/ab-promote-cell.collector.log" && [ ! -f "$R/ab-promote-cell/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) lock busy, retry $try/60 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "${R:?}/ab-promote-cell"; sleep 120; continue
  fi
  echo "$(date -u +%FT%TZ) ab-promote-cell rc=$rc" | tee -a "$R/progress.log"; break
done
python3 research/spill-a-20260919/day64-reading.py "$R" > "$R/reading-day64.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-day64.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
