#!/usr/bin/env bash
# DAY62's target-card driver (DAY62.md section 2): waits for the build receipt, then the price cell (prime,
# retire-seam, retire-seam-other; the prefix cache at 448 MB, the long-entry shape of the chain cell) under ONE
# collector hold, then day62-reading.py. A busy collector
# lock: bounded retries (60 x 120 s). Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-d62
D=research/spill-a-20260919/pro-single-day62
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
for try in $(seq 1 60); do
  python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out "$R/ab-seam-cell" --external-lock --execute bash $D/ab-mode.sh @COLLECTOR_LOCK_FD@ seam prime retire-seam retire-seam-other 448 > "$R/ab-seam-cell.collector.log" 2>&1
  rc=$?
  if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/ab-seam-cell.collector.log" && [ ! -f "$R/ab-seam-cell/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) lock busy, retry $try/60 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "${R:?}/ab-seam-cell"; sleep 120; continue
  fi
  echo "$(date -u +%FT%TZ) ab-seam-cell rc=$rc" | tee -a "$R/progress.log"; break
done
python3 research/spill-a-20260919/day62-reading.py "$R" > "$R/reading-day62.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-day62.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
