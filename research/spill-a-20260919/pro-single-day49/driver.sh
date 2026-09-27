#!/usr/bin/env bash
# DAY49's target-card driver: waits for the build receipt, then cell.sh under ONE collector hold (bounded retries on a
# busy lock, 60 x 120 s, never a signal to the holder). Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-d49
D=research/spill-a-20260919/pro-single-day49
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
for try in $(seq 1 60); do
  python3 tools/tier-battery.py --rig pro-single --timeout 7200 --out "$R/cell-collector" --external-lock --execute bash $D/cell.sh @COLLECTOR_LOCK_FD@ > "$R/cell.collector.log" 2>&1
  rc=$?
  if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/cell.collector.log" && [ ! -f "$R/cell-collector/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) cell lock busy, retry $try/60 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "${R:?}/cell-collector"; sleep 120; continue
  fi
  echo "$(date -u +%FT%TZ) cell rc=$rc" | tee -a "$R/progress.log"; break
done
