#!/usr/bin/env bash
# bounded retry of a busy collector lock: 15 x 120 s per cell, never signals the holder
cd /root/wt-a
R=/root/spill-receipts/a-day16
for w in prime off on; do
  for try in $(seq 1 15); do
    python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $R/stall-$w --external-lock --execute bash research/spill-a-20260919/pro-single-day16/stall-cell.sh @COLLECTOR_LOCK_FD@ $w > $R/stall-$w.collector.log 2>&1
    rc=$?
    if grep -q "^REFUSED: \[Errno 11\]" $R/stall-$w.collector.log && [ $rc -eq 2 ]; then
      echo "$(date -u +%FT%TZ) cell-$w lock busy, retry $try/15 in 120 s"; rm -rf $R/stall-$w; sleep 120; continue
    fi
    echo "$(date -u +%FT%TZ) cell-$w rc=$rc"; break
  done
done
echo driver-done
