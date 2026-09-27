#!/usr/bin/env bash
# BOX3 day-18 driver: bounded retry of a busy collector lock (15 x 120 s per cell), never signals the holder.
# Cells: the door gates (gates.sh under the collector), the hit gate (hitgate.sh, its own flock), the stall
# cell `on` boot (the promote arm is the day's claim; the demote arm runs too because the day-16 script runs both).
cd /root/wt-a
R=/root/spill-receipts/a-day18
D=research/spill-a-20260919/pro-single-day18
cell() { # name script args...
  local name=$1; shift
  for try in $(seq 1 15); do
    python3 tools/tier-battery.py --rig pro-single --timeout 5400 --out $R/$name --external-lock --execute bash "$@" > $R/$name.collector.log 2>&1
    rc=$?
    if grep -q "^REFUSED: \[Errno 11\]" $R/$name.collector.log && [ $rc -eq 2 ]; then
      echo "$(date -u +%FT%TZ) $name lock busy, retry $try/15 in 120 s"; rm -rf $R/$name; sleep 120; continue
    fi
    echo "$(date -u +%FT%TZ) $name rc=$rc"; break
  done
}
cell gates $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh
cell unit-cell $D/unit-cells.sh @COLLECTOR_LOCK_FD@
cell stall-on $D/stall-cell.sh @COLLECTOR_LOCK_FD@ on
echo driver-done
