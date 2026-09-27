#!/usr/bin/env bash
# BOX3 day-19 driver (the settle-time reader wait for an H2D, rule 3): bounded retry of a busy collector lock
# (15 x 120 s per cell), never signals the holder. Cells: the door gates (gates.sh under the collector: identity
# default and plain, OFF and ON; failure OFF and ON; the fault gate; the twin gate OFF and ON), the hit gate
# (hitgate.sh, its own flock, OFF and ON), the door's GPU unit cells (unit-cells.sh). No stall cell today: the
# change is an ordering move, decided by the day-18 promote-arm reading, and its decision cell is the same-window
# A/B owed in OWNER-THREAD-OFFLOAD.md.
cd /root/wt-a
R=/root/spill-receipts/a-day19
D=research/spill-a-20260919/pro-single-day19
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
echo driver-done
