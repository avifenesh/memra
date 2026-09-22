#!/usr/bin/env bash
# BOX3 day-23 driver (Move 2 owed item 2, the draft-bearing restore through the door): bounded retry of a busy
# collector lock (15 x 120 s per cell), never signals the holder. Cells: the door gates (gates.sh under the
# collector: identity default and plain, OFF and ON; failure OFF and ON; the fault gate, all cells, the d2d-restore
# cell must still refuse; the twin gate OFF and ON), the hit gate (hitgate.sh, its own flock, OFF and ON; the ON arm
# armed per C day 27, where the spec-on boot's draft-bearing hits must now read `restore submitted off the tick`),
# the GPU unit cells (unit-cells.sh: option_b, option_c, the engine's d2d_ cells).
cd /root/wt-a
R=/root/spill-receipts/a-day23
D=research/spill-a-20260919/pro-single-day23
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
