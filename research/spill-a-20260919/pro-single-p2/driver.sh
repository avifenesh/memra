#!/usr/bin/env bash
# The design-P2 target-card driver (DAY52.md section 1): waits for the build receipt, then, in one sitting, each under
# ONE collector hold: ab.sh demote ((b), (c)), ab.sh free ((f)), ab.sh promote ((d)), ab.sh chain ((g)), hump.sh ((e));
# gates.sh on the P2 binary ((a): the day-36 set, the fault gate default and plain, twin OFF and ON, the pause gate);
# hitgate.sh (its own flock, OFF then ON); unit-cells.sh ((a)); then day52-reading.py over the receipts. A busy
# collector lock: bounded retries (60 x 120 s), never a signal to the holder. Every cell executed-not-qualified. No
# host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-p2
D=research/spill-a-20260919/pro-single-p2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
if ! grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$'; then echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; fi
cell() { # name timeout script args...
  local name=$1 timeout=$2; shift 2
  for try in $(seq 1 60); do
    python3 tools/tier-battery.py --rig pro-single --timeout "$timeout" --out "$R/$name" --external-lock --execute bash "$@" > "$R/$name.collector.log" 2>&1
    rc=$?
    if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/$name.collector.log" && [ ! -f "$R/$name/CELL.jsonl" ]; then
      echo "$(date -u +%FT%TZ) $name lock busy, retry $try/60 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "${R:?}/${name:?}"; sleep 120; continue
    fi
    echo "$(date -u +%FT%TZ) $name rc=$rc" | tee -a "$R/progress.log"; break
  done
}
for c in demote free promote chain; do
  cell "ab-$c-cell" 10800 $D/ab.sh @COLLECTOR_LOCK_FD@ "$c"
done
cell hump-cell 3600 $D/hump.sh @COLLECTOR_LOCK_FD@
cell gates 7200 $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
cell unit-cell 5400 $D/unit-cells.sh @COLLECTOR_LOCK_FD@
python3 research/spill-a-20260919/day52-reading.py "$R" > "$R/reading-day52.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-day52.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
