#!/usr/bin/env bash
# Design R2's target-card driver (DAY62.md sections 4 and 5): waits for the build receipt, then, each under ONE
# collector hold: gates.sh ((a), 11 gates on the r2 binary), ab.sh seam prime retire-seam 448 ((b), (c): base
# against r2, 40 boots), ab-r1.sh seam-r1 prime retire-seam retire-seam-nosource 448 (R1's corrected cell on base, 30
# boots); then r2-reading.py and day62-reading.py for R1's cell. A busy collector lock: bounded retries (60 x 120 s),
# never a signal to the holder. Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-r2
D=research/spill-a-20260919/pro-single-r2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
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
cell gates-cell 10800 $D/gates.sh @COLLECTOR_LOCK_FD@
cell ab-r2-cell 14400 $D/ab.sh @COLLECTOR_LOCK_FD@ seam prime retire-seam 448
cell ab-r1-cell 14400 $D/ab-r1.sh @COLLECTOR_LOCK_FD@ seam-r1 prime retire-seam retire-seam-nosource 448
python3 research/spill-a-20260919/r2-reading.py "$R" > "$R/reading-r2.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-r2.log")" | tee -a "$R/progress.log"
python3 research/spill-a-20260919/day62-reading.py "$R" seam-r1 retire-seam-nosource > "$R/reading-r1.log" 2>&1
echo "$(date -u +%FT%TZ) r1 reading rc=$? $(tail -1 "$R/reading-r1.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
