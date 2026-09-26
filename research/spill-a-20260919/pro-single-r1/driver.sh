#!/usr/bin/env bash
# Design R1's target-card driver (DAY62.md section 8): waits for the build receipt, then, each under ONE collector
# hold: gates.sh ((a), 11 gates on the r1 binary), ab.sh seam retire-seam-nosource retire-seam prime 448 ((b) to (d):
# base against r1, 60 boots); then r1-reading.py. A busy collector lock: bounded retries (60 x 120 s),
# never a signal to the holder. Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-r1
D=research/spill-a-20260919/pro-single-r1
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
cell ab-r1-cell 21600 $D/ab.sh @COLLECTOR_LOCK_FD@ seam retire-seam-nosource retire-seam prime 448
python3 research/spill-a-20260919/r1-reading.py "$R" > "$R/reading-r1.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-r1.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
