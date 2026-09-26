#!/usr/bin/env bash
# Design B1's target-card driver (DAY59.md section 7): waits for the build receipt, then, each under ONE collector
# hold, unit-cells.sh ((a1), (a2) green and red), gates.sh ((a3): the identity gate default and plain, door OFF and
# ON), hitgate.sh (its own flock, OFF then ON), ab.sh short fanout prime-short 256 ((b) to (d), 40 boots); then
# b1-reading.py. A busy collector lock: bounded retries (60 x 120 s), never a signal to the holder.
# Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-b1
D=research/spill-a-20260919/pro-single-b1
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
cell unit-cell 3600 $D/unit-cells.sh @COLLECTOR_LOCK_FD@
cell gates-cell 7200 $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
cell ab-short-cell 14400 $D/ab.sh @COLLECTOR_LOCK_FD@ short fanout prime-short 256
python3 research/spill-a-20260919/b1-reading.py "$R" > "$R/reading-b1.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-b1.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
