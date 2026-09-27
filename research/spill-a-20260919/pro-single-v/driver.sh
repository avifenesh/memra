#!/usr/bin/env bash
# The design-V target-card driver (DAY47 section 1): waits for the build receipt, then, in one sitting: ab-pause.sh
# ((c), (d)) under ONE collector hold; gates.sh under the collector on the V binary ((b): the day-36 set, the fault gate
# default and plain, twin OFF and ON, the pause gate in the plain and default boots); hitgate.sh (its own flock, OFF then
# ON); unit-cells.sh under the collector ((a)). A busy collector
# lock: bounded retries (60 x 120 s), never a signal to the holder. Every cell executed-not-qualified. No host, id or
# price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-v
D=research/spill-a-20260919/pro-single-v
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
cell ab-pause 10800 $D/ab-pause.sh @COLLECTOR_LOCK_FD@
cell gates 5400 $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
cell unit-cell 5400 $D/unit-cells.sh @COLLECTOR_LOCK_FD@
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
