#!/usr/bin/env bash
# BOX4 day-34 driver (DAY34.md section 4: day 33 design F and day 34 design K in one sitting): waits for the
# build receipt (three binaries), then, in one sitting: doubleparks.sh under ONE collector hold (the day-26 double-park cell on
# the day-32, day-33 and day-34 binaries in the same hold), gates.sh under the collector on the day-34 binary (identity
# default and plain OFF/ON, failure OFF/ON, the fault gate with every cell including promote-span-refusal, twin OFF/ON), hitgate.sh
# (its own flock, OFF then ON), unit-cells.sh under the collector. Bounded retry of a busy collector lock (60 x 120 s), never signals
# the holder. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-day34
D=research/spill-a-20260919/pro-single-day34
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
if ! grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$'; then echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; fi
cp "$R/bins/memra-server-d34" "$R/bins/memra-server"
sha256sum "$R/bins/memra-server" > "$R/bins/memra-server.sha256"
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
cell doubleparks 12600 $D/doubleparks.sh @COLLECTOR_LOCK_FD@
cell gates 5400 $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
cell unit-cell 5400 $D/unit-cells.sh @COLLECTOR_LOCK_FD@
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
