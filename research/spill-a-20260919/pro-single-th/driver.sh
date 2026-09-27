#!/usr/bin/env bash
# Design T-H's target-card driver (DAY65.md section 1): waits for the build receipt, then, each under ONE collector
# hold: gates.sh ((a), 11 gates on the th binary), ab.sh demote ((b), (d)), ab.sh chain ((c), (d)), ab.sh promote
# ((d)), 20 boots each, hump.sh ((d), 4 boots); then th-reading.py. A busy collector lock: bounded retries (60 x 120 s),
# never a signal to the holder. Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-th
D=research/spill-a-20260919/pro-single-th
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
cell ab-demote-cell 10800 $D/ab.sh @COLLECTOR_LOCK_FD@ demote
cell ab-chain-cell 10800 $D/ab.sh @COLLECTOR_LOCK_FD@ chain
cell ab-promote-cell 10800 $D/ab.sh @COLLECTOR_LOCK_FD@ promote
cell hump-cell 3600 $D/hump.sh @COLLECTOR_LOCK_FD@
python3 research/spill-a-20260919/th-reading.py "$R" > "$R/reading-th.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-th.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
