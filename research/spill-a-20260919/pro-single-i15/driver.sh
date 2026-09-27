#!/usr/bin/env bash
# OWED item 15's target-card driver (DAY43.md section 1), run after the S2 sitting's driver in the same sitting (it
# waits for its own build receipt): survey.sh (the kernel's price), ab-long.sh demote-long and promote-long, then
# hump-long.sh (term (4), then the item 15 reading), each under ONE collector hold. A busy collector lock: bounded
# retries (60 x 120 s), never a signal to the holder. Every cell executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-i15
D=research/spill-a-20260919/pro-single-i15
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
cell survey-cell 1800 $D/survey.sh @COLLECTOR_LOCK_FD@
cell demote-long-cell 14400 $D/ab-long.sh @COLLECTOR_LOCK_FD@ demote-long
cell promote-long-cell 14400 $D/ab-long.sh @COLLECTOR_LOCK_FD@ promote-long
cell hump-long-cell 7200 $D/hump-long.sh @COLLECTOR_LOCK_FD@
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
