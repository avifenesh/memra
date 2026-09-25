#!/usr/bin/env bash
# DAY54's target-card driver (DAY54.md section 1): waits for the build receipt, then, each under ONE collector hold:
# ab-mode.sh short (fanout against prime-short, the prefix cache at 256 MB) and long (fanout-long against prime, 448
# MB), pause.sh, census.sh; hitgate.sh under its own flock; then day54-reading.py. A busy collector lock: bounded
# retries (60 x 120 s), never a signal to the holder. Executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-d54
D=research/spill-a-20260919/pro-single-day54
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
cell ab-short-cell 10800 $D/ab-mode.sh @COLLECTOR_LOCK_FD@ short fanout prime-short 256
cell ab-long-cell 10800 $D/ab-mode.sh @COLLECTOR_LOCK_FD@ long fanout-long prime 448
cell pause-cell 3600 $D/pause.sh @COLLECTOR_LOCK_FD@
cell census-cell 7200 $D/census.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
python3 research/spill-a-20260919/day54-reading.py "$R" > "$R/reading-day54.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-day54.log")" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
