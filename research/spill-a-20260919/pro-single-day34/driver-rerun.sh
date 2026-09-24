#!/usr/bin/env bash
# BOX4 day-34 rerun (DAY34.md section 5): attempt 1 of driver.sh refused every boot at the port guard (`cannot prove port
# 18132 is free ... Install iproute2 (ss) or lsof`: neither is on this box); iproute2 was installed. This waits for
# attempt 1's unit-cell collector to finish (its cells do not boot a server), moves attempt 1's boot-dependent
# receipts to <R>/attempt1-no-ss/, and runs the same doubleparks.sh, gates.sh and hitgate.sh on the same binaries, under
# the same collector and bounded retry. No script, clause, threshold or cell changes. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-day34
D=research/spill-a-20260919/pro-single-day34
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until [ -f "$R/unit-cell/CELL.jsonl" ] && grep -q '"exit_code"' "$R/unit-cell/CELL.jsonl" 2>/dev/null && ! pgrep -f "tier-battery.py .*unit-cell" > /dev/null; do sleep 15; done
mkdir -p "$R/attempt1-no-ss"
for p in doubleparks doubleparks.collector.log d32 d33 double-park double-park.log gates gates.collector.log; do
  [ -e "$R/$p" ] && mv "$R/$p" "$R/attempt1-no-ss/"
done
cp "$R/progress.log" "$R/attempt1-no-ss/progress.log"
echo "$(date -u +%FT%TZ) rerun start (iproute2 installed: $(command -v ss))" | tee -a "$R/progress.log"
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
echo "$(date -u +%FT%TZ) rerun-done" | tee -a "$R/progress.log"
