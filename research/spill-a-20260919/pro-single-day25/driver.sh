#!/usr/bin/env bash
# BOX3 day-25 driver (Task 1 the double park, Task 2 the retire-settle share): waits for the build receipt, then
# runs double-park.sh ONCE under tools/tier-battery.py --rig pro-single --external-lock (one hold, twenty boots),
# then hitgate.sh ONCE (its own flock, door ON). Bounded retry of a busy collector lock (60 x 120 s), never
# signals the holder (lane C shares the card today). No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-day25
D=research/spill-a-20260919/pro-single-day25
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
if ! grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$'; then echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; fi
mkdir -p "$R/bins"
cp target/release/memra-server "$BIN"
sha256sum "$BIN" > "$BIN.sha256"
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
cell double-park 7200 $D/double-park.sh @COLLECTOR_LOCK_FD@ /root/wt-a "$R" "$MODEL" "$BIN"
echo "$(date -u +%FT%TZ) double-park $(cat "$R/double-park/ev/exit.txt" 2>/dev/null)" >> "$R/progress.log"
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
