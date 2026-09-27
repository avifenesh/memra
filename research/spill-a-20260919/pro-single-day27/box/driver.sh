#!/usr/bin/env bash
# BOX3 day-27 driver: the day-26 double-park cell once more on the day-27 tree (the day-26 code) as the baseline the chosen option is measured against; one collector hold, twenty boots. Bounded retry of a busy collector lock. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-day27
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
for try in $(seq 1 30); do
  python3 tools/tier-battery.py --rig pro-single --timeout 7200 --out "$R/double-park" --external-lock --execute bash research/spill-a-20260919/pro-single-day26/double-park.sh @COLLECTOR_LOCK_FD@ /root/wt-a "$R" "$MODEL" "$BIN" > "$R/double-park.collector.log" 2>&1
  rc=$?
  if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/double-park.collector.log" && [ ! -f "$R/double-park/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) double-park lock busy, retry $try/30 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "$R/double-park"; sleep 120; continue
  fi
  echo "$(date -u +%FT%TZ) double-park rc=$rc" | tee -a "$R/progress.log"; break
done
echo "$(date -u +%FT%TZ) double-park $(cat "$R/double-park/ev/exit.txt" 2>/dev/null)" >> "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
