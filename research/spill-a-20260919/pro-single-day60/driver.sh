#!/usr/bin/env bash
# DAY60 (OWED item 11) target-card driver: C's day-29 runner verbatim over this build (the stall cell (i) and the hit
# gate, one collector hold each), then C's reader. usage: driver.sh <receipts_root> <model.gguf>
set -uo pipefail
R=$1; MODEL=$2
cd /root/wt-a || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
bash research/spill-c-20260919/day29-box-run.sh /root/wt-a "$R" "$MODEL" /root/wt-a-day16
python3 research/spill-c-20260919/day29-stall-reading.py "$R/stall/ev" > "$R/reading-day60.log" 2>&1
echo "$(date -u +%FT%TZ) reading rc=$? $(tail -1 "$R/reading-day60.log")" >> "$R/progress.log"
echo "$(date -u +%FT%TZ) A60-DONE" >> "$R/progress.log"
