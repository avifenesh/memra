#!/usr/bin/env bash
# Day 30 target-card runner (one RTX PRO 6000 Blackwell, collector rig `pro-single`, lock /tmp/memra-gpu.lock):
# waits for the tree's server build receipt, then runs day30-stall-cell.sh ONCE through
# tools/tier-battery.py --external-lock (one lock hold, eight boots), bounded lock retries (60 x 120 s; lane A
# shares the card today, the holder is never inspected or signalled). No host, id or price here.
# usage: day30-box-run.sh <tree> <receipts_root> <model.gguf>
set -uo pipefail
TREE=$1; R=$2; MODEL=$3
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
BIN=$TREE/target/release/memra-server
mkdir -p "$R/collector"
cell=stall-capture-share
for attempt in $(seq 0 60); do
    out=$R/collector/$cell; [ "$attempt" -gt 0 ] && out=$R/collector/$cell-retry$attempt
    python3 tools/tier-battery.py --rig pro-single --timeout 5400 --out "$out" --external-lock \
        --execute bash research/spill-c-20260919/day30-stall-cell.sh @COLLECTOR_LOCK_FD@ "$TREE" "$R" "$MODEL" "$BIN" \
        > "$out-driver.log" 2>&1
    rc=$?
    echo $rc > "$out.exit"
    if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
        echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy, sleeping 120s" >> "$R/lock-retries.log"
        rm -f "$out.exit"; rmdir "$out" 2>/dev/null
        sleep 120; continue
    fi
    break
done
echo "$(date -u +%FT%TZ) $cell rc=$rc $(cat "$R/stall/ev/exit.txt" 2>/dev/null)" >> "$R/progress.log"
echo "$(date -u +%FT%TZ) BOX-BATTERY-DONE" >> "$R/progress.log"
