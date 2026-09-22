#!/usr/bin/env bash
# Day 29 target-card runner (one RTX PRO 6000 Blackwell, collector rig `pro-single`, lock /tmp/memra-gpu.lock):
# waits for BOTH build receipts (build-x.log for today's tree, build-y.log for the day-16 worktree), then runs
# day29-stall-cell.sh ONCE through tools/tier-battery.py --external-lock (one lock hold, the dry boot plus twenty
# boots), then day29-hitgate-cell.sh ONCE the same way (one hold, OFF and ON arms). Bounded lock retries per cell
# (60 x 120 s; lane A shares the card today, the holder is never inspected or signalled). No host, id or price here.
# usage: day29-box-run.sh <tree> <receipts_root> <model.gguf> <tree_y>
set -uo pipefail
TREE=$1; R=$2; MODEL=$3; TREE_Y=$4
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
for b in x y; do
    until grep -q '^rc=' "$R/build-$b.log" 2>/dev/null; do sleep 20; done
    if ! grep '^rc=' "$R/build-$b.log" | tail -1 | grep -q '^rc=0$'; then
        echo "build $b failed" | tee "$R/BUILD-FAILED-$b"; exit 1
    fi
done
BIN_X=$TREE/target/release/memra-server
BIN_Y=$TREE_Y/target/release/memra-server
mkdir -p "$R/collector"
run_cell() { # $1 cell name  $2 timeout  $3.. the command under the collector
    local cell=$1 timeout=$2; shift 2
    local attempt out rc
    for attempt in $(seq 0 60); do
        out=$R/collector/$cell; [ "$attempt" -gt 0 ] && out=$R/collector/$cell-retry$attempt
        python3 tools/tier-battery.py --rig pro-single --timeout "$timeout" --out "$out" --external-lock \
            --execute "$@" > "$out-driver.log" 2>&1
        rc=$?
        echo $rc > "$out.exit"
        if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
            echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy, sleeping 120s" >> "$R/lock-retries.log"
            rm -f "$out.exit"; rmdir "$out" 2>/dev/null
            sleep 120; continue
        fi
        break
    done
    echo "$(date -u +%FT%TZ) $cell rc=$rc" >> "$R/progress.log"
    return "$rc"
}
run_cell stall-cell-i 9000 bash research/spill-c-20260919/day29-stall-cell.sh @COLLECTOR_LOCK_FD@ "$TREE" "$R" "$MODEL" "$BIN_X" "$BIN_Y" "$TREE_Y"
echo "$(date -u +%FT%TZ) stall $(cat "$R/stall/ev/exit.txt" 2>/dev/null)" >> "$R/progress.log"
run_cell hitgate 3600 bash research/spill-c-20260919/day29-hitgate-cell.sh @COLLECTOR_LOCK_FD@ "$TREE" "$R" "$MODEL" "$BIN_X"
echo "$(date -u +%FT%TZ) hitgate $(cat "$R/hitgate/exit.txt" 2>/dev/null)" >> "$R/progress.log"
echo "$(date -u +%FT%TZ) BOX-BATTERY-DONE" >> "$R/progress.log"
