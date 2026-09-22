#!/usr/bin/env bash
# Day 25 target-card runner (one RTX PRO 6000 Blackwell, collector rig `pro-single`, lock /tmp/memra-gpu.lock):
# waits for the tree's server build receipt, then (1) the Task 1 gate cells on the fixed contract fault gate
# and the identity gate over the lead's retire-seam settle, one collector hold each through day22-box-run.sh
# (its cell list argument; day22-cell.sh, cache 256 MB on the 27B), then (2) the retire-seam settle cell,
# ONE collector hold for its four boots (day25-retire-cell.sh). Bounded lock retries in both (75 x 120 s
# per cell; lane A holds the card part of today and is never inspected or signalled). No host, id or price.
# usage: day25-box-run.sh <tree> <receipts_root> <model.gguf> [gates|retire|both]
set -uo pipefail
TREE=$1; R=$2; MODEL=$3; WHICH=${4:-both}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
BIN=$TREE/target/release/memra-server
sha256sum "$BIN" > "$R/binary.sha256"
mkdir -p "$R/collector"
if [[ $WHICH == gates || $WHICH == both ]]; then
    bash research/spill-c-20260919/day22-box-run.sh "$TREE" "$R" "$MODEL" \
        fault-default fault-plain identity-default-off identity-default-on identity-plain-off identity-plain-on
fi
if [[ $WHICH == retire || $WHICH == both ]]; then
    cell=retire
    for attempt in $(seq 0 75); do
        out=$R/collector/$cell; [ "$attempt" -gt 0 ] && out=$R/collector/$cell-retry$attempt
        python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out "$out" --external-lock \
            --execute bash research/spill-c-20260919/day25-retire-cell.sh @COLLECTOR_LOCK_FD@ "$TREE" "$R" "$MODEL" "$BIN" \
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
    echo "$(date -u +%FT%TZ) $cell rc=$rc $(cat "$R/retire/ev/verdict.txt" 2>/dev/null)" >> "$R/progress.log"
fi
echo "$(date -u +%FT%TZ) BOX-BATTERY-DONE ($WHICH)" >> "$R/progress.log"
