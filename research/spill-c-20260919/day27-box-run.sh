#!/usr/bin/env bash
# day27-box-run.sh <tree> <receipts_root> <model.gguf>: the hit gate OFF and ON on the target card (one RTX PRO
# 6000 Blackwell, 600 W, the 27B) on the tree whose hit gate arms the host tier in its door ON arm (C day 27).
# Waits for the tree's server build receipt, then runs day27-cell.sh per cell with MEMRA_GPU_LOCK set to the
# canonical /tmp/memra-gpu.lock (the gate has no `--external-lock` and takes the lock itself per boot, as lane
# A day 21 ran it; the collector is not in the chain for this gate, stated in DAY27.md). Before each cell the
# driver waits, bounded, for a free lock and records the card before and after. No host, id or price here.
set -uo pipefail
TREE=$1; R=$2; MODEL=$3
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$' || { echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; }
BIN=$TREE/target/release/memra-server
sha256sum "$BIN" > "$R/binary.sha256"
mkdir -p "$R/cells"
for cell in hit-off hit-on; do
    echo "$(date -u +%FT%TZ) start $cell tree=$(git rev-parse HEAD)" >> "$R/progress.log"
    bash research/spill-c-20260919/day27-cell.sh "$cell" "$MODEL" "$BIN" "$R/cells" >> "$R/progress.log" 2>&1
    echo "$(date -u +%FT%TZ) done $cell rc=$(cat "$R/cells/$cell/gate.exit")" >> "$R/progress.log"
done
echo "$(date -u +%FT%TZ) BOX-BATTERY-DONE" >> "$R/progress.log"
