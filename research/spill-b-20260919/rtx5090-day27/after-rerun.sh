#!/usr/bin/env bash
# Day 27 local: rerun the after cell (its first attempt hit the 3600 s lock bound) once chain-local.sh has finished its
# park runs; same driver, same shape (MEMRA_CTX=65536, order AB, deferred), its own flock wait.
set -uo pipefail
WT=/home/avifenesh/projects/wt-spill-b; R=$WT/research/spill-b-20260919/rtx5090-day27
cd "$WT" || exit 1
while pgrep -f "bash $R/chain-local.sh" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) after-ab rerun start" >> "$R/chain.log"
mv "$R/after-ab" "$R/after-ab-attempt1-lockbound" 2>/dev/null; mv "$R/after-ab.exit" "$R/after-ab-attempt1-lockbound.exit" 2>/dev/null
env MEMRA_CTX=65536 RIGDIR="$R" bash research/spill-b-20260919/run-day26-cell.sh after-ab AB > "$R/after-ab.launch.log" 2>&1
echo "$(date -u +%FT%TZ) after-ab rerun exit=$? (cell exit $(cat "$R/after-ab.exit" 2>/dev/null))" >> "$R/chain.log"
