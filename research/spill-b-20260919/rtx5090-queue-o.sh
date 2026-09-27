#!/usr/bin/env bash
# WP-B local 5090 queue-o (2026-09-27): asks again for DAY48's local boots after O1-enforce (the runner stopped when
# O1-enforce-vg's idle wait ran out while another lane held the card). Same runner, specs, binary and receipt root, in
# the tree the half started in (wt-b-integ71); queue-m's rule (a started spec never runs again; an exit 3 is asked
# again, up to eight times). Never a signal to anything.
set -uo pipefail
export WT=$HOME/projects/wt-b-integ71
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day48
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
started() { command grep -q "boot $1 start" "$R/run.log" 2>/dev/null; }
specs=(O1-enforce:enforce O1-enforce-vg:enforce-vg O2-enforce-vg:enforce-vg O2-enforce:enforce)
for try in 1 2 3 4 5 6 7 8; do
  left=(); for spec in "${specs[@]}"; do started "${spec%%:*}" || left+=("$spec"); done
  [ ${#left[@]} = 0 ] && break
  log "queue-o: asking for ${left[*]} (try $try)"
  BIN=$WT/target/day48/tip/memra-server MODEL=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf \
    MODEL_KEY=o15 YIELD_S=240 CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64" \
    bash "$D/day48-run.sh" "$R" "${left[@]}"
  rc=$?
  [ $rc = 3 ] || break
done
python3 "$D/day48-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
log "queue-o done"
