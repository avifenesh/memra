#!/usr/bin/env bash
# DAY68 sections 11 and 12: the owed 5090 cells rebuilt and run after the scratch directory was removed whole at
# 19:23:49Z on 2026-09-27 (section 12). Every build first, outside every hold, from a frozen snapshot of the lane (a
# detached worktree at the commit that holds this file); then each half in its own bounded hold(s), in the lead's order:
# R1 (its card whole: the gates, then the timed cell the lead's ruling asks to repeat), P2, T-H', then item 16 with the
# warm-up doubled, S4 and V. The cell root sits under ~/.local/share since DAY68 section 12's second start, off the home
# directory's top level that disk-cleanup sessions list. Run as a copy: cp this file to
# /home/avifenesh/.local/share/memra-lane-a-cells/chain-day68c.sh and run that copy with the lane commit as its argument.
# Executed-not-qualified.
set -uo pipefail
S=/home/avifenesh/.local/share/memra-lane-a-cells
LANE=/home/avifenesh/projects/wt-spill-a
COMMIT=${1:?lane commit}
MODEL9=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$S/chain-day68c.log"; }
mkdir -p "$S"
cat > "$S/IN-USE.txt" <<TXT
In use by WP-A (lane/spill-a-20260919), DAY68 sections 11 and 12: the 5090 halves' trees, builds and receipts, driven
by $S/chain-day68c.sh from $(date -u +%FT%TZ). Removed by WP-A's own clean step when the chain is read, not by a sweep.
TXT
log "chain start (lane snapshot $COMMIT)"
git -C "$LANE" worktree add -q --detach "$S/lane-snap" "$COMMIT" || { log "REFUSED: lane snapshot"; exit 2; }
L=$S/lane-snap/research/spill-a-20260919
# Builds, in order, none inside a hold.
bash "$L/rtx5090-half.sh" prepare r1; log "r1 build: $(tail -1 "$S/r1/receipts/build.log" 2>&1)"
bash "$L/rtx5090-half-sv.sh" prepare p2; log "p2 build: $(tail -1 "$S/p2/receipts/build.log" 2>&1) unit: $(tail -1 "$S/p2/receipts/unit/build.log" 2>&1)"
bash "$L/rtx5090-half.sh" prepare th2; log "th2 build: $(tail -1 "$S/th2/receipts/build.log" 2>&1)"
mkdir -p "$S/i16/bins"
git -C "$LANE" worktree add -q --detach "$S/i16/tree" 7b849a817 && cp "$L/rtx5090-day45/hot-hump-run-2x.sh" "$S/i16/" \
  && bash "$S/i16/tree/research/spill-a-20260919/rtx5090-day45/build.sh" "$S/i16/bins" > "$S/i16/build.out" 2>&1
log "i16 build: $(tail -1 "$S/i16/build.out" 2>&1)"
bash "$L/rtx5090-half-sv.sh" prepare s4; log "s4 build: $(tail -1 "$S/s4/receipts/build.log" 2>&1)"
bash "$L/rtx5090-half-sv.sh" prepare v; log "v build: $(tail -1 "$S/v/receipts/build.log" 2>&1)"
log "builds done"
# The cards, each from its frozen copy.
bash "$S/r1/scripts/rtx5090-half.sh" card r1 > "$S/r1/card.out" 2>&1; step_rc=$?; log "r1 card rc=$step_rc $(tail -1 "$S/r1/receipts/run.log")"
bash "$S/p2/scripts/rtx5090-half-sv.sh" card p2 > "$S/p2/card.out" 2>&1; step_rc=$?; log "p2 card rc=$step_rc $(tail -1 "$S/p2/receipts/run.log")"
bash "$S/th2/scripts/rtx5090-half.sh" card th2 > "$S/th2/card.out" 2>&1; step_rc=$?; log "th2 card rc=$step_rc $(tail -1 "$S/th2/receipts/run.log")"
T=$S/i16/tree; B=$S/i16/bins
A_TREE=$T bash "$S/i16/hot-hump-run-2x.sh" "$S/i16/cell-2x" "$MODEL9" "$B/base/memra-server" "$B/g4/memra-server" \
  "$B/g3/memra-server" "$B/gpp/memra-server" > "$S/i16/card-2x.out" 2>&1
step_rc=$?; log "item 16 2x rc=$step_rc $(tail -1 "$S/i16/cell-2x/run.log")"
bash "$S/s4/scripts/rtx5090-half-sv.sh" card s4 > "$S/s4/card.out" 2>&1; step_rc=$?; log "s4 card rc=$step_rc $(tail -1 "$S/s4/receipts/run.log")"
bash "$S/v/scripts/rtx5090-half-sv.sh" card v > "$S/v/card.out" 2>&1; step_rc=$?; log "v card rc=$step_rc $(tail -1 "$S/v/receipts/run.log")"
log "chain done"
