#!/usr/bin/env bash
# DAY68: the owed 5090 cells in order, each in its own bounded hold, from frozen copies only: R1's half (section 1),
# L''s half (section 1), then item 16 (section 3, DAY45's cell from its registration tree 7b849a817). Run as a copy:
# cp this file to /home/avifenesh/spill-a-cells/chain-day68.sh and run that copy. Executed-not-qualified.
set -uo pipefail
S=/home/avifenesh/spill-a-cells
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$S/chain-day68.log"; }
log "chain start"
bash "$S/r1/scripts/rtx5090-half.sh" card r1 > "$S/r1/card.out" 2>&1; log "r1 card rc=$? $(tail -1 "$S/r1/receipts/run.log")"
bash "$S/l2/scripts/rtx5090-half.sh" card l2 > "$S/l2/card.out" 2>&1; log "l2 card rc=$? $(tail -1 "$S/l2/receipts/run.log")"
T=$S/i16/tree; B=$S/i16/bins
grep -q BUILD-DONE "$S/i16/build.out" || { log "item 16 build receipt missing"; exit 2; }
bash "$T/research/spill-a-20260919/rtx5090-day45/hot-hump-run.sh" "$S/i16/cell" \
  /home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
  "$B/base/memra-server" "$B/g4/memra-server" "$B/g3/memra-server" "$B/gpp/memra-server" > "$S/i16/card.out" 2>&1
log "item 16 rc=$? $(tail -1 "$S/i16/cell/run.log")"
log "chain done"
