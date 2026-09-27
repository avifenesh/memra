#!/usr/bin/env bash
# DAY68 section 9: the owed 5090 cells after the first chain, in order, each in its own bounded hold, from frozen copies
# only: R1's timed cell repeated once without builds (the lead's W ruling, DAY61 section 4: the first run's o2 boots
# overlapped DAY70's builds), item 16 with the warm-up doubled (DAY45 section 1's registered repeat), then S4's and V's
# halves (not run: the lock stayed busy 6 h each). Run as a copy: cp this file to
# /home/avifenesh/spill-a-cells/chain-day68b.sh and run that copy. Executed-not-qualified.
set -uo pipefail
S=/home/avifenesh/spill-a-cells
MODEL27=/home/avifenesh/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MODEL9=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$S/chain-day68b.log"; }
hold() { # $1 name: one bounded hold of the rig's lock on fd 9 (180 x 120 s), then no compute app and >= 20000 MiB free
  exec 9>/tmp/memra-5090.lock
  local held=0 attempt apps free
  for attempt in $(seq 1 180); do
    if flock -n 9; then held=1; break; fi
    log "$1: lock busy, attempt $attempt of 180, waiting 120 s"; sleep 120
  done
  [ "$held" = 1 ] || { log "NOT RUN ($1): the 5090 lock stayed busy"; return 2; }
  for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 | head -1)
    if [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] 2>/dev/null; then log "$1 taken"; return 0; fi
    log "$1: card not idle under the hold (apps=[${apps//$'\n'/; }] free=${free} MiB), attempt $attempt of 15"; sleep 60
  done
  log "NOT RUN ($1): the card never went idle under the hold"; flock -u 9; exec 9>&-; return 2
}
release() { flock -u 9; exec 9>&-; log "$1 released"; }
log "chain start"
# 1. R1's timed cell, once more (the gates stand from the first run; the repeat reads them from there).
R=$S/r1/receipts-repeat
mkdir -p "$R" && ln -sfn ../receipts/bins "$R/bins" && ln -sfn ../receipts/gates "$R/gates"
if hold "r1 repeat"; then
  ( while :; do echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-4 /proc/loadavg)"; sleep 5; done ) > "$R/host-load-5s.log" 2>&1 &
  LOADER=$!
  cd "$S/r1/tree" && A_OUT=$R A_TREE=$S/r1/tree MEMRA_DAY38_MODEL=$MODEL27 timeout 21600 bash "$S/r1/scripts/ab.sh" 9 seam \
    retire-seam-nosource retire-seam prime 448 > "$R/ab-r1-cell.out" 2>&1
  step_rc=$?; kill "$LOADER" 2>/dev/null; release "r1 repeat"; log "r1 repeat ab rc=$step_rc"
  cd "$S/r1/tree" && python3 research/spill-a-20260919/r1-reading.py "$R" > "$R/reading-r1.log" 2>&1
  step_rc=$?; log "r1 repeat reading rc=$step_rc $(tail -1 "$R/reading-r1.log")"
fi
# 2. item 16, the warm-up doubled (its own bounded hold inside the script).
T=$S/i16/tree; B=$S/i16/bins
A_TREE=$T bash "$S/i16/hot-hump-run-2x.sh" "$S/i16/cell-2x" "$MODEL9" "$B/base/memra-server" "$B/g4/memra-server" \
  "$B/g3/memra-server" "$B/gpp/memra-server" > "$S/i16/card-2x.out" 2>&1
step_rc=$?; log "item 16 2x rc=$step_rc $(tail -1 "$S/i16/cell-2x/run.log")"
# 3, 4. S4's and V's halves (their two holds inside the frozen driver).
bash "$S/s4/scripts/rtx5090-half-sv.sh" card s4 > "$S/s4/card.out" 2>&1; step_rc=$?; log "s4 card rc=$step_rc $(tail -1 "$S/s4/receipts/run.log")"
bash "$S/v/scripts/rtx5090-half-sv.sh" card v > "$S/v/card.out" 2>&1; step_rc=$?; log "v card rc=$step_rc $(tail -1 "$S/v/receipts/run.log")"
log "chain done"
