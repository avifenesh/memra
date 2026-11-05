#!/usr/bin/env bash
# kv-vmm Max-Q cell (DAY37 O1, the --kv-allocator vmm deciding cell, replayed at the r4 tree on an RTX PRO 6000
# Blackwell Max-Q): the UNTIMED half, for a gap on one card while the other cards train.
#
#   CARD=<idx> bash /mnt/Data/work/memra-maxq/kv-vmm/run.sh
#
# Run it from the session that holds the card's window: FD 9 open and flock-held on /mnt/Data/work/evals/gpu<idx>.lock.
# Runs A2, the A1 gate set (both arms) and the 19 untimed serving boots (A1-MIX, A4, A5, A6, A7) under one hold of
# /tmp/memra-gpu.lock, pinned to the card (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD), nice 10, a total
# deadline (WINDOW_MIN, default 225 min) and the Max-Q thermal guard. Then the reader and the rule: out-*/VERDICT.txt.
# The timed half (stage 0, the stream pairs: A3 (i), A3 (ii), A1-STREAM) is run-timed.sh, in a quiet slot, into the
# same out dir.
#
# Resume after a deadline or a stop:  OUT=<the out dir> CARD=<idx> bash .../run.sh   (finished steps are skipped)
# Check the kit without touching the card:  DRY_RUN=1 CARD=<idx> bash .../run.sh   (no lock, no GPU work)
set -uo pipefail
. "$(dirname "$0")/lib.sh"
kit_nice "$@"
kit_env
new_out
DRY=${DRY_RUN:-0}
WINDOW_MIN=${WINDOW_MIN:-225}
[[ "$WINDOW_MIN" =~ ^[0-9]+$ ]] || refuse "WINDOW_MIN must be minutes"
START=$(date +%s); DEADLINE=$((START + WINDOW_MIN * 60))
echo "$(date -u +%FT%TZ) run.sh start card=$CARD out=$OUT resumed=$RESUMED dry=$DRY window_min=$WINDOW_MIN deadline=$(date -u -d @"$DEADLINE" +%FT%TZ) dropped_env=[${KIT_DROPPED}]" >> "$OUT/chain.log"
echo "out: $OUT"

check_static
if [ "$DRY" = 1 ]; then
  check_window report
  check_card report dry
  check_model
  box_snapshot dry
  echo "dry run done: static checks passed (card and window lines above say whether a run would start now); nothing was started"; exit 0
fi
check_window refuse
check_card refuse untimed
check_model
take_lock
box_snapshot start

KIT_RUN_ID=kvvmm-$(date -u +%Y%m%dT%H%M%SZ)-$$
trap cleanup EXIT
trap 'exit 130' INT TERM; trap 'exit 129' HUP
start_watch untimed "$DEADLINE"
log "run marker $KIT_RUN_ID; telemetry pid $TELEM_PID, watchdog pid $WATCH_PID"
env KIT_RUN_ID="$KIT_RUN_ID" bash "$K/chain-untimed.sh" &
CHAIN=$!
wait "$CHAIN"; rc=$?
log "untimed chain rc=$rc$( [ -s "$OUT/STOP-REASON" ] && echo " stopped: $(tail -1 "$OUT/STOP-REASON")")"
read_and_rule
grep '^DAY37 VERDICT' "$OUT/VERDICT.txt" || cat "$OUT/VERDICT.txt"
if [ -s "$OUT/STOP-REASON" ] || ! grep -q 'MAXQ-KVVMM-UNTIMED-DONE' "$OUT/chain.log"; then
  echo "not finished; resume in the next window: OUT=$OUT CARD=<idx> bash $K/run.sh"
else
  echo "untimed half done; timed half (quiet box): OUT=$OUT CARD=<idx> bash $K/run-timed.sh"
fi
echo "out: $OUT"
exit "$rc"
