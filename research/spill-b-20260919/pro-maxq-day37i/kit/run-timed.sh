#!/usr/bin/env bash
# kv-vmm Max-Q cell: the TIMED half, for a quiet box (no compute process on any card; training stopped).
#
#   OUT=<out dir of run.sh> CARD=<idx> bash /mnt/Data/work/memra-maxq/kv-vmm/run-timed.sh
#
# Stage 0 as a recorded reading, then the 10 A3 stream pairs (A3 (i), A3 (ii), A1-STREAM), into the untimed half's out
# dir; then the reader and the rule over the whole root (VERDICT.txt). Same window lock (FD 9 on
# /mnt/Data/work/evals/gpu<idx>.lock), canonical lock hold, card pinning, nice 10, thermal guard and cleanup as run.sh.
# Refuses unless every card of the box is free of compute processes (the registered box-wide idle check), and on
# GPU 0 (airflow-limited on this box) unless ALLOW_GPU0=1. Total deadline WINDOW_MIN (default 120 min).
# Without OUT it starts a fresh root (the timed half alone; the untimed half can then run into it with OUT=).
set -uo pipefail
. "$(dirname "$0")/lib.sh"
kit_nice "$@"
kit_env
new_out
WINDOW_MIN=${WINDOW_MIN:-120}
[[ "$WINDOW_MIN" =~ ^[0-9]+$ ]] || refuse "WINDOW_MIN must be minutes"
START=$(date +%s); DEADLINE=$((START + WINDOW_MIN * 60))
echo "$(date -u +%FT%TZ) run-timed.sh start card=$CARD out=$OUT resumed=$RESUMED window_min=$WINDOW_MIN deadline=$(date -u -d @"$DEADLINE" +%FT%TZ) dropped_env=[${KIT_DROPPED}]" >> "$OUT/chain.log"
echo "out: $OUT"
if [ "$CARD" = 0 ]; then
  [ "${ALLOW_GPU0:-0}" = 1 ] || refuse "GPU 0 is airflow-limited on this box: run the timed half on another card, or set ALLOW_GPU0=1"
  echo "$(date -u +%FT%TZ) ALLOW_GPU0=1: timed half on GPU 0" >> "$OUT/chain.log"
fi
check_static
check_window refuse
check_card refuse timed
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id,used_memory --format=csv,noheader)
[ -z "$busy" ] || refuse "the box is not quiet (compute processes on: $(echo "$busy" | tr '\n' ';')); the timed half needs every card idle"
check_model
take_lock
box_snapshot timed-start
if [ -e "$OUT/card.csv" ] && [ -e "$OUT/preflight/card-untimed.csv" ]; then
  echo "$(date -u +%FT%TZ) untimed half card: $(tail -1 "$OUT/preflight/card-untimed.csv" | cut -d, -f1,5); timed half card: $CARD,${POWER_LIMIT}" >> "$OUT/chain.log"
fi

KIT_RUN_ID=kvvmm-timed-$(date -u +%Y%m%dT%H%M%SZ)-$$
trap cleanup EXIT
trap 'exit 130' INT TERM; trap 'exit 129' HUP
start_watch timed "$DEADLINE"
log "run marker $KIT_RUN_ID; telemetry pid $TELEM_PID, watchdog pid $WATCH_PID"
env KIT_RUN_ID="$KIT_RUN_ID" KIT_IDLE_SCOPE=box bash "$K/chain-timed.sh" &
CHAIN=$!
wait "$CHAIN"; rc=$?
log "timed chain rc=$rc$( [ -s "$OUT/STOP-REASON" ] && echo " stopped: $(tail -1 "$OUT/STOP-REASON")")"
read_and_rule
grep '^DAY37 VERDICT' "$OUT/VERDICT.txt" || cat "$OUT/VERDICT.txt"
if [ -s "$OUT/STOP-REASON" ] || ! grep -q 'MAXQ-KVVMM-TIMED-DONE' "$OUT/chain.log"; then
  echo "timed half not finished; rerun in the next quiet slot: OUT=$OUT CARD=<idx> bash $K/run-timed.sh"
fi
echo "out: $OUT"
exit "$rc"
