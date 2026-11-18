#!/bin/bash
# memra#1258 part 2 (PR #1321), review round 1: the fold's cells on ONE RTX PRO 6000 Blackwell
# Max-Q card, run under tools/memra-gpu-run (which holds the box lock and the card's lock and
# sets CUDA_VISIBLE_DEVICES to the card's UUID). Untimed cells: they read memory and outputs.
#
# usage: maxq-cells.sh STEPS_FILE
#   STEPS_FILE: one step per line, LABEL<TAB>MODE<TAB>COMMAND. MODE is `emul24` (the card is held
#   down to MAXQ_FREE_MIB of free memory by a fixed ballast for the whole step: a 24 GB card's
#   budget), `emul:N` (held down to N MiB free), `emul:@FILE` (N is read from FILE, which an
#   earlier step wrote) or `roomy` (no ballast). COMMAND runs with OUT set to the step's output
#   directory.
#   A step that ended (any exit status) is written `DONE label` in progress.log and is skipped by
#   the next run, so the resume point is: start the same command again.
#
# Stop: TERM to this script (memra-gpu-run forwards its own). The running step's whole process
# session gets TERM, then KILL after MAXQ_KILL_AFTER seconds; the ballast goes; the step's
# directory is moved to cut/ (it has no DONE line and runs again from zero); exit 143.
#
# Heat guard: every MAXQ_HEAT_EVERY seconds the card's temperature and fan are read; three
# samples in a row at MAXQ_HEAT_C or more with the fan at MAXQ_HEAT_FAN percent or more (or a fan
# that cannot be read) stop the run the same way, with a HEAT-STOP line. Samples go to heat.log.
#
# Environment: MAXQ_KIT (default ~/.cache/memra-1321-maxq), MAXQ_FREE_MIB (24390: total less used
# on the RTX 5090 Laptop before a boot), MAXQ_HEAT_C (92), MAXQ_HEAT_FAN (95), MAXQ_HEAT_EVERY (5),
# MAXQ_KILL_AFTER (15), MAXQ_BALLAST_CMD (the holder; default the harness's --ballast-hold),
# MAXQ_STOP_AT (a time in `date -d` form, such as 2026-11-18T02:53: the run stops itself then,
# the same way as on TERM, with a STOP-AT line; for a card that is owed to someone at a set time).
set -u
K=${MAXQ_KIT:-$HOME/.cache/memra-1321-maxq}
STEPS=${1:?steps file}
U=${CUDA_VISIBLE_DEVICES:?run under tools/memra-gpu-run}
FREE_MIB=${MAXQ_FREE_MIB:-24390}
HEAT_C=${MAXQ_HEAT_C:-92}; HEAT_FAN=${MAXQ_HEAT_FAN:-95}; HEAT_EVERY=${MAXQ_HEAT_EVERY:-5}
KILL_AFTER=${MAXQ_KILL_AFTER:-15}
LOG=$K/progress.log
mkdir -p "$K/out" "$K/cut"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
card() { nvidia-smi -i "$U" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }

STEP=""; BAL=""; HEAT=""; TIMER=""; OUT=""; LABEL=""
session_pids() { ps -eo pid=,sid= | awk -v s="$1" '$2 == s { print $1 }'; }
stop_group() {  # TERM to every process of a session, KILL after KILL_AFTER seconds
  # A step and the ballast holder are session leaders (setsid). The session, not the process
  # group: `timeout` puts its command in a group of its own, and a group kill misses it (the
  # first run on the card left a run-gen behind that way).
  local sid=$1 i pids
  pids=$(session_pids "$sid")
  # shellcheck disable=SC2086
  [ -z "$pids" ] || kill -TERM $pids 2>/dev/null
  for i in $(seq "$KILL_AFTER"); do [ -z "$(session_pids "$sid")" ] && return 0; sleep 1; done
  pids=$(session_pids "$sid")
  # shellcheck disable=SC2086
  [ -z "$pids" ] || kill -KILL $pids 2>/dev/null
}
ballast_stop() {
  [ -n "$BAL" ] || return 0
  exec 9>&- 2>/dev/null
  stop_group "$BAL"
  wait "$BAL" 2>/dev/null
  BAL=""
}
ballast_start() {  # hold the card down to $1 MiB free (nvidia-smi's total less used); 0 when held
  local fifo=$K/ballast.fifo total used bytes i FREE_MIB=$1
  rm -f "$fifo"; mkfifo "$fifo" || return 1
  : > "$OUT/ballast-fixed.log"
  # shellcheck disable=SC2086
  setsid ${MAXQ_BALLAST_CMD:-python3 $K/harness/same-pass-admission-gate.py --ballast-hold} < "$fifo" >> "$OUT/ballast-fixed.log" 2>&1 &
  BAL=$!
  exec 9> "$fifo"
  for i in $(seq 60); do grep -q '^BALLAST READY' "$OUT/ballast-fixed.log" && break; sleep 0.5; done
  grep -q '^BALLAST READY' "$OUT/ballast-fixed.log" || { log "ballast: the holder did not come up"; return 1; }
  IFS=, read -r total used < <(card memory.total,memory.used)
  bytes=$(( (total - used - FREE_MIB) * 1048576 ))
  [ "$bytes" -gt 0 ] || { log "ballast: the card has only $(( total - used )) MiB free, under $FREE_MIB"; return 1; }
  echo "$bytes" >&9
  for i in $(seq 120); do grep -q '^BALLAST \(HELD\|FAILED\)' "$OUT/ballast-fixed.log" && break; sleep 0.5; done
  grep -q "^BALLAST HELD total=$bytes\$" "$OUT/ballast-fixed.log" || { log "ballast: not held: $(tail -n 1 "$OUT/ballast-fixed.log")"; return 1; }
  IFS=, read -r total used < <(card memory.total,memory.used)
  echo "fixed ballast $(( bytes / 1048576 )) MiB held; card total $total MiB used $used MiB free $(( total - used )) MiB (target $FREE_MIB)" | tee -a "$OUT/guard.log"
}
finish() {  # stop everything this script started; the cut step's directory goes aside
  trap '' TERM INT
  [ -n "$HEAT" ] && kill "$HEAT" 2>/dev/null
  [ -n "$TIMER" ] && stop_group "$TIMER"
  [ -n "$STEP" ] && stop_group "$STEP"
  ballast_stop
  if [ -n "$STEP" ] && [ -d "$OUT" ]; then
    mv "$OUT" "$K/cut/$(basename "$OUT")-cut$(date +%H%M%S)"
    log "cut: $LABEL (its directory is under cut/, no DONE line)"
  fi
}
trap 'log "TERM: stopping"; finish; log "stopped. Resume: the same command (DONE steps are skipped)"; exit 143' TERM INT

# The heat guard: its own process, it signals this script.
(
  n=0; k=0
  while :; do
    IFS=, read -r t f p < <(card temperature.gpu,fan.speed,power.draw)
    hot=0
    case "$t" in ''|*[!0-9]*) ;; *)
      if [ "$t" -ge "$HEAT_C" ]; then
        case "$f" in ''|*[!0-9]*) hot=1 ;; *) [ "$f" -ge "$HEAT_FAN" ] && hot=1 ;; esac
      fi ;;
    esac
    if [ "$hot" = 1 ]; then n=$(( n + 1 )); else n=0; fi
    if [ "$hot" = 1 ] || [ $(( k % 12 )) = 0 ]; then echo "$(date -Is) temp=$t fan=$f power=$p hot_in_a_row=$n" >> "$K/heat.log"; fi
    if [ "$n" -ge 3 ]; then
      echo "$(date -Is) HEAT-STOP: $n samples in a row at ${HEAT_C} C or more with the fan at ${HEAT_FAN}% or more (temp=$t fan=$f)" | tee -a "$K/heat.log" >> "$LOG"
      kill -TERM $$
      exit 0
    fi
    k=$(( k + 1 )); sleep "$HEAT_EVERY"
  done
) &
HEAT=$!

echo "$$" > "$K/driver.pid"  # TERM to this pid stops the run
if [ -n "${MAXQ_STOP_AT:-}" ]; then
  at=$(date -d "$MAXQ_STOP_AT" +%s) || { echo "MAXQ_STOP_AT: not a time: $MAXQ_STOP_AT" >&2; kill "$HEAT"; exit 2; }
  left=$(( at - $(date +%s) ))
  [ "$left" -gt 0 ] || { log "MAXQ_STOP_AT $MAXQ_STOP_AT is already past; nothing run"; kill "$HEAT"; exit 3; }
  # A session of its own, so that stopping it takes its sleep along.
  setsid bash -c 'sleep "$1"; echo "$(date -Is) STOP-AT $4 reached: stopping" >> "$2"; kill -TERM "$3"' _ "$left" "$LOG" "$$" "$MAXQ_STOP_AT" &
  TIMER=$!
fi
log "run start pid $$ card $U kit $K steps $STEPS (free target for emul24: $FREE_MIB MiB${MAXQ_STOP_AT:+; stops itself at $MAXQ_STOP_AT})"
rc_all=0
while IFS=$'\t' read -r LABEL mode cmd <&7; do
  [ -z "$LABEL" ] && continue
  case "$LABEL" in \#*) continue ;; esac
  if grep -qxF "DONE $LABEL" "$LOG" 2>/dev/null; then continue; fi
  OUT=$K/out/$LABEL
  [ ! -e "$OUT" ] || mv "$OUT" "$K/cut/$LABEL-stale$(date +%H%M%S)"
  mkdir -p "$OUT"
  { date -Is; echo "step $LABEL mode $mode"; echo "$cmd"
    nvidia-smi -i "$U" --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,fan.speed,power.draw,power.limit --format=csv,noheader; } > "$OUT/guard.log"
  log "start $LABEL ($mode)"
  case "$mode" in
    emul24) want=$FREE_MIB ;;
    emul:@*) want=$(tr -d ' \n' < "${mode#emul:@}" 2>/dev/null) ;;
    emul:*) want=${mode#emul:} ;;
    *) want="" ;;
  esac
  case "$mode" in emul*)
    case "$want" in ''|*[!0-9]*) log "rc=96 $LABEL (no free-memory target for mode $mode; step not run)"; rc_all=1; rm -f "$OUT/guard.log"; rmdir "$OUT" 2>/dev/null; continue ;; esac
    ballast_start "$want" || { ballast_stop; log "rc=97 $LABEL (the fixed ballast could not be placed; step not run)"; rc_all=1; rm -f "$OUT/guard.log" "$OUT/ballast-fixed.log"; rmdir "$OUT" 2>/dev/null; continue; }
    ;;
  esac
  OUT=$OUT KIT=$K setsid bash -c "$cmd" > "$OUT/step.log" 2>&1 < /dev/null &
  STEP=$!
  wait "$STEP"; rc=$?
  # A step leaves nothing behind: its session is ended whatever its status.
  stop_group "$STEP"; STEP=""
  ballast_stop
  { echo "step rc=$rc $(date -Is)"; nvidia-smi -i "$U" --query-gpu=memory.used,temperature.gpu,fan.speed --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -F "$U"; } >> "$OUT/guard.log"
  log "rc=$rc $LABEL"
  echo "DONE $LABEL" >> "$LOG"
  [ "$rc" = 0 ] || rc_all=1
done 7< "$STEPS"
kill "$HEAT" 2>/dev/null
[ -n "$TIMER" ] && stop_group "$TIMER"
log "RUN-END rc=$rc_all"
exit $rc_all
