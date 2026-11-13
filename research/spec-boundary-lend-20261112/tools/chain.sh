#!/usr/bin/env bash
# Rig: a chain of memra#1221 part 2 steps on one GB10 unit, one task per lease (unit_cell.sh: the
# landing/release yield, the 3-minute quiet rule, the cell's process group stopped before its
# lease ends). A failed step stops the chain unless it is marked "may-fail".
# usage: [WAITPID=pid] chain.sh UNIT STEPFILE    STEPFILE lines: NAME TTL_MIN [may-fail] -- REMOTE CMD
set -u
UNIT=$1; STEPS=$2
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-1221b/logs/chain-$UNIT.log
log() { echo "$(date -Is) $*" >> "$L"; }
[ -n "${WAITPID:-}" ] && { waitfor --gone "$WAITPID" --timeout 120m --quiet || { log "pid $WAITPID still alive; stopped"; exit 1; }; }
while IFS= read -r line <&3; do
  [ -z "$line" ] && continue
  case $line in \#*) continue ;; esac
  head=${line%% -- *}; cmd=${line#* -- }
  set -- $head; name=$1; ttl=$2; mayfail=${3:-}
  bash "$H/unit_cell.sh" "$UNIT" "$name" "$ttl" "$cmd"
  rc=$?; log "$name rc=$rc"
  if [ $rc != 0 ] && [ "$mayfail" != may-fail ]; then log "chain stopped at $name"; exit 1; fi
done 3< "$STEPS"
log "chain done: $STEPS"
