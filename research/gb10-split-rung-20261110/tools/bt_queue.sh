#!/usr/bin/env bash
# Rig: lever 1's served B/T on one GB10 unit, a whole context per unit, from a shared queue of
# contexts (bt_unit.sh with the arms and the contexts as arguments): when the unit is ours (a
# short lease taken first-come, so the other lanes' and landing tickets go first) it claims the
# next unclaimed context (runs/bt-claims-HEAD/CTX, mkdir is atomic) and runs its six cells there,
# B T T B B T, one lease per cell (split_driver.sh). Run once per unit with the same arms.
# usage: bt_queue.sh UNIT N BASE HEAD "CTX..." [WAIT_FILE]
#   N: the unit's number in the receipts; BASE, HEAD: bins/base-BASE and bins/head-HEAD on the
#   unit; WAIT_FILE: start once this file exists on the rig (the arms' binaries are in place).
set -u
UNIT=$1; N=$2; B=$3; H=$4; CTXS=$5; WAIT=${6:-}
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
Q=$R/runs/bt-claims-$H; mkdir -p "$Q"
L=$R/runs/bt-queue-unit$N.log
log() { echo "$(date -Is) $*" >> "$L"; }
[ -n "$WAIT" ] && { waitfor --file "$WAIT" --timeout 300m --quiet || { log "no $WAIT"; exit 1; }; }
log "start on unit $N: B base-$B, T head-$H, contexts $CTXS"
for ctx in $CTXS; do
  [ -d "$Q/$ctx" ] && continue
  id=$(sparklock acquire --unit "$UNIT" --ttl 2m --for "memra deep split rung served B/T: unit $N takes its next context" --wait 300m 2>>"$L")
  [ -n "$id" ] || { log "no lease"; exit 1; }
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  mkdir "$Q/$ctx" 2>/dev/null || { log "$ctx taken by the other unit"; continue; }
  echo "unit $N" > "$Q/$ctx/unit"
  case $ctx in 131k) TTL=20 ;; 260k) TTL=25 ;; *) TTL=15 ;; esac
  log "$ctx: sp${ctx%k}-unit$N-$H"
  TTL=$TTL bash "$T/split_driver.sh" "$UNIT" "sp${ctx%k}-unit$N-$H" "base-$B" "head-$H" "$ctx"
  log "$ctx rc=$?"
done
log "all done"
