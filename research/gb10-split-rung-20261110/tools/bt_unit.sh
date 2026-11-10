#!/usr/bin/env bash
# Rig: lever 1's served B/T (B = main 42bd4b07e4, T = the final head 9592c2dceb) on one GB10
# unit, a whole context per unit: when the unit is ours (a short lease taken first-come, so the
# other lanes' and landing tickets go first) it claims the next context nobody has claimed
# (runs/bt-claims/CTX, mkdir is atomic) and runs its six cells there (split_driver.sh, one lease
# per cell). Run once per unit; the two take the contexts 131k, 260k, 32k, 12k between them.
# usage: bt_unit.sh UNIT N [WAIT_PID]   (N: the unit's number in the receipts)
set -u
UNIT=$1; N=$2; WAITPID=${3:-}
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
H=9592c2dceb; B=42bd4b07e4
Q=$R/runs/bt-claims; mkdir -p "$Q"
L=$R/runs/bt-unit$N.log
log() { echo "$(date -Is) $*" >> "$L"; }
[ -n "$WAITPID" ] && waitfor --gone "$WAITPID" --timeout 300m --quiet
log "start on $UNIT"
for ctx in 131k 260k 32k 12k; do
  [ -d "$Q/$ctx" ] && continue
  id=$(sparklock acquire --unit "$UNIT" --ttl 2m --for "memra deep split rung served B/T: unit $N takes its next context" --wait 300m 2>>"$L")
  [ -n "$id" ] || { log "no lease"; exit 1; }
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  mkdir "$Q/$ctx" 2>/dev/null || { log "$ctx taken by the other unit"; continue; }
  echo "$UNIT" > "$Q/$ctx/unit"
  case $ctx in 131k) TTL=20 ;; 260k) TTL=25 ;; *) TTL=15 ;; esac
  log "$ctx: sp${ctx%k}-unit$N"
  TTL=$TTL bash "$T/split_driver.sh" "$UNIT" "sp${ctx%k}-unit$N" "base-$B" "head-$H" "$ctx"
  log "$ctx rc=$?"
done
log "all done"
