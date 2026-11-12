#!/usr/bin/env bash
# Rig: served cells of one campaign on one unit, one sparklock lease per cell (unit_cell.sh).
# A cell NN-X runs the binary under bins/ that ARMS maps X to; a failed cell stops the campaign.
# PRE is a command run on the unit before each cell, inside its lease (fill_cache.sh: the forced
# trim condition); its output goes to the cell log.
# MID is a command specpub_cell.sh runs once the server is ready, before the requests.
# usage: [NOT_AFTER=HH:MM] [PRE="cmd"] [MID="cmd"] ab_run.sh UNIT CAMPAIGN SET "X=DIR ..." CELL...   (SET 131k or 260k)
set -u
UNIT=$1; CAMP=$2; SET=$3; ARMS=$4; shift 4
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-1221/logs/measure-$UNIT.log
log() { echo "$(date -Is) $*" >> "$L"; }
ttl=15; [ "$SET" = 260k ] && ttl=20
for c in "$@"; do
  bash "$H/unit_cell.sh" "$UNIT" "$CAMP-$c" $ttl "${PRE:+$PRE; }${MID:+MID='$MID' }bash box/specpub_cell.sh $CAMP $c none none $SET none $ARMS && grep -q 'SPECPUB-CELL-DONE $c' runs/$CAMP/campaign.log"; rc=$?
  log "$CAMP-$c rc=$rc"
  [ $rc = 0 ] || { log "$CAMP: $c failed (rc=$rc): stopped"; exit 1; }
done
log "$CAMP done: $*"
