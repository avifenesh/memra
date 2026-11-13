#!/usr/bin/env bash
# Rig: memra#1221 part 2's design measurement on one GB10 unit, one task per lease: the build of
# arms B, Z, O (box/spark_build.sh), and interleaved cells (box/specpub_cell.sh, no cache fill).
# A failed step stops the chain. WAITPID: start only after that process is gone.
# usage: [WAITPID=pid] [NOT_AFTER=HH:MM] drive_d.sh UNIT STEP...
#   STEP: build | CAMP:SET:TTL:CELL,CELL,...   (e.g. d260k:260k:20:01-B,02-Z)
set -u
UNIT=$1; shift
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-1221b/logs/drive-$UNIT.log
HEAD=$(cat "$HOME/.cache/memra-1221b/head.txt")
ARMS="B=B-$HEAD Z=Z-$HEAD O=O-$HEAD"
log() { echo "$(date -Is) $*" >> "$L"; }
[ -n "${WAITPID:-}" ] && { waitfor --gone "$WAITPID" --timeout 30m --quiet || { log "pid $WAITPID still alive; stopped"; exit 1; }; }
for s in "$@"; do
  if [ "$s" = build ]; then
    bash "$H/unit_cell.sh" "$UNIT" "build-$HEAD-$UNIT" 30 "bash box/spark_build.sh 25 B Z O && test -x bins/O-$HEAD/memra-server"
    rc=$?; log "build rc=$rc"; [ $rc = 0 ] || exit 1
    continue
  fi
  IFS=: read -r camp set ttl list <<< "$s"
  for c in ${list//,/ }; do
    bash "$H/unit_cell.sh" "$UNIT" "$camp-$c" "$ttl" "bash box/specpub_cell.sh $camp $c none none $set none $ARMS && grep -q 'SPECPUB-CELL-DONE $c' runs/$camp/campaign.log"
    rc=$?; log "$camp-$c rc=$rc"
    [ $rc = 0 ] || { log "$camp: $c failed (rc=$rc): stopped"; exit 1; }
  done
done
log "chain done: $*"
