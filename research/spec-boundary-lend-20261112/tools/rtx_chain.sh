#!/usr/bin/env bash
# Rig: memra#1221 part 2's RTX 5090 cells, one gpulock lease per cell. Before each cell the 5090
# must have shown no landing or release lease or ticket for 3 minutes (observed).
# usage: rtx_chain.sh STEPFILE   lines: NAME TTL_MIN -- CMD (run on the rig inside the lease)
set -u
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-1221b/logs/rtx-chain.log
SEEN=$HOME/.cache/memra-1221b/logs/landing-last-seen-5090
[ -f "$SEEN" ] || date +%s > "$SEEN"
quiet() { s=$(gpulock status 2>/dev/null) || return 1; now=$(date +%s)
  if echo "$s" | grep -qiE 'memra landing|release'; then echo "$now" > "$SEEN"; return 1; fi
  [ $(( now - $(cat "$SEEN") )) -ge 180 ]; }
export -f quiet; export SEEN
while IFS= read -r line <&3; do
  [ -z "$line" ] && continue; case $line in \#*) continue ;; esac
  head=${line%% -- *}; cmd=${line#* -- }; set -- $head; name=$1; ttl=$2
  waitfor --until "bash -c quiet" --every 30s --timeout 300m --quiet >> "$L" 2>&1 || { echo "$(date -Is) $name: never quiet" >> "$L"; exit 1; }
  gpulock run --ttl "${ttl}m" --for "memra #1221b spec boundary share: 5090 $name (one task per lease)" --wait 240m -- bash -c "$cmd" >> "$L" 2>&1; rc=$?
  echo "$(date -Is) $name rc=$rc" >> "$L"
  [ $rc = 0 ] || exit 1
done 3< "$1"
echo "$(date -Is) rtx chain done: $1" >> "$L"
