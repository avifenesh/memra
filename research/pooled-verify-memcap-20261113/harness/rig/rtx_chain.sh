#!/usr/bin/env bash
# Rig: memra#1266's RTX 5090 cells, one gpulock lease per step line (TTL <= 30 min). Before each
# step the 5090 must have shown no "memra landing" or "release" lease or ticket for 3 minutes
# (observed), and the rig must be on AC with nv_tgp 150 and ppt_pl1_spl 140 (recorded, never set).
# usage: rtx_chain.sh STEPFILE   lines: NAME TTL_MIN -- CMD (run inside the lease)
set -u
L=<lane>/logs/rtx-chain.log
mkdir -p "$(dirname "$L")"
SEEN=<lane>/logs/landing-last-seen-5090
date +%s > "$SEEN"
quiet() { s=$(gpulock status 2>/dev/null) || return 1; now=$(date +%s)
  if echo "$s" | grep -qiE 'memra landing|release'; then echo "$now" > "$SEEN"; return 1; fi
  [ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ] || return 1
  [ "$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value)" = 150 ] || return 1
  [ "$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value)" = 140 ] || return 1
  [ $(( now - $(cat "$SEEN") )) -ge 180 ]; }
export -f quiet; export SEEN
echo "$(date -Is) chain start $1 pid $$" >> "$L"
while IFS= read -r line <&3; do
  [ -z "$line" ] && continue; case $line in \#*) continue ;; esac
  head=${line%% -- *}; cmd=${line#* -- }; set -- $head; name=$1; ttl=$2
  waitfor --until "bash -c quiet" --every 30s --timeout 300m --quiet >> "$L" 2>&1 || { echo "$(date -Is) $name: never quiet" >> "$L"; exit 1; }
  echo "$(date -Is) $name: queue for the lease" >> "$L"
  gpulock run --ttl "${ttl}m" --for "memra #1266 pooled verify OOM: 5090 $name (one step per lease)" --wait 240m -- bash -c "$cmd" >> "$L" 2>&1; rc=$?
  echo "$(date -Is) $name rc=$rc" >> "$L"
  [ $rc = 0 ] || [ "${CHAIN_KEEP_GOING:-0}" = 1 ] || exit 1
done 3< "$1"
echo "$(date -Is) rtx chain done: $1" >> "$L"
