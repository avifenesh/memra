#!/usr/bin/env bash
# Rig: memra#1266's RTX 5090 cells, one gpulock lease per step line (TTL <= 30 min). Before each
# step the 5090 must have shown no "memra landing" or "release" lease or ticket for 3 minutes
# (observed), and the rig must be on AC with nv_tgp 150 and ppt_pl1_spl 140 (recorded, never set).
# A ticket of ours that is still waiting is withdrawn as soon as such a lease or ticket shows.
# usage: rtx_chain.sh STEPFILE   lines: NAME TTL_MIN [perf] -- CMD (run inside the lease). "perf" marks
# a timed cell: a --perf lease (acquire, renewed by timed_take.sh, which applies the owner's take
# rule); the others (gates) run in a gpulock run lease of TTL_MIN on the lease default, balanced.
set -u
L=<lane>/logs/rtx-chain.log
mkdir -p "$(dirname "$L")"
SEEN=<lane>/logs/landing-last-seen-5090
H=$(cd "$(dirname "$0")" && pwd)
date +%s > "$SEEN"
quiet() { s=$(gpulock status 2>/dev/null) || return 1; now=$(date +%s)
  if echo "$s" | grep -qiE 'memra landing|release'; then echo "$now" > "$SEEN"; return 1; fi
  [ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ] || return 1
  [ "$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value)" = 150 ] || return 1
  [ "$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value)" = 140 ] || return 1
  [ $(( now - $(cat "$SEEN") )) -ge ${QUIET_S:-180} ]; }
export -f quiet; export SEEN
echo "$(date -Is) chain start $1 pid $$" >> "$L"
while IFS= read -r line <&3; do
  [ -z "$line" ] && continue; case $line in \#*) continue ;; esac
  head=${line%% -- *}; cmd=${line#* -- }; set -- $head; name=$1; ttl=$2; pf=""; [ "${3:-}" = perf ] && pf=--perf
  why="memra #1266 pooled verify OOM: 5090 $name (one step per lease)"
  # A waiter of ours never sits ahead of a landing's or a release's ticket: while ours is still
  # waiting, the queue is checked every 15 s, and when one shows ours is withdrawn (the waiting
  # gpulock run and its group stopped; gpulock prunes a dead waiter's ticket) and queued again
  # only after the quiet period.
  while :; do
    waitfor --until "bash -c quiet" --every 30s --timeout 300m --quiet >> "$L" 2>&1 || { echo "$(date -Is) $name: never quiet" >> "$L"; exit 1; }
    echo "$(date -Is) $name: queue for the lease ${pf:-(no --perf)}" >> "$L"
    if [ -n "$pf" ]; then
      # A timed step: a --perf lease taken with acquire, so timed_take.sh can renew it through its
      # rests and retries (the owner's take rule); released here when the step ends.
      idf=$(mktemp)
      setsid gpulock acquire --ttl 30m --for "$why" --wait 240m --perf > "$idf" 2>> "$L" &
    else
      setsid gpulock run --ttl "${ttl}m" --for "$why" --wait 240m -- bash -c "$cmd" >> "$L" 2>&1 &
    fi
    w=$!
    waitfor --until "bash $H/withdraw_probe.sh $w '$why' gpulock status" --every 15s --timeout 300m --quiet >> "$L" 2>&1
    if kill -0 "$w" 2>/dev/null && gpulock status 2>/dev/null | grep -i 'waiting:' | grep -qF "$why"; then
      kill -TERM -"$w" 2>/dev/null; wait "$w" 2>/dev/null
      echo "$(date -Is) $name: a landing or release ticket showed; our waiter withdrawn" >> "$L"
      date +%s > "$SEEN"
      continue
    fi
    wait "$w"; rc=$?
    if [ -n "$pf" ]; then
      id=$(cat "$idf"); rm -f "$idf"
      if [ -z "$id" ]; then
        echo "$(date -Is) $name: no lease (acquire rc=$rc)" >> "$L"; rc=1
      else
        echo "$(date -Is) $name: lease $id (--perf)" >> "$L"
        trap 'gpulock release "$id" >> "$L" 2>&1' EXIT
        bash "$H/timed_take.sh" "$id" "$name" "$cmd" >> "$L" 2>&1; rc=$?
        gpulock release "$id" >> "$L" 2>&1; trap - EXIT
      fi
    fi
    break
  done
  echo "$(date -Is) $name rc=$rc" >> "$L"
  [ $rc = 0 ] || [ "${CHAIN_KEEP_GOING:-0}" = 1 ] || exit 1
done 3< "$1"
echo "$(date -Is) rtx chain done: $1" >> "$L"
