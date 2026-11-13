#!/usr/bin/env bash
# Rig, memra#1266: one timed step on the RTX 5090 under the owner's take rule (2026-11-13), inside
# a --perf lease the caller holds (ID). Before a take the card must read under 86 C. After the
# take, take_check.py judges every arm the step writes: a take with software thermal slowdown in
# more than 5% of an arm's samples under load, or an arm that started at 86 C or hotter, does not
# count. Its directories are kept as DIR.takeN, the lease is renewed and held through a 5-minute
# rest, the card must read under about 70 C, and the step runs again: up to 3 retries. After the
# fourth take that does not count, the best one (the lowest worst-arm share) is banked under the
# arms' names with its take.json, and the chain log gets a THERMAL-BANKED line for the
# coordinator. Each lease period is at most 30 minutes; the lease is renewed before every take
# and every rest.
# usage: timed_take.sh LEASE_ID NAME 'CMD'
set -u
ID=$1; NAME=$2; CMD=$3
H=$(cd "$(dirname "$0")" && pwd)
mapfile -t DIRS < <(python3 "$H/take_check.py" dirs "$CMD")
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' '; }
renew() { gpulock renew "$ID" --ttl 30m > /dev/null 2>&1 || echo "$(date -Is) $NAME: renew of $ID failed"; }
log() { echo "$(date -Is) $NAME: $*"; }
best=0; best_share=2
for take in 1 2 3 4; do
  renew
  if [ "$take" -gt 1 ]; then
    log "take $((take - 1)) does not count; rest 5 min under the lease ($(temp) C)"
    waitfor --until false --every 60s --timeout "${TAKE_REST:-5m}" --quiet > /dev/null 2>&1
    renew
    waitfor --until "[ \$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1) -lt 71 ]" --every 15s --timeout 10m --quiet > /dev/null 2>&1 \
      || log "the card reads $(temp) C after 15 min of rest"
    renew
  fi
  waitfor --until "[ \$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1) -lt 86 ]" --every 15s --timeout 10m --quiet > /dev/null 2>&1 \
    || log "the card still reads $(temp) C"
  renew
  log "take $take starts at $(temp) C"
  # The lease is renewed every 10 minutes while the take runs (a bench can run up to 30 minutes).
  bash -c "$CMD" &
  cp=$!
  while ! waitfor --gone "$cp" --timeout 10m --quiet > /dev/null 2>&1; do renew; done
  wait "$cp"; rc=$?
  [ "$rc" = 0 ] || { log "take $take rc=$rc"; exit "$rc"; }
  if python3 "$H/take_check.py" check "$take" "${DIRS[@]}"; then
    log "take $take counts"
    exit 0
  fi
  worst=$(python3 -c 'import json,sys; print(max(json.load(open(f + "/take.json"))["sw_thermal_load_share"] for f in sys.argv[1:]))' "${DIRS[@]}")
  if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' "$worst" "$best_share"; then best=$take; best_share=$worst; fi
  for d in "${DIRS[@]}"; do mv -T "$d" "$d.take$take"; done
done
for d in "${DIRS[@]}"; do mv -T "$d.take$best" "$d"; done
log "THERMAL-BANKED take $best of 4 (worst arm sw thermal share $best_share); the other takes are kept as DIR.takeN"
exit 0
