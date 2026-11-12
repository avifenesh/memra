#!/usr/bin/env bash
# memra#1255: run one task on the 5090 under one gpulock lease (30 min at most), after 3 minutes
# with no "memra landing" or "release" ticket held or waiting. After that watch it queues a
# first-come ticket for up to 45 min (gpulock --wait); the 29 min bound applies from the take.
# usage: lease.sh "why" CMD...
set -u
WHY=$1; shift
LOG=$HOME/.cache/memra-1255/lease.log
echo "$(date -Is) watch 3m before: $WHY" >> "$LOG"
if waitfor --until 'gpulock status 2>&1 | grep -qiE "memra landing|release"' --timeout 3m --every 10s --quiet; then
  echo "$(date -Is) a landing or release ticket is present; not queueing" | tee -a "$LOG"
  gpulock status 2>&1 | tee -a "$LOG"
  exit 75
fi
# Power guard (coordinator, 2026-11-12): no 5090 cell unless the laptop is on AC with its PPT
# values restored (nv_tgp=150, ppt_pl1_spl>=90). Read only, never changed.
AC=$(cat /sys/class/power_supply/ADP0/online 2>/dev/null)
TGP=$(cat /sys/class/firmware-attributes/*/attributes/nv_tgp/current_value 2>/dev/null)
PL1=$(cat /sys/class/firmware-attributes/*/attributes/ppt_pl1_spl/current_value 2>/dev/null)
if [ "$AC" != 1 ] || [ "${TGP:-0}" != 150 ] || [ "${PL1:-0}" -lt 90 ]; then
  echo "$(date -Is) power guard: AC=$AC nv_tgp=$TGP ppt_pl1_spl=$PL1; not taking the GPU" | tee -a "$LOG"
  exit 76
fi
echo "$(date -Is) power AC=$AC nv_tgp=$TGP ppt_pl1_spl=$PL1" >> "$LOG"
echo "$(date -Is) take: $WHY" >> "$LOG"
nvidia-smi --query-gpu=enforced.power.limit,power.limit,memory.used --format=csv,noheader >> "$LOG" 2>&1
gpulock run --ttl 30m --wait 45m --for "memra #1255 $WHY" -- timeout 29m "$@"
rc=$?
echo "$(date -Is) done rc=$rc: $WHY" >> "$LOG"
exit $rc
