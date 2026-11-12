#!/usr/bin/env bash
# memra#1255: a sequence of 5090 tasks, one lease each (lease.sh). A step refused because the GPU
# is held or a landing/release ticket is present waits (bounded) for the GPU to free, then retries.
# A step refused by the power guard (rc 76: battery, or PPT values not restored) waits for AC and
# the PPT values, then gives a landing's perf retake the GPU first: it waits up to 20 min for a
# "memra landing" lease to appear, and for the GPU to be free again, before it retries.
# usage: campaign.sh STEPFILE   (one task per line: LABEL<TAB>COMMAND...)
set -u
S=$HOME/.cache/memra-1255
LOG=$S/campaign.log
power_ok='test "$(cat /sys/class/power_supply/ADP0/online)" = 1 && test "$(cat /sys/class/firmware-attributes/*/attributes/nv_tgp/current_value)" = 150 && test "$(cat /sys/class/firmware-attributes/*/attributes/ppt_pl1_spl/current_value)" -ge 90'
while IFS=$'\t' read -r label cmd; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  if grep -qxF "DONE $label" "$LOG" 2>/dev/null; then continue; fi
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "$(date -Is) start $label (attempt $attempt)" >> "$LOG"
    bash "$S/lease.sh" "$label" bash -c "$cmd" >> "$S/campaign.out" 2>&1 < /dev/null
    rc=$?
    echo "$(date -Is) rc=$rc $label" >> "$LOG"
    if [ "$rc" = 76 ]; then
      waitfor --until "$power_ok" --timeout 4h --every 60s --quiet || { echo "$(date -Is) gave up waiting for power: $label" >> "$LOG"; exit 1; }
      echo "$(date -Is) power restored; giving a landing retake the GPU first" >> "$LOG"
      waitfor --until 'gpulock status 2>&1 | grep -q "memra landing"' --timeout 20m --every 20s --quiet && \
        echo "$(date -Is) a landing took the GPU" >> "$LOG"
      waitfor --until 'gpulock status 2>&1 | grep -q "^5090: free"' --timeout 3h --every 30s --quiet || { echo "$(date -Is) gave up waiting: $label" >> "$LOG"; exit 1; }
      continue
    fi
    if [ "$rc" != 75 ] && ! tail -n 3 "$S/campaign.out" | grep -q "gpulock: 5090 is held\|legacy lock .* is held\|compute processes with no lease\|free but queued"; then
      break
    fi
    # rc 75: a landing or release ticket was held or waiting. Wait until none is (a free GPU with
    # a landing queued for it is not free for this lane), then retry; lease.sh queues otherwise.
    if [ "$rc" = 75 ]; then
      waitfor --until '! gpulock status 2>&1 | grep -qiE "memra landing|release"' --timeout 3h --every 30s --quiet || { echo "$(date -Is) gave up waiting: $label" >> "$LOG"; exit 1; }
    else
      waitfor --until 'gpulock status 2>&1 | grep -q "^5090: free"' --timeout 3h --every 30s --quiet || { echo "$(date -Is) gave up waiting: $label" >> "$LOG"; exit 1; }
    fi
  done
  { [ "$rc" = 0 ] || [ "$rc" = 1 ] || [ "$rc" = 3 ]; } && echo "DONE $label" >> "$LOG"
done < "$1"
echo "$(date -Is) CAMPAIGN-END" >> "$LOG"
