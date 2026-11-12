#!/usr/bin/env bash
# Laptop (memra#1233): take one lease and run a command under it, behind landing and release
# tickets at every moment before the lease is granted.
#   - no ticket is queued while a landing/release ticket is listed (held or waiting);
#   - a ticket already waiting is withdrawn when a landing/release ticket appears, and re-queued
#     once none is listed (the waiter queued before the landing must not stay ahead of it);
#   - a lease that was granted runs to its end: nothing is preempted, the work under it is one
#     cell of at most 30 minutes.
# usage: yield-lease.sh LOG gpulock|sparklock:UNIT TTL WAIT "why" -- CMD...
# exit: CMD's status under a granted lease; 75 when no lease was granted in 12 tries.
set -u
LOG=$1; KIND=$2; TTL=$3; WAIT=$4; WHY=$5; shift 6
case $KIND in
  gpulock) STATUS=(gpulock status); RUN=(gpulock run --ttl "$TTL" --wait "$WAIT" --for "$WHY" --) ;;
  sparklock:*) U=${KIND#sparklock:}; STATUS=(sparklock status --unit "$U")
               RUN=(sparklock run --unit "$U" --ttl "$TTL" --wait "$WAIT" --for "$WHY" --) ;;
  *) echo "unknown lock $KIND" >&2; exit 2 ;;
esac
busy() { "${STATUS[@]}" 2>/dev/null | grep -qiE 'landing|release'; }
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  while busy; do command sleep 20; done
  : > "$LOG"
  setsid "${RUN[@]}" "$@" > "$LOG" 2>&1 &
  pid=$!
  withdrawn=0
  while kill -0 "$pid" 2>/dev/null; do
    if ! grep -q "leased as" "$LOG" && busy; then
      # Still waiting and a landing/release ticket is listed: withdraw this ticket.
      kill -- -"$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      grep -q "leased as" "$LOG" || withdrawn=1
      # A killed waiter can leave its ticket listed: release it by id.
      for id in $("${STATUS[@]}" 2>/dev/null | grep -F "waiting:" | grep -F "$WHY" | grep -o 'id [^ ]*' | cut -d' ' -f2); do
        case $KIND in
          gpulock) gpulock release "$id" > /dev/null 2>&1 ;;
          *) sparklock release "$id" --unit "$U" > /dev/null 2>&1 ;;
        esac
      done
      break
    fi
    command sleep 5
  done
  if [ "$withdrawn" = 1 ]; then
    echo "$(date -Is) withdrew the waiting ticket for a landing/release ticket ($WHY)" >> "$LOG.yield"
    continue
  fi
  wait "$pid"; rc=$?
  grep -q "leased as" "$LOG" && exit $rc
done
exit 75
