#!/bin/bash
# Laptop: one job in one sparklock lease on a GB10 unit, at most MAX minutes (the lease's TTL, never renewed), the job
# in its own process group (setsid; its group id in ~/.cache/dsv41-speed/JOB.pgid), stopped by group at the deadline,
# the lease released at the end.
# usage: r1dig_lease.sh UNIT "REASON" SCRIPT "ARGS" DONE_FILE MARKER [MAX_MIN]
#   UNIT b23d | cb33; SCRIPT in ~/.cache/dsv41-memra/r1-dig/spark (copied to ~/.cache/dsv41-speed on the unit);
#   DONE_FILE relative to ~/.cache/dsv41-speed; MARKER a fixed string.
set -u
U=$1; REASON=$2; SCRIPT=$3; ARGS=$4; DONE=$5; MARK=$6; MAX=${7:-30}
H=spark-$U
S=/home/avifenesh/.cache/dsv41-memra/r1-dig/spark
TAG=$(basename "$SCRIPT" .sh)-$(echo "$ARGS" | tr ' /' '__')
LOG=$S/lease-$U-$TAG.log
PG=.cache/dsv41-speed/$TAG.pgid
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
scp -q "$S/$SCRIPT" "$H:.cache/dsv41-speed/$SCRIPT" || { say "copy failed"; exit 1; }
out=$(sparklock acquire --unit "$U" --ttl "${MAX}m" --for "dsv41 lane: $REASON" --wait 8h 2>&1)
ID=$(echo "$out" | grep -o 'claude-code-[0-9T]*-[0-9]*' | tail -1)
[ -n "$ID" ] || { say "no lease: $out"; exit 2; }
say "lease $ID on $U: $REASON"
ssh "$H" "rm -f $PG; (cd ~/.cache/dsv41-speed && setsid bash -c 'echo \$\$ > ~/$PG; exec bash $SCRIPT $ARGS') > ~/.cache/dsv41-speed/$TAG.$U.out 2>&1 < /dev/null & echo started" | tee -a "$LOG"
waitfor --ssh "$H" "grep -qF '$MARK' ~/.cache/dsv41-speed/$DONE" --timeout "$((MAX - 1))m" --every 30s --quiet >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  say "job not done by the deadline (waitfor rc $rc): stopping its process group"
  ssh "$H" "test -s ~/$PG && kill -TERM -\$(cat ~/$PG)" >> "$LOG" 2>&1
fi
ssh "$H" "cat ~/.cache/dsv41-speed/$DONE" >> "$LOG" 2>&1
sparklock release "$ID" --unit "$U" >> "$LOG" 2>&1
say "released; LEASE-DONE rc $rc"
