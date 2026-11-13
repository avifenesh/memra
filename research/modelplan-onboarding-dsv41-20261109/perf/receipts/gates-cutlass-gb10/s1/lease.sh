#!/bin/bash
# Laptop: one task in one sparklock lease on UNIT (env, default cb33). Queue, copy SCRIPT to the unit, run
# `setsid bash SCRIPT ARGS` there in its own process group, renew every 90 s round while the done-marker is absent
# (at most MAXR rounds), kill the group at the cap, release. usage: lease.sh "REASON" DONE_FILE MARKER MAXR SCRIPT [ARGS]
set -u
S=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix/spark
U=${UNIT:-cb33}; H=spark-$U; R='~/.cache/dsv41-cfix'
REASON=$1; DONE=$2; MARK=$3; MAXR=$4; SCRIPT=$5; shift 5; ARGS="$*"
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
LOG=$S/lease-$U-$SCRIPT-$(date +%H%M%S).log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
$SSH $H 'mkdir -p ~/.cache/dsv41-cfix' && scp -q $S/$SCRIPT $H:.cache/dsv41-cfix/$SCRIPT || { say "copy failed"; exit 1; }
say "queueing on $U: $REASON"
out=$(sparklock acquire --unit $U --ttl 30m --for "dsv41 lane: cutlass decode fixes: $REASON" --wait 480m 2>&1)
ID=$(echo "$out" | grep -o 'claude-code-[0-9T]*-[0-9]*' | tail -1)
[ -n "$ID" ] || { say "no lease: $out"; exit 2; }
say "lease $ID"
$SSH $H "rm -f $R/$DONE; cd $R && (setsid bash $SCRIPT $ARGS > $R/$SCRIPT.out 2>&1 < /dev/null & echo \$! > $R/$SCRIPT.pgid); echo started pgid \$(cat $R/$SCRIPT.pgid)" | tee -a $LOG
i=0
while [ $i -lt $MAXR ]; do
  waitfor --timeout 90s --every 15s --ssh $H "grep -q '$MARK' $R/$DONE" > /dev/null 2>&1 && break
  sparklock renew $ID --unit $U --ttl 30m > /dev/null 2>&1
  i=$((i + 1))
done
if ! $SSH $H "grep -q '$MARK' $R/$DONE 2>/dev/null"; then
  say "TIME CAP: killing the group"
  $SSH $H "kill -- -\$(cat $R/$SCRIPT.pgid) 2>/dev/null; sleep 2; kill -9 -- -\$(cat $R/$SCRIPT.pgid) 2>/dev/null; true"
fi
$SSH $H "cat $R/$DONE" >> $LOG
sparklock release $ID --unit $U >> $LOG 2>&1
say "released; LEASE-DONE"
