#!/usr/bin/env bash
# Rig: one cell on a GB10 unit under one sparklock lease. The cell runs detached on the unit
# (~/.cache/memra-specpub/cells/NAME/{cmd.sh,log,rc}); this script waits for its rc file in
# bounded steps, then releases the lease. A waiter whose reason names a landing or a release goes
# first at a cell boundary, even behind a ticket of ours queued earlier: the lease is handed back
# until that waiter has taken the unit, and no ticket of ours is queued until the unit has shown
# no such lease or ticket for 5 minutes (landing_quiet.sh). YIELD_ANY=1 (a unit another lane has
# first call on) yields to any waiter that is not one of this lane's cells. (The deep split rung
# lane's unit_cell.sh, for this lane's directory.)
# NOT_AFTER=HH:MM (today) is a unit's handover time: a cell starts only if its whole lease ends 5
# minutes before it, and no ticket of ours waits past that start. Exit 3: the window is closed.
# usage: [YIELD_ANY=1] [NOT_AFTER=HH:MM] unit_cell.sh UNIT NAME TTL_MIN 'REMOTE COMMAND'
set -u
UNIT=$1; NAME=$2; TTL=$3; CMD=$4
R='~/.cache/memra-specpub'
L=$HOME/.cache/memra-specpub/logs/unit-cells.log
WHY="memra spec boundary publish (GB10 decode lever 5): cell $NAME (one cell per lease)"
# A release runs its landing tickets in a row and the unit is the release's between them: no
# ticket of ours is queued until the unit has shown no landing lease or ticket for 5 minutes.
H=$(cd "$(dirname "$0")" && pwd)
SEEN=$HOME/.cache/memra-specpub/logs/landing-last-seen-$UNIT
# Seconds left to start this cell: unbounded without NOT_AFTER.
left() { [ -n "${NOT_AFTER:-}" ] || { echo 18000; return; }; echo $(( $(date -d "today $NOT_AFTER" +%s) - (TTL + 5) * 60 - $(date +%s) )); }
closed() { echo "$(date -Is) $UNIT $NAME: the unit's window closes at ${NOT_AFTER:-}; not started" >> "$L"; exit 3; }
[ "$(left)" -gt 0 ] || closed
waitfor --until "bash $H/landing_quiet.sh $UNIT $SEEN 300" --every 30s --timeout "$(left)s" --quiet >> "$L" 2>&1 \
  || { echo "$(date -Is) $UNIT $NAME: the unit did not go quiet in time" >> "$L"; [ "$(left)" -gt 0 ] && exit 1; closed; }
[ "$(left)" -gt 0 ] || closed
id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "$WHY" --wait "$(left)s" 2>>"$L")
[ -n "$id" ] || { echo "$(date -Is) $UNIT $NAME: no lease" >> "$L"; [ "$(left)" -gt 0 ] && exit 1; closed; }
echo "$(date -Is) $UNIT $NAME: lease $id" >> "$L"
PAT='landing|release|0\.141\.0'; [ "${YIELD_ANY:-0}" = 1 ] && PAT='^((?!memra spec boundary publish).)*$'
landing() { sparklock status --unit "$UNIT" | grep -i 'waiting:' | grep -qiP "$PAT"; }
while landing; do
  echo "$(date -Is) $UNIT $NAME: a waiter that goes first is queued; yielding" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  waitfor --until "! sparklock status --unit $UNIT | grep -i 'waiting:' | grep -qiP '$PAT'" --every 30s --timeout 120m --quiet >> "$L" 2>&1
  waitfor --until "bash $H/landing_quiet.sh $UNIT $SEEN 300" --every 30s --timeout 240m --quiet >> "$L" 2>&1
  [ "$(left)" -gt 0 ] || closed
  id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "$WHY" --wait "$(left)s" 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $UNIT $NAME: no lease after yielding" >> "$L"; [ "$(left)" -gt 0 ] && exit 1; closed; }
  echo "$(date -Is) $UNIT $NAME: lease $id" >> "$L"
done
[ "$(left)" -gt 0 ] || { sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1; closed; }
printf '%s\n' "set -u" "cd $R" "$CMD" | /usr/bin/ssh "spark-$UNIT" "mkdir -p $R/cells/$NAME && rm -f $R/cells/$NAME/rc && cat > $R/cells/$NAME/cmd.sh"
/usr/bin/ssh -n "spark-$UNIT" "(bash $R/cells/$NAME/cmd.sh > $R/cells/$NAME/log 2>&1; echo \$? > $R/cells/$NAME/rc) > /dev/null 2>&1 < /dev/null & echo started" >> "$L" 2>&1
deadline=$(( $(date +%s) + TTL * 60 - 60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  waitfor --ssh "spark-$UNIT" "test -f $R/cells/$NAME/rc" --timeout 9m --every 20s --quiet > /dev/null 2>&1 && break
done
rc=$(/usr/bin/ssh -n "spark-$UNIT" "cat $R/cells/$NAME/rc 2>/dev/null || echo timeout")
echo "$(date -Is) $UNIT $NAME: rc=$rc" >> "$L"
sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
echo "$(date -Is) $UNIT $NAME: released" >> "$L"
[ "$rc" = 0 ]
