#!/usr/bin/env bash
# Rig: one gate cell on a GB10 unit under one sparklock lease. The cell runs detached on the unit
# (~/.cache/memra-split-rung/cells/NAME/{cmd.sh,log,rc}); this script waits for its rc file in
# bounded steps, then releases the lease.
# usage: unit_cell.sh UNIT NAME TTL_MIN 'REMOTE COMMAND'
set -u
UNIT=$1; NAME=$2; TTL=$3; CMD=$4
R='~/.cache/memra-split-rung'
L=$HOME/.cache/memra-split-rung/runs/unit-cells.log
id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "memra deep split rung: gate cell $NAME (one cell per lease)" --wait 240m 2>>"$L")
[ -n "$id" ] || { echo "$(date -Is) $UNIT $NAME: no lease" >> "$L"; exit 1; }
echo "$(date -Is) $UNIT $NAME: lease $id" >> "$L"
# A landing waiter goes first at a cell boundary, even behind a ticket of ours queued earlier:
# hand the unit back, wait until it has taken it, queue again. YIELD_ANY=1 (a unit another lane
# has first call on) yields to any waiter that is not one of this lane's cells.
PAT=landing; [ "${YIELD_ANY:-0}" = 1 ] && PAT='^((?!memra deep split rung).)*$'
foreign() { sparklock status --unit "$UNIT" | grep -i 'waiting:' | grep -qiP "$PAT"; }
while foreign; do
  echo "$(date -Is) $UNIT $NAME: a waiter that goes first is queued; yielding" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  waitfor --until "! sparklock status --unit $UNIT | grep -i 'waiting:' | grep -qiP '$PAT'" --every 30s --timeout 90m --quiet >> "$L" 2>&1
  id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "memra deep split rung: gate cell $NAME (one cell per lease)" --wait 240m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $UNIT $NAME: no lease after yielding" >> "$L"; exit 1; }
  echo "$(date -Is) $UNIT $NAME: lease $id" >> "$L"
done
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
