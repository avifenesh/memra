#!/usr/bin/env bash
# Rig: one gate cell's receipt from a GB10 unit (cells/NAME/{cmd.sh,rc,log} and, for a margin
# cell, the probe's log under gate/): the command, the exit code, the cell's own output with the
# prompt token list and capture-time kernel lookups left out, home paths written as ~.
# usage: fetch_cell.sh UNIT NAME OUT_FILE
set -u
UNIT=$1; NAME=$2; OUT=$3
C=.cache/memra-split-rung/cells/$NAME
{
  echo "# cmd"
  /usr/bin/ssh -n "spark-$UNIT" "tail -n +3 $C/cmd.sh"
  echo "# rc $(/usr/bin/ssh -n "spark-$UNIT" "cat $C/rc")"
  if /usr/bin/ssh -n "spark-$UNIT" "ls $C/gate/*.log" > /dev/null 2>&1; then
    echo "# gate"
    /usr/bin/ssh -n "spark-$UNIT" "cat $C/log"
    echo "# probe (pool growth lines left out)"
    /usr/bin/ssh -n "spark-$UNIT" "cat $C/gate/*.log" | grep -v -i 'fa-pool'
  else
    echo "# log (prompt token list and capture-time lookups left out)"
    /usr/bin/ssh -n "spark-$UNIT" "cat $C/log" | grep -v -E '^prompt tokens: \[|CAPTURE-TIME kernel lookup'
  fi
} | sed "s#$HOME#~#g; s#/home/[a-z0-9_-]*/#~/#g" > "$OUT"
