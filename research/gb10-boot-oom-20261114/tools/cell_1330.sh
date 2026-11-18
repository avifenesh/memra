#!/usr/bin/env bash
# Runs on the workstation under the caller's sparklock lease (memra#1330): copy this lane's tools
# to the unit and start one recovery_cell.sh there, detached in its own session with its pid in
# ~/.cache/memra-1330/jobs/NAME.pid; returns at once. `cell_1330.sh HOST stop NAME` asks the cell
# to end after its round (STOP file); `cell_1330.sh HOST kill NAME` stops its whole process group.
# usage: cell_1330.sh HOST start NAME 'VAR=... VAR=...'   (the cell's settings, see recovery_cell.sh)
#        cell_1330.sh HOST stop|kill|status NAME
#        cell_1330.sh HOST until NAME UNIX
set -u
HOST=$1; ACT=$2; NAME=$3
H=$(cd "$(dirname "$0")" && pwd)
R=.cache/memra-1330
T=$R/research/gb10-boot-oom-20261114/tools
case $ACT in
  start)
    VARS=$4
    /usr/bin/ssh -n "$HOST" "mkdir -p $R/jobs $R/runs $R/tmp $T $R/research/admit-headroom-unified-20261113/tools" || exit 1
    rsync -a --delete "$H/" "$HOST:$T/" || exit 1
    rsync -a "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" "$HOST:$R/research/admit-headroom-unified-20261113/tools/" || exit 1
    /usr/bin/ssh -n "$HOST" "cd $R; (setsid nohup bash -c 'echo \$\$ > jobs/$NAME.pid; export TMPDIR=\$HOME/$R/tmp; NAME=$NAME $VARS bash \$HOME/$T/recovery_cell.sh') > jobs/$NAME.log 2>&1 < /dev/null & echo started" ;;
  until)  # move the cell's end to unix time $4 (keep it inside the lease)
    /usr/bin/ssh -n "$HOST" "echo $4 > $R/runs/$NAME/UNTIL; echo until \$(date -d @$4 +%H:%M)" ;;
  stop)
    /usr/bin/ssh -n "$HOST" "touch $R/runs/$NAME/STOP; echo stop requested" ;;
  kill)
    /usr/bin/ssh -n "$HOST" "p=\$(cat $R/jobs/$NAME.pid); g=\$(ps -o pgid= -p \$p | tr -d ' '); if [ -n \"\$g\" ]; then kill -TERM -- -\$g; echo stopped group \$g; else echo no process \$p; fi" ;;
  status)
    /usr/bin/ssh -n "$HOST" "p=\$(cat $R/jobs/$NAME.pid 2>/dev/null); ps -o pid=,etime= -p \$p 2>/dev/null || echo not running; tail -n 4 $R/runs/$NAME/run.log; echo creations=\$(grep -vc skipped $R/runs/$NAME/creations.jsonl 2>/dev/null) failed_first=\$(grep -c '\"failed_first\": true' $R/runs/$NAME/creations.jsonl 2>/dev/null) holders_failed_first=\$(grep -c '\"failed_first\": true' $R/runs/$NAME/holders.jsonl 2>/dev/null); test -e $R/runs/$NAME/DONE && echo DONE; true" ;;
esac
