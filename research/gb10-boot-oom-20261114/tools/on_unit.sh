#!/usr/bin/env bash
# Runs on the workstation, as the command of `sparklock run --unit U` (memra#1316): copy this
# lane's tools to the unit, start one job there detached in its own process group, and wait for
# its DONE file. A job that outlives TIMEOUT is stopped (its whole group), so nothing runs on the
# unit after the lease ends.
# usage: on_unit.sh HOST JOB TIMEOUT 'COMMAND run from ~/.cache/memra-1316 on the unit'
#   the command must end by creating ~/.cache/memra-1316/jobs/JOB.done
set -u
HOST=$1; JOB=$2; TMO=$3; CMD=$4
H=$(cd "$(dirname "$0")" && pwd)
R=.cache/memra-1316
/usr/bin/ssh -n "$HOST" "mkdir -p $R/jobs $R/runs $R/research/gb10-boot-oom-20261114 $R/research/admit-headroom-unified-20261113/tools $R/research/spec-capture-headroom-20261112/tools" || exit 1
rsync -a --delete "$H/" "$HOST:$R/research/gb10-boot-oom-20261114/tools/" || exit 1
rsync -a "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" "$HOST:$R/research/admit-headroom-unified-20261113/tools/" || exit 1
rsync -a "$H/../../spec-capture-headroom-20261112/tools/tsrun.py" "$HOST:$R/research/spec-capture-headroom-20261112/tools/" || exit 1
/usr/bin/ssh -n "$HOST" "cd $R; rm -f jobs/$JOB.done; (setsid nohup bash -c 'echo \$\$ > jobs/$JOB.pid; $CMD') > jobs/$JOB.log 2>&1 < /dev/null & echo started"
waitfor --ssh "$HOST" "test -f $R/jobs/$JOB.done" --every 20s --timeout "$TMO" --quiet
rc=$?
if [ $rc != 0 ]; then
  echo "on_unit: $JOB not done (waitfor rc=$rc); stopping its process group"
  /usr/bin/ssh -n "$HOST" "p=\$(cat $R/jobs/$JOB.pid); g=\$(ps -o pgid= -p \$p | tr -d ' '); [ -n \"\$g\" ] && kill -TERM -- -\$g; echo stopped group \$g"
fi
/usr/bin/ssh -n "$HOST" "tail -n 25 $R/jobs/$JOB.log"
exit $rc
