#!/bin/bash
# Laptop side: sync wt-spec to $BOX:src-spec, run spec_build.sh WHAT there, wait (bounded), print the errors and the tail.
# usage: sync_build.sh [check|build|clippy|fmt|...]
set -u
what=${1:-build}
cd $HERE || exit 1
rsync -a --delete --exclude target --exclude .git --include '/research/modelplan-onboarding-dsv41-20261109/***' \
  --exclude '/research/*' wt-spec/ $BOX:$LANE/src-spec/ || exit 1
rsync -a spec/scripts/ $BOX:$LANE/spec/scripts/ || exit 1
ssh -o ConnectTimeout=20 $BOX "(cd $LANE && nohup bash spec/scripts/spec_build.sh $what) > $LANE/spec/build.log 2>&1 < /dev/null & echo started"
waitfor --ssh $BOX 'grep -q BUILD-RC $LANE/spec/build.log' --timeout 9m --every 10s > /dev/null 2>&1
ssh -o ConnectTimeout=20 $BOX 'grep -E "^(error|warning: unused|warning: this|Diff in)" -A18 $LANE/spec/build.log | cut -c1-220 | head -160; tail -n 2 $LANE/spec/build.log'
