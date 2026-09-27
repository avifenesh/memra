#!/usr/bin/env bash
# Wait for the first build session, then check out the shipped lane tip and build again (incremental).
set -uo pipefail
R=/root/spill-receipts/c-day13
while tmux has-session -t =c-day13 2>/dev/null; do sleep 15; done
mkdir -p $R/build-first && cp $R/build/* $R/build-first/ 2>/dev/null
cd /root/wt-c
git fetch -q $R/c13.bundle lane/spill-c-20260919:refs/bundle/c13 && git checkout -q --detach refs/bundle/c13
git rev-parse HEAD > $R/build/requested-source.txt
bash $R/build.sh
cat $R/build/exit
