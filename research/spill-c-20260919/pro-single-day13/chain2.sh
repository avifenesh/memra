#!/usr/bin/env bash
# Wait for pass 1, archive it as attempt1-cache128, rebuild at the shipped tip, run pass 2.
set -uo pipefail
R=/root/spill-receipts/c-day13
while tmux has-session -t =c13run 2>/dev/null; do sleep 15; done
mkdir -p $R/attempt1-cache128
for d in smoke-off smoke-on hostgate-identity-off-default hostgate-identity-on-default hostgate-identity-off-plain hostgate-identity-on-plain hostgate-failure-off-default hostgate-failure-on-default hostgate-failure-off-plain hostgate-failure-on-plain reclaim-off reclaim-on; do
  mv $R/$d $R/attempt1-cache128/ 2>/dev/null; mv $R/$d-driver.log $R/$d.exit $R/attempt1-cache128/ 2>/dev/null
done
mv $R/smoke-off-server.log $R/smoke-on-server.log $R/driver-stdout.log $R/lock-retries.log $R/attempt1-cache128/ 2>/dev/null
cp $R/driver.log $R/attempt1-cache128/driver-pass1.log
mkdir -p $R/build-pass1 && cp $R/build/* $R/build-pass1/
cp $R/bins/memra-server $R/bins/memra-server.pass1
cd /root/wt-c
git fetch -q $R/c13.bundle lane/spill-c-20260919:refs/bundle/c13 && git checkout -q --detach refs/bundle/c13
bash $R/build.sh
bash $R/driver2.sh
