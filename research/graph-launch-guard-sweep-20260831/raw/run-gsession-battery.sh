#!/bin/bash
# Waits for the identity battery, then runs 2 graph-session squeeze runs.
set -u
cd /home/ubuntu/guard-lane
for i in $(seq 1 360); do
  grep -q "identity: ALL DONE" battery.txt && break
  sleep 30
done
for t in g1 g2; do
  bash cell-squeeze-gsession.sh $t > run-gsess-$t.out 2>&1
  echo "$(date -u +%H:%M:%S) gsession battery: run $t rc=$?" >> battery.txt
  sleep 10
done
echo "$(date -u +%H:%M:%S) gsession: ALL DONE" >> battery.txt
