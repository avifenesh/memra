#!/bin/bash
set -u
cd /home/ubuntu/guard-lane
for i in $(seq 1 120); do
  grep -q "gsession: ALL DONE" battery.txt && break
  sleep 20
done
for t in g3 g4 g5 g6 g7; do
  bash cell-squeeze-gsession.sh $t > run-gsess-$t.out 2>&1
  echo "$(date -u +%H:%M:%S) gsession2: run $t rc=$?" >> battery.txt
  sleep 10
done
echo "$(date -u +%H:%M:%S) gsession2: ALL DONE" >> battery.txt
