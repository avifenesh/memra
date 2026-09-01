#!/bin/bash
# Chained identity battery: waits for the vg-mtp squeeze battery to finish, then runs
# interleaved base/lane pairs x3 on BOTH serve shapes.
set -u
cd /home/ubuntu/guard-lane
for i in $(seq 1 240); do
  grep -q "battery: ALL DONE" battery.txt && break
  sleep 30
done
for p in 1 2 3; do
  for shape in mtp dspark; do
    bash cell-identity-ab.sh $p $shape > run-ab-$shape-$p.out 2>&1
    echo "$(date -u +%H:%M:%S) identity: pair $p shape=$shape rc=$?" >> battery.txt
    sleep 5
  done
done
echo "$(date -u +%H:%M:%S) identity: ALL DONE" >> battery.txt
