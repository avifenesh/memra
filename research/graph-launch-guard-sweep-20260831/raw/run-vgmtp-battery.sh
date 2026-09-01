#!/bin/bash
# Chained N=5 vg-mtp squeeze battery (fresh server per run; Once-per-process note).
set -u
cd /home/ubuntu/guard-lane
for t in m2 m3 m4 m5 m6; do
  bash cell-squeeze-vg-mtp.sh $t > run-vgmtp-$t.out 2>&1
  echo "$(date -u +%H:%M:%S) battery: run $t rc=$?" >> battery.txt
  sleep 10
done
echo "$(date -u +%H:%M:%S) battery: ALL DONE" >> battery.txt
