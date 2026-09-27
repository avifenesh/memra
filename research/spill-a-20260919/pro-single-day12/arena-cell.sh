#!/usr/bin/env bash
# memra#385 harness cell: one cuMemHostAlloc versus 8 parallel chunks at 75 percent of MemFree
# (lane B shares this box's host RAM and page cache; MemFree never evicts it), 5 interleaved pairs
# per order (AB then BA), one process, one window. Harness
# receipt for this box only; the decision cell is the 2x B200 pair. Runs under the collector's
# lock because it takes the device's primary context.
# usage: arena-cell.sh <cell-name>
set -uo pipefail
cell=$1
R=/root/spill-receipts/a-day12
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
git rev-parse HEAD
free -b
grep -i "MemAvailable\|Hugepagesize\|HugePages_Total\|AnonHugePages" /proc/meminfo
cat /sys/kernel/mm/transparent_hugepage/enabled
nproc; cat /proc/loadavg
python3 tools/pinned-host-reserve-bench.py --fraction 0.75 --basis free --chunks 8 --pairs-per-order 5 --device 0 --out "$R/$cell/receipt.json"
rc=$?
free -b
exit $rc
