#!/usr/bin/env bash
# memra#385 day-15 cell (research/spill-a-20260919/DAY15.md, pre-registered): today's one cuMemHostAlloc
# against one candidate (chunked: 8 threads, bytes/8 each; thp: mmap + madvise(MADV_HUGEPAGE) +
# cuMemHostRegister), sized by the engine's admission rule (MemAvailable minus the 32 GiB margin at start),
# correctness pass per arm first (byte-exact H2D/D2H roundtrip, driver flags read back), then AB x 5 and
# BA x 5, every allocation freed before the next, one process, one window, under the collector's lock
# (the harness takes the device's primary context). Pre-registered fallback: a driver refusal of the
# admissible size reruns once at 75 percent of MemFree (the day-12 size) under the cell name "-fallback".
# usage: arena-cell.sh <cell-name> <chunked|thp>
set -uo pipefail
cell=$1; cand=$2
R=/root/spill-receipts/a-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
echo "tree $(git rev-parse HEAD)"
echo "== regime before"; date -u +%FT%TZ; free -b
grep -E "MemTotal|MemFree|MemAvailable|Hugepagesize|HugePages_Total|HugePages_Free|AnonHugePages" /proc/meminfo
echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled) defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
echo "cgroup: $(cat /proc/self/cgroup) memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
nproc; numactl -H | head -3; cat /proc/loadavg; uname -r
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name --format=csv
mkdir -p "$R/$cell"
python3 tools/pinned-host-reserve-bench.py --basis admissible --arms "single,$cand" --chunks 8 --pairs-per-order 5 \
  --roundtrip --cell "$cell" --floor 1.10 --hugepage-min-fraction 0.9 --device 0 --out "$R/$cell/receipt.json"
rc=$?
if [ $rc = 2 ]; then
  echo "== admissible size refused by the driver (rc=2); pre-registered fallback: 75 percent of MemFree"
  mkdir -p "$R/$cell-fallback"
  python3 tools/pinned-host-reserve-bench.py --fraction 0.75 --basis free --arms "single,$cand" --chunks 8 --pairs-per-order 5 \
    --roundtrip --cell "$cell-fallback" --floor 1.10 --hugepage-min-fraction 0.9 --device 0 --out "$R/$cell-fallback/receipt.json"
  rc=$?
fi
echo "== regime after"; date -u +%FT%TZ; free -b; grep -E "MemFree|MemAvailable|AnonHugePages" /proc/meminfo; cat /proc/loadavg
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv
exit $rc
