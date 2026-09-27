#!/usr/bin/env bash
# The pinned-ab cell (DAY13.md pre-registration): tier-transfer-gate pinned-ab under the inherited
# collector lock; host regime beside the collector's GPU sampler.
# usage: pinned-cell.sh <lockfd> <bytes> <pairs>
set -uo pipefail
fd=$1; bytes=$2; pairs=$3
R=/root/spill-receipts/a-day13
EV=${CELL_OUT:?}/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
sha256sum "$R/bins/tier-transfer-gate" | tee "$EV/binary.sha256"
lscpu | grep -i "model name\|^CPU(s)\|L2 cache\|L3 cache\|NUMA node(s)" > "$EV/cpu.txt"
grep -i "MemTotal\|MemFree\|MemAvailable\|Hugepagesize" /proc/meminfo > "$EV/meminfo.txt"
cat /proc/loadavg | tee "$EV/loadavg-before.txt"
env CUDA_VISIBLE_DEVICES=0 "$R/bins/tier-transfer-gate" pinned-ab --bytes "$bytes" --pairs "$pairs"
rc=$?
cat /proc/loadavg | tee "$EV/loadavg-after.txt"
exit $rc
