#!/usr/bin/env bash
# The pinned-ab cell (DAY14.md pre-registration, the day-13 shape): tier-transfer-gate pinned-ab
# under the inherited collector lock; host regime beside the collector's GPU sampler. This laptop
# reports no power limit (nvidia-smi power.limit [N/A]): the card's own reading is kept in ev/.
# usage: pinned-cell.sh <lockfd> <bytes> <pairs>
set -uo pipefail
fd=$1; bytes=$2; pairs=$3
W=/home/avifenesh/projects/wt-spill-a
R=$W/research/spill-a-20260919/rtx5090-day14
EV=${CELL_OUT:?}/ev
cd "$W"
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
sha256sum "$R/bins/tier-transfer-gate" | tee "$EV/binary.sha256"
lscpu | grep -i "model name\|^CPU(s)\|L2 cache\|L3 cache\|NUMA node(s)" > "$EV/cpu.txt"
grep -i "MemTotal\|MemFree\|MemAvailable\|Hugepagesize" /proc/meminfo > "$EV/meminfo.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit,power.max_limit,power.default_limit,enforced.power.limit,driver_version --format=csv > "$EV/card.csv" 2>&1
cat /proc/loadavg | tee "$EV/loadavg-before.txt"
env CUDA_VISIBLE_DEVICES=0 "$R/bins/tier-transfer-gate" pinned-ab --bytes "$bytes" --pairs "$pairs"
rc=$?
cat /proc/loadavg | tee "$EV/loadavg-after.txt"
exit $rc
