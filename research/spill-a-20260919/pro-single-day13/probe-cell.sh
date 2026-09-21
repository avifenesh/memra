#!/usr/bin/env bash
# Driver-independent read-rate probe (pinned-read-probe.py: ctypes on libcuda, hashlib.sha256 and a
# bulk copy over write-combined and cached cuMemHostAlloc memory) under the inherited collector lock.
# usage: probe-cell.sh <lockfd> <bytes> <pairs>
set -uo pipefail
fd=$1; bytes=$2; pairs=$3
R=/root/spill-receipts/a-day13
EV=${CELL_OUT:?}/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$R/pinned-read-probe.py" | tee "$EV/probe.sha256"
python3 --version | tee "$EV/python.txt"
cat /proc/loadavg | tee "$EV/loadavg-before.txt"
env CUDA_VISIBLE_DEVICES=0 python3 "$R/pinned-read-probe.py" --bytes "$bytes" --pairs "$pairs"
rc=$?
cat /proc/loadavg | tee "$EV/loadavg-after.txt"
exit $rc
