#!/usr/bin/env bash
# WP-B day 25: run tools/health-fault-gate.sh on the local RTX 5090 Laptop GPU under the rig lock and the CPU quota.
# Usage: research/spill-b-20260919/run-day25-gate.sh <label> [HFG_ARMS]
set -uo pipefail
WT=/home/avifenesh/projects/wt-spill-b
cd "$WT" || exit 2
LABEL=${1:?label}
OUT=$WT/research/spill-b-20260919/rtx5090-day25/$LABEL
mkdir -p "$OUT"
LOCK=/tmp/memra-5090.lock
exec 9>"$LOCK"
echo "$(date -u +%FT%TZ) waiting for $LOCK (flock -w 1800)" | tee "$OUT/lock.txt"
flock -w 1800 9 || { echo "$(date -u +%FT%TZ) lock not acquired within 1800 s; cell not run" | tee -a "$OUT/lock.txt"; exit 3; }
echo "$(date -u +%FT%TZ) lock acquired" | tee -a "$OUT/lock.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps-before.csv"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$OUT/gpu-before.csv"
git rev-parse HEAD > "$OUT/source.txt"; git status --short >> "$OUT/source.txt"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G \
  env HFG_OUT="$OUT/gate" ${2:+HFG_ARMS="$2"} tools/health-fault-gate.sh 2>&1 | tee "$OUT/gate.log"
echo "${PIPESTATUS[0]}" > "$OUT/gate.exit"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps-after.csv"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$OUT/gpu-after.csv"
echo "$(date -u +%FT%TZ) done, gate exit $(cat "$OUT/gate.exit")" | tee -a "$OUT/lock.txt"
