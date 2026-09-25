#!/usr/bin/env bash
# DAY39 section 3, item 3's first cell on this host, under the collector's lock hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash fill-survey.sh @COLLECTOR_LOCK_FD@`. The fill probe
# (built by build.sh) at the 27B's and the 9B's staging shapes, T = 1, 2, 4, 8, 12, bitwise, and the span copies' time on
# this card (--gpu). A reading that picks item 3's design for this host class by DAY39 section 1's rule.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day38
cd /root/wt-a || exit 1
mkdir -p "$R/fill"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/fill/LOCK.json"
cp "$R/host-shape.txt" "$R/fill/host-shape.txt" 2>/dev/null
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/fill/compute-apps.before.csv" 2>&1
timeout 900 "$R/fill-target/release/day39-fill-survey" --gpu > "$R/fill/survey.log" 2>&1
rc=$?; echo "$rc" > "$R/fill/survey.exit"; cat "$R/fill/survey.log"
exit $rc
