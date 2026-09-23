#!/usr/bin/env bash
# WP-B day 31 (DAY31.md section 1): one ORDER of the MEMRA_ADMIT_BY_MEMORY open-output cell, four boots on ONE
# binary, run under the collector's lock hold:
#   tools/tier-battery.py --rig <rig> --external-lock --execute bash day31-order.sh @COLLECTOR_LOCK_FD@ <O1|O2>
# O1 = off, on2048, on8192, on32768 with the inner day-26 order AB; O2 = the reverse arm order with inner order BA.
# Every boot is run-day26-cell.sh with CLIENT=day31-client.py and PARSER=day31-parse.py; OFF boots have the two door
# variables removed from the environment, ON boots set them. MEMRA_TIMEOUT_MS_MAX is the same on every arm.
# env: R (receipt root), RIG_LOCK (the canonical lock the collector holds), BIN, WT, MODEL, MODEL_KEY, BURST,
#      MEMRA_CTX (local only), NO_SCOPE (target only). Nothing here signals a process this script did not start.
set -uo pipefail
fd=${1:?fd}; order=${2:?O1|O2}
: "${R:?receipt root}" "${RIG_LOCK:?lock}" "${BIN:?binary}" "${WT:?worktree}" "${BURST:?burst}"
cd "$WT" || exit 1
mkdir -p "$R/boots"
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$RIG_LOCK" --owner collector > "$R/LOCK-$order.json" 2>&1
echo "$(date -u +%FT%TZ) order=$order lock-proof rc=$?" >> "$R/order.log"
sha256sum "$BIN" > "$R/binary-$order.sha256"
if [ "$order" = O1 ]; then arms="off on2048 on8192 on32768"; inner=AB; else arms="on32768 on8192 on2048 off"; inner=BA; fi
export CLIENT=day31-client.py PARSER=day31-parse.py CLIENT_ARGS="--burst $BURST" LOCK=none N=5 \
  RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000
for arm in $arms; do
  cell="$order-$arm"
  echo "$(date -u +%FT%TZ) boot $cell start" >> "$R/order.log"
  if [ "$arm" = off ]; then
    env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS \
      bash research/spill-b-20260919/run-day26-cell.sh "$cell" "$inner" "$BIN" > "$R/boots/$cell.launch.log" 2>&1
  else
    env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS="${arm#on}" \
      bash research/spill-b-20260919/run-day26-cell.sh "$cell" "$inner" "$BIN" > "$R/boots/$cell.launch.log" 2>&1
  fi
  echo "$(date -u +%FT%TZ) boot $cell rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/$cell/REPORT.txt" 2>/dev/null)" >> "$R/order.log"
  sleep 5
done
echo "$(date -u +%FT%TZ) order=$order done" >> "$R/order.log"
