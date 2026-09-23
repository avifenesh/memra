#!/usr/bin/env bash
# WP-B day 31 D4 (DAY31-D4.md section 1): one boot, the door ON at open_output_tokens=32768 with the host tier armed
# (MEMRA_KV_HOST_MB=8192), run under the collector's lock hold:
#   tools/tier-battery.py --rig rtx5090 --external-lock --execute bash d4-boot.sh @COLLECTOR_LOCK_FD@
# env: R (receipt root), RIG_LOCK, BIN, WT, BURST, MEMRA_CTX. Nothing here signals a process this script did not start.
set -uo pipefail
fd=${1:?fd}
: "${R:?receipt root}" "${RIG_LOCK:?lock}" "${BIN:?binary}" "${WT:?worktree}" "${BURST:?burst}"
cd "$WT" || exit 1
mkdir -p "$R/boots"
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$RIG_LOCK" --owner collector > "$R/LOCK-D4.json" 2>&1
echo "$(date -u +%FT%TZ) D4 lock-proof rc=$?" >> "$R/order.log"
export CLIENT=day31-client.py PARSER=day31-parse.py CLIENT_ARGS="--burst $BURST" LOCK=none N=5 \
  RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000
echo "$(date -u +%FT%TZ) boot D4-host start" >> "$R/order.log"
env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 MEMRA_KV_HOST_MB=8192 \
  bash research/spill-b-20260919/run-day26-cell.sh D4-host AB "$BIN" > "$R/boots/D4-host.launch.log" 2>&1
echo "$(date -u +%FT%TZ) boot D4-host rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/D4-host/REPORT.txt" 2>/dev/null)" >> "$R/order.log"
