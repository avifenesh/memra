#!/usr/bin/env bash
# tpd-battery TIMED BLOCK orchestrator: raises /root/TIMING-IN-FLIGHT for the WHOLE
# interleaved sequence and drops it at the end (TIMING-IN-FLIGHT protocol — other lanes on
# this box poll the marker and hold their own work while it exists), then runs
#   cell 2 pool rounds (v1 / diet / dietmap, interleaved)      -> the diet pricing table
#   cell 3 deep rounds (v1 / diet / dietgp, interleaved)       -> the grouped-prime TTFT row
#   cell 2 served PP-3 calibration boot                        -> the PP-3 comparator
# The marker is dropped even on failure (trap), and the block refuses to start if a marker
# is already up (another lane's timed work).
# Usage: timed_block.sh <round-idx...>      (default 1 2 3; call again with 4 5 to escalate)
set -uo pipefail
OUT=/root/out-tpd
MARK=/root/TIMING-IN-FLIGHT
[ $# -gt 0 ] && ROUNDS=("$@") || ROUNDS=(1 2 3)

if [ -e "$MARK" ]; then
  echo "REFUSE: $MARK already exists (another lane's timed work):"; cat "$MARK"; exit 2
fi
cleanup() { rm -f "$MARK"; echo "[tpd] marker DOWN $(date -u +%FT%TZ)"; }
trap cleanup EXIT
{
  echo "tpd-battery TP-2 DIET RE-PRICE timed block"
  echo "rounds=${ROUNDS[*]} cards=0,1 (+0,1,2 for the served calibration boot)"
  echo "raised $(date -u +%FT%TZ) by the tpd-battery window agent"
} > "$MARK"
echo "[tpd] marker UP $(date -u +%FT%TZ)"; cat "$MARK"

rc=0
bash "$OUT/c2_price.sh" pool "${ROUNDS[@]}" || rc=1
bash "$OUT/c2_price.sh" report || rc=1
bash "$OUT/c2_price.sh" deep "${ROUNDS[@]}" || rc=1
bash "$OUT/c2_price.sh" report-deep || rc=1
bash "$OUT/c2_served.sh" || rc=1
echo "TIMED_BLOCK_DONE rc=$rc rounds=${ROUNDS[*]}"
