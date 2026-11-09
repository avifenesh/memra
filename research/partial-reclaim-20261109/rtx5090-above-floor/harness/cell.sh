#!/usr/bin/env bash
# One cell on the local RTX 5090: the arms of one workload in the given order, a fresh boot per arm. Run it under
# one gpulock lease per cell:
#   gpulock run --ttl 2h --for "memra #1124: 5090 above-floor <cell>" -- bash cell.sh WORKLOAD "A B B A" OUT [VAR=VALUE ...]
# A and B are the binaries in $CELL_A and $CELL_B. LIMIT_ARGS passes through to the limit workload.
set -u
WL=$1; ORDER=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$OUT/cell.log"; }
log "cell $WL order [$ORDER] A=$(basename "${CELL_A:?}") B=$(basename "${CELL_B:?}") env: $* LIMIT_ARGS=${LIMIT_ARGS:-} lease: ${GPULOCK_LEASE_ID:-?}"
i=0
for a in $ORDER; do
  i=$((i + 1)); bin=$CELL_A; [ "$a" = B ] && bin=$CELL_B
  log "$WL $i $a start"
  bash "$HERE/arm.sh" "$WL-$i-$a" "$bin" "$OUT" "$WL" "$@"
  rc=$?
  log "$WL $i $a rc=$rc $(grep -a 'reclaim_on_defer_lines\|oom_lines' "$OUT/$WL-$i-$a/arm.txt" | tr '\n' ' ') $(grep -a -o 'budget [0-9]*MB' "$OUT/$WL-$i-$a/arm.txt" | head -1)"
done
log "cell $WL done"
