#!/usr/bin/env bash
# Resume x5 escalation after the N4 scp-race boot failure: cycles N4, C5, N5.
set -u
D=/home/ubuntu/toolchain-ab
ROWS=$D/receipts/rows.jsonl
PROG=$D/receipts/progress.txt
run_cycle() {
  arm=$1; boot=$2; TAG="${arm}${boot}"
  BINP=$D/bin/memra-server-native-cuda132
  [ "$arm" = C ] && BINP=$D/bin/memra-server-container-cuda131
  echo "$(date -u +%H:%M:%SZ) BOOT_START $TAG" >> "$PROG"
  "$D/boot.sh" "$TAG" "$BINP" >> "$PROG" 2>&1 || { echo "BOOT_FAIL $TAG" >> "$PROG"; "$D/stop.sh" >> "$PROG" 2>&1; exit 2; }
  python3 "$D/digits.py" "$TAG" "$boot" "$ROWS" >> "$PROG" 2>&1 || echo "BENCH_FAIL $TAG" >> "$PROG"
  "$D/stop.sh" >> "$PROG" 2>&1
  echo "$(date -u +%H:%M:%SZ) CYCLE_DONE $TAG" >> "$PROG"
}
run_cycle N 4
run_cycle C 5
run_cycle N 5
echo "AB_X5_DONE" >> "$PROG"
