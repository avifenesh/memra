#!/usr/bin/env bash
# Interleaved fresh-boot A/B: container(C, cuda13.1 fleet image) vs native(N, cuda13.2
# on-box), x3 boots per arm, order C1 N1 C2 N2 C3 N3. Each cycle: PID-verified boot,
# sealed digits protocol (smoke + warmup + 8 reps), stop + GPU drain.
set -u
D=/home/ubuntu/toolchain-ab
ROWS=$D/receipts/rows.jsonl
PROG=$D/receipts/progress.txt
declare -A BINS=( [C]=$D/bin/memra-server-container-cuda131 [N]=$D/bin/memra-server-native-cuda132 )
for boot in 1 2 3; do
  for arm in C N; do
    TAG="${arm}${boot}"
    echo "$(date -u +%H:%M:%SZ) BOOT_START $TAG" >> "$PROG"
    "$D/boot.sh" "$TAG" "${BINS[$arm]}" >> "$PROG" 2>&1 || { echo "BOOT_FAIL $TAG" >> "$PROG"; "$D/stop.sh" >> "$PROG" 2>&1; exit 2; }
    python3 "$D/digits.py" "$TAG" "$boot" "$ROWS" >> "$PROG" 2>&1 || echo "BENCH_FAIL $TAG" >> "$PROG"
    "$D/stop.sh" >> "$PROG" 2>&1
    echo "$(date -u +%H:%M:%SZ) CYCLE_DONE $TAG" >> "$PROG"
  done
done
echo "AB_DONE" >> "$PROG"
