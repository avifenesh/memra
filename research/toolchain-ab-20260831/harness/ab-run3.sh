#!/usr/bin/env bash
# ERA CELL (alternative axis after the toolchain null): OLD = memra c9a617ca994b built
# on-box + the exact 140-era serving env (incl. the later-removed BANK_V2/SEL_DOWN8
# doors, no vision, 08-29 models.toml) vs PIN = memra 3999a92a6e18 on-box build + the
# current deploy-shape env. Same box, same protocol, interleaved fresh-boot x3.
set -u
D=/home/ubuntu/toolchain-ab
ROWS=$D/receipts/rows-era.jsonl
PROG=$D/receipts/progress.txt
for boot in 1 2 3; do
  for arm in O P; do
    TAG="${arm}${boot}"
    if [ "$arm" = O ]; then
      BINP=$D/bin/memra-server-old-c9a617ca; L=$D/launch-140era.sh
    else
      BINP=$D/bin/memra-server-native-cuda132; L=$D/launch.sh
    fi
    echo "$(date -u +%H:%M:%SZ) BOOT_START $TAG" >> "$PROG"
    "$D/boot.sh" "$TAG" "$BINP" "$L" >> "$PROG" 2>&1 || { echo "BOOT_FAIL $TAG" >> "$PROG"; "$D/stop.sh" >> "$PROG" 2>&1; exit 2; }
    python3 "$D/digits.py" "$TAG" "$boot" "$ROWS" >> "$PROG" 2>&1 || echo "BENCH_FAIL $TAG" >> "$PROG"
    "$D/stop.sh" >> "$PROG" 2>&1
    echo "$(date -u +%H:%M:%SZ) CYCLE_DONE $TAG" >> "$PROG"
  done
done
echo "ERA_DONE" >> "$PROG"
