#!/usr/bin/env bash
# Interleaved fresh-boot A/B runner (amended A/B law: x3 interleaved, x5 on anomaly).
# Usage: run-ab.sh <cell> <boot-list> <armA:bin:mode> <armB:bin:mode> [...]
#   <boot-list> is a space-free list like "1 2 3" (quote it) or "4 5" to resume an
#   escalation without renumbering the banked boots.
# Cycles: for boot in list: for each arm: boot -> price -> stop.
set -u
CELL=${1:?cell name}
BOOTS=${2:?boot list}
shift 2
D=/home/ubuntu/perf-chain
ROWS=$D/receipts/rows-$CELL.jsonl
PROG=$D/receipts/progress-$CELL.txt
# PREFLIGHT: validate EVERY arm before booting anything. A missing binary or an unknown
# mode used to surface as a BOOT_FAIL mid-rotation, which aborts the whole cell after the
# earlier arms have already burned card time. Resolve each arm's binary by UNIQUE sha
# prefix so a mistyped sha cannot silently become a missing file either.
RESOLVED=()
for spec in "$@"; do
  A=${spec%%:*}; rest=${spec#*:}; BSPEC=${rest%%:*}; M=${rest#*:}
  if [ -x "$BSPEC" ]; then
    BR=$BSPEC
  else
    # treat as a sha prefix under bin/
    set -- $(ls -1 "$D"/bin/memra-server-"${BSPEC##*memra-server-}"* 2>/dev/null)
    if [ "$#" -ne 1 ]; then
      echo "PREFLIGHT_FAIL arm=$A: binary spec '$BSPEC' resolved to $# candidates" | tee -a "$PROG"; exit 3
    fi
    BR=$1
  fi
  grep -q "^  $M)" "$D/harness/launch.sh" 2>/dev/null || grep -q "  $M)" "$D/harness/launch.sh" || {
    echo "PREFLIGHT_FAIL arm=$A: launch.sh has no mode '$M'" | tee -a "$PROG"; exit 3; }
  RESOLVED+=("$A:$BR:$M")
  echo "PREFLIGHT_OK arm=$A bin=$BR md5=$(md5sum "$BR" | cut -d' ' -f1) fp=$(grep -aom1 'memra-[0-9a-f]\{12\}' "$BR") mode=$M" >> "$PROG"
done
set -- "${RESOLVED[@]}"
echo "$(date -u +%FT%TZ) RUN_START cell=$CELL boots=[$BOOTS] arms=$*" >> "$PROG"
for boot in $BOOTS; do
  for spec in "$@"; do
    ARM=${spec%%:*}; rest=${spec#*:}; BINP=${rest%%:*}; MODE=${rest#*:}
    TAG="${ARM}${boot}"
    echo "$(date -u +%H:%M:%SZ) BOOT_START $TAG bin=$BINP mode=$MODE" >> "$PROG"
    "$D/harness/boot.sh" "$TAG" "$BINP" "$MODE" >> "$PROG" 2>&1 || {
      echo "BOOT_FAIL $TAG" >> "$PROG"; "$D/harness/stop.sh" >> "$PROG" 2>&1; exit 2; }
    python3 "$D/harness/digits.py" "$TAG" "$boot" "$ROWS" >> "$PROG" 2>&1 || echo "BENCH_FAIL $TAG" >> "$PROG"
    "$D/harness/stop.sh" >> "$PROG" 2>&1
    echo "$(date -u +%H:%M:%SZ) CYCLE_DONE $TAG" >> "$PROG"
  done
done
echo "$(date -u +%FT%TZ) RUN_DONE cell=$CELL" >> "$PROG"
