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
D=/home/ubuntu/bankv3/lane
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
    # PASSES: one or more request shapes per boot, ALWAYS in the same order in every arm, so the
    # ordering (and any cache/thermal state it leaves) is a constant of the cell rather than a
    # confound between arms. Default is the vendor-default sampled shape alone, which is what
    # milestone 4 ran; the flip battery sets PASSES="greedy vendor" to get the byte-deterministic
    # table and the serving table out of the SAME boots -- paired arms, half the card time, and no
    # cross-boot clock drift between the two tables.
    for pass_shape in ${PASSES:-vendor}; do
      BV3_SAMPLING=$pass_shape python3 "$D/harness/price.py" "$TAG" "$boot" \
        "$D/receipts/rows-$CELL-$pass_shape.jsonl" >> "$PROG" 2>&1 \
        || echo "BENCH_FAIL $TAG shape=$pass_shape" >> "$PROG"
    done
    # ENGAGEMENT AFTER the requests, because the [nvfp4-sweep] announce needs a decode to have
    # happened -- and BEFORE the next arm, because an engagement failure invalidates the whole
    # cell and there is no reason to spend more card time on it. This is a hard abort, not a
    # warning: LAW:engagement-receipt-before-any-perf-row, and this lane's own stopped rotation.
    "$D/harness/assert-engagement.sh" "$TAG" "$MODE" >> "$PROG" 2>&1 || {
      echo "ENGAGEMENT_REFUSED $TAG mode=$MODE — CELL VOID, rows above are not aggregatable" >> "$PROG"
      "$D/harness/stop.sh" >> "$PROG" 2>&1; exit 7; }
    "$D/harness/stop.sh" >> "$PROG" 2>&1
    echo "$(date -u +%H:%M:%SZ) CYCLE_DONE $TAG" >> "$PROG"
  done
done
echo "$(date -u +%FT%TZ) RUN_DONE cell=$CELL" >> "$PROG"
