#!/usr/bin/env bash
# Run a unittest discovery with a non-vacuity floor.
#
#   tools/unittest-floor.sh <start-dir> <pattern> <min-tests> [extra unittest args]
#
# `python3 -m unittest discover` exits 0 when discovery finds nothing ("Ran 0 tests ... OK"), so a
# start dir that is not a package, a renamed file that misses the pattern, or a suite moved into a
# subdirectory all read as a green check over zero tests (revuto on memra#592, 2026-09-21). Every
# other standing step in ci.yml carries a floor (`--min-passed`, `STATIC_FLOOR`, `report --expect`);
# this gives the two unittest steps the same teeth. The floor is a stated minimum, the count the
# receipts measured when the step was written; raise it when the suite grows, never lower it to pass.
set -euo pipefail
if [ $# -lt 3 ]; then
    echo "usage: $0 <start-dir> <pattern> <min-tests> [unittest args]" >&2
    exit 2
fi
START=$1; PATTERN=$2; MIN=$3; shift 3
case "$MIN" in ''|*[!0-9]*) echo "unittest-floor: <min-tests> must be a non-negative integer, got '$MIN'" >&2; exit 2;; esac
LOG=$(mktemp -t unittest-floor.XXXXXX)
trap 'rm -f "$LOG"' EXIT
set +e
python3 -m unittest discover -s "$START" -p "$PATTERN" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
RAN=$(grep -oE '^Ran [0-9]+ tests?' "$LOG" | tail -1 | grep -oE '[0-9]+' || true)
if [ -z "$RAN" ]; then
    echo "unittest-floor: FAIL: no 'Ran N tests' line for $START ($PATTERN)" >&2
    exit 1
fi
# The floor first: Python 3.12+ exits 5 on "NO TESTS RAN", older versions exit 0; both are the
# vacuity this script exists to catch and both report as the floor failure.
if [ "$RAN" -lt "$MIN" ]; then
    echo "unittest-floor: FAIL: ran $RAN tests, floor is $MIN for $START ($PATTERN); discovery lost tests" >&2
    exit 1
fi
if [ "$RC" -ne 0 ]; then
    echo "unittest-floor: FAIL: unittest exited $RC for $START ($PATTERN); ran $RAN" >&2
    exit "$RC"
fi
echo "unittest-floor: OK: ran $RAN tests (floor $MIN) for $START ($PATTERN)"
