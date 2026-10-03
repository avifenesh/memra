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
exec python3 "$(dirname "$0")/unittest_floor.py" "$@"
