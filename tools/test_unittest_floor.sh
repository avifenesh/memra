#!/usr/bin/env bash
# Teeth for tools/unittest-floor.sh: an empty discovery must red, a failing test must red, a
# passing suite at or above its floor must green, and a floor above the count must red.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d -t unittest-floor-teeth.XXXXXX)
trap 'rm -rf "$T"' EXIT
FAILS=0
check() { local name=$1 expect=$2 got=$3; if [ "$got" = "$expect" ]; then echo "ok   $name"; else echo "FAIL $name (expected exit $expect, got $got)"; FAILS=$((FAILS+1)); fi; }
mkdir -p "$T/empty" "$T/suite"
cat > "$T/suite/test_a.py" <<'PY'
import unittest
class A(unittest.TestCase):
    def test_one(self): pass
    def test_two(self): pass
PY
cat > "$T/suite/broken_b.py" <<'PY'
import unittest
class B(unittest.TestCase):
    def test_fails(self): self.fail("planted")
PY
rc=0; tools/unittest-floor.sh "$T/empty" 'test_*.py' 1 >/dev/null 2>&1 || rc=$?; check "empty discovery reds (Ran 0 tests)" 1 "$rc"
rc=0; tools/unittest-floor.sh "$T/suite" 'test_*.py' 2 >/dev/null 2>&1 || rc=$?; check "suite at its floor greens" 0 "$rc"
rc=0; tools/unittest-floor.sh "$T/suite" 'test_*.py' 3 >/dev/null 2>&1 || rc=$?; check "floor above the count reds" 1 "$rc"
rc=0; tools/unittest-floor.sh "$T/suite" '*_b.py' 1 >/dev/null 2>&1 || rc=$?; check "a failing test reds through the floor" 1 "$rc"
rc=0; tools/unittest-floor.sh "$T/suite" 'test_*.py' x >/dev/null 2>&1 || rc=$?; check "non-integer floor is a usage error" 2 "$rc"
echo "test_unittest_floor: $((5-FAILS)) ok, $FAILS FAIL"
[ "$FAILS" -eq 0 ]
