#!/usr/bin/env bash
# Teeth for tools/check-workflow-keys.py: a duplicate job key must red (the 2026-09-21 main
# shape, which PyYAML's safe_load passes), a duplicate nested key must red, a clean file must
# green, an empty directory must red (vacuity), and the real tree must green.
set -euo pipefail
cd "$(dirname "$0")/.."
chk=tools/check-workflow-keys.py
T=$(mktemp -d -t workflow-keys-teeth.XXXXXX)
trap 'rm -rf "$T"' EXIT
FAILS=0
check() { local name=$1 expect=$2 got=$3; if [ "$got" = "$expect" ]; then echo "ok   $name"; else echo "FAIL $name (expected exit $expect, got $got)"; FAILS=$((FAILS+1)); fi; }

mkdir -p "$T/dup-job" "$T/dup-nested" "$T/clean" "$T/empty"
cat > "$T/dup-job/ci.yml" <<'YML'
name: ci
on: [push]
jobs:
  portable-suites:
    runs-on: ubuntu-24.04
    steps:
      - run: bash tools/ci-portable.sh
  boundary:
    runs-on: ubuntu-24.04
    steps:
      - run: echo boundary
  portable-suites:
    runs-on: ubuntu-24.04
    steps:
      - run: tools/portable-suites.sh
YML
cat > "$T/dup-nested/ci.yml" <<'YML'
name: ci
on: [push]
jobs:
  gates:
    runs-on: ubuntu-24.04
    steps:
      - name: one
        run: echo a
        run: echo b
YML
cat > "$T/clean/ci.yml" <<'YML'
name: ci
on: [push]
jobs:
  gates:
    runs-on: ubuntu-24.04
    steps:
      - run: echo a
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: echo b
YML

# The control: safe_load accepts the duplicate-job file, which is why this tool exists.
rc=0; python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$T/dup-job/ci.yml" || rc=$?
check "control: yaml.safe_load accepts the duplicate job key (exit 0)" 0 "$rc"

out=""; rc=0; out=$("$chk" "$T/dup-job" 2>&1) || rc=$?
check "duplicate job key reds" 1 "$rc"
if printf '%s\n' "$out" | grep -q "duplicate mapping key 'portable-suites' at line 12"; then echo "ok   the refusal names the key and its line"; else echo "FAIL refusal text: $out"; FAILS=$((FAILS+1)); fi
rc=0; "$chk" "$T/dup-nested" >/dev/null 2>&1 || rc=$?
check "duplicate nested key (two run: in one step) reds" 1 "$rc"
rc=0; out=$("$chk" "$T/clean" 2>&1) || rc=$?
check "clean file greens" 0 "$rc"
if printf '%s\n' "$out" | grep -q "jobs gates, build"; then echo "ok   the green names the jobs"; else echo "FAIL green text: $out"; FAILS=$((FAILS+1)); fi
rc=0; "$chk" "$T/empty" >/dev/null 2>&1 || rc=$?
check "empty workflow directory reds (vacuity)" 1 "$rc"
rc=0; "$chk" >/dev/null 2>&1 || rc=$?
check "the tree's own .github/workflows greens" 0 "$rc"

echo "test_workflow_keys: $((9-FAILS)) ok, $FAILS FAIL"
[ "$FAILS" -eq 0 ]
