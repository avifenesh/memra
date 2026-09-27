#!/usr/bin/env bash
# Teeth for tools/check-workflow-keys.py (stdlib walker). A duplicate job key must red (the
# 2026-09-21 main shape, which PyYAML's safe_load passes), duplicates nested in a step and in a
# `with:` mapping must red, the block-style constructs real workflow files use must green
# (block scalars whose bodies contain `key:` lines, sibling mappings and consecutive steps with
# the same key names, inline and full-line comments, flow sequences, a quoted key), a flow
# mapping must be refused as "cannot answer" (exit 2, never green), an empty directory must
# red, the real tree must green, and every one of those verdicts must hold on an interpreter
# WITHOUT PyYAML (revuto on #601: a fake `yaml` package that raises ImportError is put first
# on PYTHONPATH; the checker must neither import it nor print a traceback).
set -euo pipefail
cd "$(dirname "$0")/.."
chk=tools/check-workflow-keys.py
T=$(mktemp -d -t workflow-keys-teeth.XXXXXX)
trap 'rm -rf "$T"' EXIT
PASS=0; FAILS=0
ok() { echo "ok   $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL $1"; FAILS=$((FAILS+1)); }
check() { local name=$1 expect=$2 got=$3; if [ "$got" = "$expect" ]; then ok "$name"; else bad "$name (expected exit $expect, got $got)"; fi; }

mkdir -p "$T/dup-job" "$T/dup-nested" "$T/dup-with" "$T/clean" "$T/constructs" "$T/flow" "$T/empty" "$T/shadow/yaml"
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
cat > "$T/dup-with/ci.yml" <<'YML'
name: ci
on: [push]
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - uses: some/action@0000000000000000000000000000000000000000 # v1
        with:
          key: a
          path: b
          key: c
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
cat > "$T/constructs/ci.yml" <<'YML'
name: ci
# jobs:
#   jobs: a full-line comment that looks like keys
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
jobs:
  first:
    runs-on: ubuntu-24.04 # inline comment with a colon: here
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
        with:
          fetch-depth: 0
      - name: block scalar whose body repeats key-shaped lines
        run: |
          echo "name: one"
          echo "name: two"
          run: not a key
          run: not a key either
      - name: folded block with indicators
        run: >-
          printf 'a: %s\n' one
          printf 'a: %s\n' two
      - name: quoted keys, one whose content is an alias indicator
        "env":
          A: 1
          "*": star
        run: echo "$A" # trailing comment
  second:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
      - name: same step keys as the job above, legal
        run: echo again
YML
cat > "$T/flow/ci.yml" <<'YML'
name: ci
on: [push]
jobs:
  build:
    runs-on: ubuntu-24.04
    strategy: {matrix: {os: [a, b]}}
    steps:
      - run: echo a
YML
cat > "$T/shadow/yaml/__init__.py" <<'PY'
raise ImportError("shadowed: this interpreter has no PyYAML (tools/test_workflow_keys.sh)")
PY

# The control: safe_load accepts the duplicate-job file, which is why this tool exists. Not
# counted when PyYAML is absent here; the checker itself needs none (the arms below prove it).
if python3 -c "import yaml" 2>/dev/null; then
    rc=0; python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$T/dup-job/ci.yml" || rc=$?
    check "control: yaml.safe_load accepts the duplicate job key (exit 0)" 0 "$rc"
else
    echo "skip control: PyYAML absent on this interpreter (the checker needs none)"
fi

verdicts() {
    # verdicts <label> <env-prefix...>: the same eight verdicts, with and without PyYAML shadowed
    local label=$1; shift
    local out rc
    rc=0; out=$("$@" python3 "$chk" "$T/dup-job" 2>&1) || rc=$?
    check "$label duplicate job key reds" 1 "$rc"
    if printf '%s\n' "$out" | grep -q "duplicate mapping key 'portable-suites' at line 12 column 3 (first at line 4)"; then ok "$label the refusal names the key, its line and column and the first line"; else bad "$label refusal text: $out"; fi
    rc=0; "$@" python3 "$chk" "$T/dup-nested" >/dev/null 2>&1 || rc=$?
    check "$label duplicate nested key (two run: in one step) reds" 1 "$rc"
    rc=0; out=$("$@" python3 "$chk" "$T/dup-with" 2>&1) || rc=$?
    check "$label duplicate key inside with: reds" 1 "$rc"
    if printf '%s\n' "$out" | grep -q "duplicate mapping key 'key' at line 11 column 11 (first at line 9)"; then ok "$label the with: refusal names line 11 column 11"; else bad "$label with: refusal text: $out"; fi
    rc=0; out=$("$@" python3 "$chk" "$T/clean" 2>&1) || rc=$?
    check "$label clean file greens" 0 "$rc"
    if printf '%s\n' "$out" | grep -q "jobs gates, build"; then ok "$label the green names the jobs"; else bad "$label green text: $out"; fi
    rc=0; out=$("$@" python3 "$chk" "$T/constructs" 2>&1) || rc=$?
    check "$label block scalars, sibling mappings, comments, flow sequences and quoted keys (one is \"*\") green" 0 "$rc"
    if printf '%s\n' "$out" | grep -q "jobs first, second"; then ok "$label the constructs file lists its two jobs"; else bad "$label constructs text: $out"; fi
    rc=0; out=$("$@" python3 "$chk" "$T/flow" 2>&1) || rc=$?
    check "$label a flow mapping is refused as cannot-answer (exit 2, never green)" 2 "$rc"
    if printf '%s\n' "$out" | grep -q "CANNOT ANSWER: .*line 6: flow mapping"; then ok "$label the cannot-answer names the line and the construct"; else bad "$label flow text: $out"; fi
    rc=0; "$@" python3 "$chk" "$T/empty" >/dev/null 2>&1 || rc=$?
    check "$label empty workflow directory reds (vacuity)" 1 "$rc"
    rc=0; out=$("$@" python3 "$chk" 2>&1) || rc=$?
    check "$label the tree's own .github/workflows greens" 0 "$rc"
    if printf '%s\n' "$out" | grep -qE 'Traceback|ModuleNotFoundError|ImportError'; then bad "$label a traceback or import error leaked: $out"; else ok "$label no traceback, no import error in any verdict"; fi
}
verdicts "default:" env
verdicts "no-PyYAML:" env PYTHONPATH="$T/shadow"
# The shadow really shadows: importing yaml under it must raise.
rc=0; PYTHONPATH="$T/shadow" python3 -c "import yaml" 2>/dev/null || rc=$?
check "the shadow package raises on import yaml (the no-PyYAML arm is real)" 1 "$rc"

echo "test_workflow_keys: $PASS ok, $FAILS FAIL"
[ "$FAILS" -eq 0 ]
