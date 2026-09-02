#!/usr/bin/env bash
# Teeth for tools/flags-table-census.py and its wiring (memra #22).
#
# Every arm forces the outcome it asserts: a fixture that only ever watches the real registry
# pass is not evidence the check can fail. Arms: the real registry is green; an unescaped pipe
# inside a backticked cell is red and the message names the line; a three-cell row in a
# two-column table is red; an escaped pipe is green; pipes inside a fenced code block are
# ignored; a file with no table refuses (exit 2) rather than passing vacuously; and the
# wiring arms prove the census has a caller in tools/docs-registry-census.sh, that
# docs-registry-census.sh has a caller in ci.yml, and that this fixture itself is reached
# from a step ci.yml already runs (tools/test_check_flags.sh chains into it). Wiring is
# checked on comment-stripped text, so this comment cannot satisfy it.
#
# CPU only, no network, no cargo. Throwaway files under mktemp.
set -uo pipefail
cd -- "$(dirname -- "$0")/.."
CENSUS=tools/flags-table-census.py
[[ -x "$CENSUS" ]] || { echo "test_docs_registry_census: missing or non-executable $CENSUS" >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  ok   $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL $1" >&2; }

# --- arm 1: the real registry is green -------------------------------------------------------
if out=$(python3 "$CENSUS" docs/FLAGS.md 2>&1); then
    ok "real docs/FLAGS.md: $out"
else
    bad "real docs/FLAGS.md should be green; got: $out"
fi

# --- arm 2: unescaped pipe inside a backticked cell -> red, line named ------------------------
cat > "$tmp/unescaped.md" <<'EOF'
# fixture

| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | a clean row |
| `MEMRA_BAD` | off | logs `route=spec|plain` per request |
EOF
rc=0; out=$(python3 "$CENSUS" "$tmp/unescaped.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'unescaped.md:6: 4 cells' <<<"$out"; then
    ok "unescaped pipe is red and names line 6"
else
    bad "unescaped pipe: rc=$rc out=$out"
fi

# --- arm 3: three-cell row in a two-column table -> red ------------------------------------
cat > "$tmp/twocol.md" <<'EOF'
| flag | what it does |
|---|---|
| `MEMRA_A` | fine |
| `MEMRA_B` | on | a default cell that has no header |
EOF
rc=0; out=$(python3 "$CENSUS" "$tmp/twocol.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'twocol.md:4: 3 cells, but the table header at line 1 has 2' <<<"$out"; then
    ok "three cells in a two-column table is red"
else
    bad "two-column overflow: rc=$rc out=$out"
fi

# --- arm 4: escaped pipe -> green ----------------------------------------------------------
cat > "$tmp/escaped.md" <<'EOF'
| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | logs `route=spec\|plain` per request |
EOF
if out=$(python3 "$CENSUS" "$tmp/escaped.md" 2>&1) && grep -q 'rows=1' <<<"$out"; then
    ok "escaped pipe is green"
else
    bad "escaped pipe should be green; got: $out"
fi

# --- arm 5: pipes inside a fenced block are not table rows -----------------------------------
cat > "$tmp/fenced.md" <<'EOF'
| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | fine |

```
| this | is | a | log | line | not | a | row |
```
EOF
if out=$(python3 "$CENSUS" "$tmp/fenced.md" 2>&1) && grep -q 'tables=1 rows=1' <<<"$out"; then
    ok "fenced pipes are ignored"
else
    bad "fenced block: got: $out"
fi

# --- arm 6: no table at all -> refuses (exit 2), never a vacuous pass ----------------------
printf '# nothing here\n\nprose only\n' > "$tmp/empty.md"
rc=0; out=$(python3 "$CENSUS" "$tmp/empty.md" 2>&1) || rc=$?
if (( rc == 2 )); then
    ok "a registry with no table refuses (exit 2)"
else
    bad "no-table input should exit 2; rc=$rc out=$out"
fi

# --- wiring arms (comment-stripped text) -----------------------------------------------------
# Capture, then grep: `producer | grep -q` under pipefail races the producer SIGPIPE and reads
# a real match as a miss on any file longer than one pipe buffer (ci.yml is one).
strip() { sed -e 's/^[[:space:]]*#.*$//' "$1"; }
census_text=$(strip tools/docs-registry-census.sh)
ci_text=$(strip .github/workflows/ci.yml)
flags_fixture_text=$(strip tools/test_check_flags.sh)
if grep -q 'flags-table-census.py' <<<"$census_text"; then
    ok "docs-registry-census.sh invokes the table census"
else
    bad "docs-registry-census.sh does not invoke flags-table-census.py"
fi
if grep -q 'tools/docs-registry-census.sh' <<<"$ci_text"; then
    ok "ci.yml runs docs-registry-census.sh"
else
    bad "ci.yml does not run docs-registry-census.sh"
fi
if grep -q 'tools/test_docs_registry_census.sh' <<<"$flags_fixture_text"; then
    ok "test_check_flags.sh chains into this fixture (its ci.yml step is the caller)"
else
    bad "this fixture has no caller: test_check_flags.sh does not chain into it"
fi

EXPECTED_ASSERTIONS=9
total=$((pass + fail))
printf '\ntest_docs_registry_census: %d passed, %d failed (%d assertions, expected %d)\n' \
    "$pass" "$fail" "$total" "$EXPECTED_ASSERTIONS"
if (( total != EXPECTED_ASSERTIONS )); then
    printf 'test_docs_registry_census: BROKEN FIXTURE, recorded %d assertions, expected %d\n' \
        "$total" "$EXPECTED_ASSERTIONS" >&2
    exit 3
fi
(( fail == 0 ))
