#!/usr/bin/env bash
# Regression tests for tools/check-no-remove-var-gates.sh (memra#136).
#
# Verifies that:
# 1. Real repository passes with 0 un-allowlisted unsets.
# 2. Injected remove_var on a door flag in tests/ fails with exit code 1.
# 3. Allowlisted remove_var passes with exit code 0.
# 4. Explicit set_var("...", "0") passes with exit code 0.

set -uo pipefail

cd -- "$(dirname -- "$0")/.."
SCRIPT=$PWD/tools/check-no-remove-var-gates.sh
[[ -x "$SCRIPT" ]] || { echo "test_check_no_remove_var_gates: missing $SCRIPT" >&2; exit 2; }
command -v rg >/dev/null || { echo "test_check_no_remove_var_gates: rg is required" >&2; exit 2; }

pass=0
fail=0

check_case() {
    local name=$1 expected_rc=$2
    shift 2
    local out rc=0
    out=$("$@" 2>&1) || rc=$?
    if [[ $rc -eq $expected_rc ]]; then
        echo "PASS: $name"
        pass=$((pass + 1))
    else
        echo "FAIL: $name (expected rc=$expected_rc, got rc=$rc)"
        echo "$out"
        fail=$((fail + 1))
    fi
}

echo "=== Case 1: Real tree is clean ==="
check_case "real-tree-clean" 0 "$SCRIPT"

echo "=== Case 2: Throwaway fixture with un-allowlisted remove_var ==="
tmp_repo=$(mktemp -d)
trap 'rm -rf "$tmp_repo"' EXIT
mkdir -p "$tmp_repo/crates/memra-test/tests" "$tmp_repo/tools"
cp "$SCRIPT" "$tmp_repo/tools/check-no-remove-var-gates.sh"
touch "$tmp_repo/tools/gate-remove-var-allowlist.txt"

cat <<'RUST' > "$tmp_repo/crates/memra-test/tests/gate_test.rs"
fn test_unsetting_door() {
    unsafe { std::env::remove_var("MEMRA_TEST_DOOR"); }
}
RUST

check_case "uncovered-remove-var-fails" 1 "$tmp_repo/tools/check-no-remove-var-gates.sh"

echo "=== Case 3: Allowlisted remove_var passes ==="
echo "crates/memra-test/tests/gate_test.rs:MEMRA_TEST_DOOR # intentional test unset" >> "$tmp_repo/tools/gate-remove-var-allowlist.txt"
check_case "allowlisted-remove-var-passes" 0 "$tmp_repo/tools/check-no-remove-var-gates.sh"

echo "=== Case 4: Pinned set_var passes without allowlist ==="
cat <<'RUST' > "$tmp_repo/crates/memra-test/tests/gate_test.rs"
fn test_pinned_door() {
    unsafe { std::env::set_var("MEMRA_TEST_DOOR", "0"); }
}
RUST
: > "$tmp_repo/tools/gate-remove-var-allowlist.txt"
check_case "pinned-set-var-passes" 0 "$tmp_repo/tools/check-no-remove-var-gates.sh"

echo ""
echo "test_check_no_remove_var_gates summary: $pass passed, $fail failed"
[[ $fail -eq 0 ]] || exit 1
