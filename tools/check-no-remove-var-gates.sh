#!/usr/bin/env bash
# Lint to prevent vacuous gates caused by remove_var on door flags (memra#136).
#
# WHY THIS EXISTS:
# When a gate or test drives a door's OFF arm using `std::env::remove_var("MEMRA_X")`
# rather than pinning the documented off value (`std::env::set_var("MEMRA_X", "0")`),
# the gate passes as long as the door defaults OFF. The moment that door's default
# flips to ON, unsetting resolves to ON: the "off" arm runs the ON path, the gate
# compares ON against ON, and passes while proving nothing (a vacuous pass).
#
# Caught on 2026-09-03 flipping MEMRA_MLA_DECODE_SPLIT to default ON:
# tests/mla_decode_split_gpu.rs::set_door used remove_var for its off arm, so its
# split-vs-unsplit bit-identity comparison would have become split-vs-split.
#
# Rules enforced by this lint:
# 1. Tests (crates/*/tests/**) and gate/probe binaries (crates/*/src/bin/**) must
#    not use `remove_var("MEMRA_*")` to drive an OFF arm.
# 2. Legitimate unsets (e.g. clearing outer shell aliases before a run, or testing
#    absence-of-flag error handling) must be listed in tools/gate-remove-var-allowlist.txt
#    with an explicit explanation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
ALLOWLIST="${ALLOWLIST:-$REPO_ROOT/tools/gate-remove-var-allowlist.txt}"

command -v rg >/dev/null || { echo "check-no-remove-var-gates: rg is required" >&2; exit 2; }

target_dirs=()
while IFS= read -r dir; do
    [[ -d "$dir" ]] && target_dirs+=("$dir")
done < <(find "$REPO_ROOT/crates" -mindepth 2 -maxdepth 3 -type d \( -name 'tests' -o -path '*/src/bin' \) 2>/dev/null | sort)

if (( ${#target_dirs[@]} == 0 )); then
    echo "check-no-remove-var-gates: no tests or bin directories found — census would be vacuously green" >&2
    exit 2
fi

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
raw_matches="$temp_dir/raw_matches"

# Scan for remove_var calling MEMRA_* names
rg -n --no-heading 'remove_var\s*\([[:space:]]*&?"(MEMRA_[A-Z0-9_]+)"' "${target_dirs[@]}" --glob '*.rs' 2>/dev/null > "$raw_matches" || true

# Parse allowlist into an associative array (path:flag -> 1)
declare -A allowlist_map=()
if [[ -f "$ALLOWLIST" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Strip comments and trim whitespace
        line="${line%%#*}"
        line=$(echo "$line" | xargs)
        [[ -z "$line" ]] && continue
        # Allowlist format: <relative_path>:<flag>
        allowlist_map["$line"]=1
    done < "$ALLOWLIST"
fi

uncovered_count=0
allowlisted_count=0
uncovered_lines=()

while IFS= read -r match_line || [[ -n "$match_line" ]]; do
    [[ -z "$match_line" ]] && continue
    # match_line format: /path/to/crates/foo/tests/bar.rs:123:    std::env::remove_var("MEMRA_FLAG");
    file_path="${match_line%%:*}"
    rest="${match_line#*:}"
    lineno="${rest%%:*}"
    content="${rest#*:}"

    # Extract flag name
    flag=$(echo "$content" | rg -o 'MEMRA_[A-Z0-9_]+' | head -n 1 || true)
    [[ -z "$flag" ]] && continue

    rel_path="${file_path#$REPO_ROOT/}"
    key="$rel_path:$flag"

    if [[ -n "${allowlist_map[$key]:-}" ]]; then
        allowlisted_count=$((allowlisted_count + 1))
    else
        uncovered_count=$((uncovered_count + 1))
        uncovered_lines+=("  $rel_path:$lineno: $flag")
        uncovered_lines+=("    $content")
    fi
done < "$raw_matches"

if (( uncovered_count > 0 )); then
    echo "check-no-remove-var-gates: VACUOUS GATE HAZARD (memra#136)" >&2
    echo "Found $uncovered_count un-allowlisted remove_var call(s) on door flags in tests/gate binaries:" >&2
    for line in "${uncovered_lines[@]}"; do
        echo "$line" >&2
    done
    echo "" >&2
    echo "Unsetting a door via remove_var causes a gate to pass vacously if that door's default" >&2
    echo "flips to ON (unset resolves to ON, so OFF arm runs ON and bit-identity is vacuous)." >&2
    echo "Fix:" >&2
    echo "  1. Pin the documented off value (e.g. std::env::set_var(\"<flag>\", \"0\"))" >&2
    echo "  2. If unset is genuinely the state under test (e.g. alias resolution), add" >&2
    echo "     an entry '<path>:<flag>' with comment to tools/gate-remove-var-allowlist.txt" >&2
    exit 1
fi

echo "check-no-remove-var-gates: OK ($allowlisted_count allowlisted unsets, 0 un-allowlisted across ${#target_dirs[@]} target dirs)"
exit 0
