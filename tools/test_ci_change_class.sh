#!/usr/bin/env bash
# Teeth for tools/ci-change-class.sh: every refusal-to-skip forced, the one skip proven, and
# the ci.yml wiring asserted in its fail-closed form. Throwaway repos under mktemp, no network.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cls=$here/tools/ci-change-class.sh
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pass=0
ok()  { pass=$((pass+1)); echo "ok   $*"; }
bad() { echo "FAIL $*" >&2; exit 1; }

repo=$tmp/repo
git init -q "$repo"
g() { git -C "$repo" "$@"; }
g config user.email t@t; g config user.name t
commit() { # commit <msg> <path>...
  local msg=$1; shift
  for p in "$@"; do mkdir -p "$repo/$(dirname "$p")"; echo "$RANDOM" >> "$repo/$p"; done
  g add -A; g commit -q -m "$msg"
}
commit root README.md crates/memra-engine/src/lib.rs
base=$(g rev-parse HEAD)

expect() { # expect <code> <arm-name> -- <args...>
  local want=$1 name=$2; shift 3
  local out rc=0
  out=$("$cls" "$@" "$repo") || rc=$?
  [ "$rc" -eq 0 ] || bad "$name: exit $rc (must never be non-zero): $out"
  printf '%s\n' "$out" | grep -qx "code=$want" || bad "$name: wanted code=$want, got: $out"
  ok "$name ($(printf '%s\n' "$out" | grep '^reason='))"
}

# arm 1: docs-only PR skips the compile
commit docs docs/X.md research/lane/RESULTS.md agent-knowledge/gpu/a.md LICENSE
expect false "arm1 docs-only pull_request" -- pull_request "$base" "" "$(g rev-parse HEAD)"
# arm 2: docs-only push skips too
expect false "arm2 docs-only push" -- push "" "$base" "$(g rev-parse HEAD)"
# arm 3: a source file compiles
commit src crates/memra-engine/src/x.rs
expect true "arm3 source file" -- pull_request "$base" "" "$(g rev-parse HEAD)"
# arm 4: mixed (docs + tools) compiles
g reset -q --hard "$base"; commit mixed docs/Y.md tools/thing.sh
expect true "arm4 docs+tools" -- pull_request "$base" "" "$(g rev-parse HEAD)"
# arm 5: a crate README is a package input, not docs
g reset -q --hard "$base"; commit crate-readme crates/memra-gguf/README.md
expect true "arm5 crate README" -- pull_request "$base" "" "$(g rev-parse HEAD)"
# arm 6: workflow files compile everything
g reset -q --hard "$base"; commit wf .github/workflows/ci.yml
expect true "arm6 workflow file" -- push "" "$base" "$(g rev-parse HEAD)"
# arm 7: Cargo manifests compile
g reset -q --hard "$base"; commit cargo Cargo.toml
expect true "arm7 Cargo.toml" -- push "" "$base" "$(g rev-parse HEAD)"
# arm 8: zero before sha (branch creation / force-push) fails closed
g reset -q --hard "$base"; commit docs2 docs/Z.md
expect true "arm8 zero before" -- push "" 0000000000000000000000000000000000000000 "$(g rev-parse HEAD)"
# arm 9: unreachable base fails closed
expect true "arm9 unreachable base" -- pull_request deadbeefdeadbeefdeadbeefdeadbeefdeadbeef "" "$(g rev-parse HEAD)"
# arm 10: empty diff fails closed
expect true "arm10 empty diff" -- push "" "$(g rev-parse HEAD)" "$(g rev-parse HEAD)"
# arm 11: unknown event fails closed
expect true "arm11 unknown event" -- workflow_dispatch "" "" "$(g rev-parse HEAD)"
# arm 12: missing args fail closed, exit 0
out=$("$cls" 2>&1) && printf '%s\n' "$out" | grep -qx 'code=true' || bad "arm12 missing args: $out"
ok "arm12 missing args"
# arm 13: a non-repo directory fails closed, exit 0
out=$("$cls" push "" "$base" "$base" "$tmp") && printf '%s\n' "$out" | grep -qx 'code=true' || bad "arm13 non-repo: $out"
ok "arm13 non-repo dir"

# arm 14: ci.yml wiring, in the fail-closed form. Every compile job must gate on
# `code != 'false'` (a missing output compiles) and none on `== 'true'` (a missing output
# would skip the compile). Comment lines stripped so this cannot be satisfied by prose.
ci=$here/.github/workflows/ci.yml
live=$(grep -vE '^\s*#' "$ci")
printf '%s\n' "$live" | grep -q 'tools/ci-change-class.sh' || bad "arm14: ci.yml does not run the classifier"
printf '%s\n' "$live" | grep -q 'tools/test_ci_change_class.sh' || bad "arm14: ci.yml does not run this fixture"
if printf '%s\n' "$live" | grep -q "outputs.code == 'true'"; then
  bad "arm14: ci.yml gates a job on code == 'true' (fail-open: a missing output skips the compile)"
fi
gated=$(printf '%s\n' "$live" | grep -c "needs.changes.outputs.code != 'false'" || true)
[ "$gated" -ge 6 ] || bad "arm14: expected at least 6 compile jobs gated on code != 'false', found $gated"
printf '%s\n' "$live" | grep -q '!cancelled()' || bad "arm14: the gate must carry a status function (!cancelled()) or GitHub re-implies success() and a failed classifier skips every compile"
ok "arm14 ci.yml wiring ($gated gated jobs, fail-closed form)"

echo "test_ci_change_class: $pass arms PASS"
