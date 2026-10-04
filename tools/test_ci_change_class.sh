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
mkdir -p "$repo/crates/memra-engine" "$repo/crates/memra-kv"
printf '[workspace]\nmembers = ["crates/memra-engine", "crates/memra-kv"]\n' > "$repo/Cargo.toml"
printf '[package]\nname = "memra-engine"\n' > "$repo/crates/memra-engine/Cargo.toml"
printf '[package]\nname = "memra-kv"\n' > "$repo/crates/memra-kv/Cargo.toml"
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

# arm 15: a research file a crate includes at compile time is a compile input, not docs
# (lane D day 13: four include_str! sites under crates/ read research/ files; the 2026-09-02
# "zero research/ literals" note the docs-only class rested on was stale).
g reset -q --hard "$base"
mkdir -p "$repo/crates/memra-kv/src/tiered" "$repo/research/lane/fixtures"
printf 'fn f() -> &%sstatic str { include_str!("../../../../research/lane/fixtures/load.csv") }\n' "'" > "$repo/crates/memra-kv/src/tiered/t.rs"
echo "a,b" > "$repo/research/lane/fixtures/load.csv"
g add -A; g commit -q -m include-site
base2=$(g rev-parse HEAD)
echo "a,b,c" >> "$repo/research/lane/fixtures/load.csv"
g add -A; g commit -q -m fixture-only
expect true "arm15 included research file" -- pull_request "$base2" "" "$(g rev-parse HEAD)"
out=$("$cls" pull_request "$base2" "" "$(g rev-parse HEAD)" "$repo")
printf '%s\n' "$out" | grep -qx 'packages=memra-kv' || bad "arm15 reason: $out"
# arm 16: a sibling research file nobody includes is still docs-only, with the include present
g reset -q --hard "$base2"; commit receipt research/lane/RESULTS.md
expect false "arm16 non-included research file beside an include" -- pull_request "$base2" "" "$(g rev-parse HEAD)"
g reset -q --hard "$base"

# arm 17: the multi-line include form (macro name and literal on different lines) is seen too
# (revuto on #611: crates/memra-engine/src/ep_map.rs and two template sites use it).
g reset -q --hard "$base"
mkdir -p "$repo/crates/memra-engine/src" "$repo/research/ep-map"
printf 'fn g() -> &%sstatic str {\n    include_str!(\n        "../../../research/ep-map/example.json"\n    )\n}\n' "'" > "$repo/crates/memra-engine/src/ep.rs"
echo '{}' > "$repo/research/ep-map/example.json"
g add -A; g commit -q -m multiline-include-site
base3=$(g rev-parse HEAD)
echo '{"a":1}' > "$repo/research/ep-map/example.json"
g add -A; g commit -q -m multiline-fixture-only
expect true "arm17 multi-line included research file" -- pull_request "$base3" "" "$(g rev-parse HEAD)"
out=$("$cls" pull_request "$base3" "" "$(g rev-parse HEAD)" "$repo")
printf '%s\n' "$out" | grep -qx 'packages=memra-engine' || bad "arm17 reason: $out"
g reset -q --hard "$base"

# arm 18: the census on THIS repository's tree resolves every known include and nothing is left
# unresolved. Known sites as of 2026-09-21; a removed include drops a line here and this arm
# says so, which is the point.
census=$("$cls" census HEAD "$here")
printf '%s\n' "$census" | grep -q '^?$' && bad "arm18 real census has an unresolved row: $census"
for want in research/spill-b-20260919/fixtures/recompute-load.csv \
            research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja \
            research/reasoning-schema-20260823/ornith15.chat_template.jinja \
            research/ep-placement-map-20260831/example-map-coactivation.json; do
  if [ -e "$here/$want" ]; then
    printf '%s\n' "$census" | grep -qxF "$want" || bad "arm18 real census misses $want: $census"
  fi
done
ok "arm18 real-tree census ($(printf '%s\n' "$census" | grep -c .) paths, none unresolved)"

# arm 14: every selected job fails closed when its output is absent.
ci=$here/.github/workflows/ci.yml
live=$(grep -vE '^\s*#' "$ci")
# Full selection is no longer delegated to PR-modifiable planner code.
changes=$(awk '/^  changes:/{s=1} /^  gates:/{s=0} s{print}' "$ci")
for selected in build clippy server engine portable core lanes arch publish requires_cuda; do
  grep -Fq "      $selected: \"true\"" <<< "$changes" || bad "arm14: full $selected is not a workflow literal"
done
grep -Fq '      packages: ""' <<< "$changes" || bad "arm14: full packages do not expand to workspace"
grep -Fq '      contracts: ""' <<< "$changes" || bad "arm14: full contracts do not run the available inventory"
if grep -Eq 'ci-change-class.sh full|public_ci.py full-plan' <<< "$changes"; then
  bad "arm14: candidate planner can select the full inventory"
fi
grep -q 'tools/test_ci_change_class.sh' <<< "$live" || bad "arm14: missing fixture caller"
for job in build clippy server engine portable arch publish; do
  grep -Fq "!cancelled() && (needs.changes.result != 'success' || needs.changes.outputs.$job != 'false')" <<< "$live" || bad "arm14: $job does not fail closed"
done
if grep -Eq "needs.changes.outputs.(build|clippy|server|engine|portable|arch|publish) == 'true'" <<< "$live"; then
  bad "arm14: fail-open positive comparison"
fi
grep -Fq "needs.changes.result == 'success' && needs.changes.outputs.packages || ''" <<< "$live" || bad "arm14: failed planner can retain a partial package list"
ok "arm14 per-component CI selection fails closed"

echo "test_ci_change_class: $pass arms PASS"
