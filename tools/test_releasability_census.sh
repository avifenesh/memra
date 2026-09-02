#!/usr/bin/env bash
# Teeth for the three releasability censuses. Every refusal is FORCED, so each is proven able
# to fail for the right reason — and the last arm proves ci.yml actually calls them, because a
# gate nobody invokes rots exactly like the thing it guards.
#
# The lesson these encode: on 2026-08-23 tools/test_release_guard.sh had five green arms while
# main was unreleasable, because all five inspected fixtures. So the censuses under test here
# read the REAL manifests and REAL workflows (arms 5, 8, 11 assert that on this very repo), and
# fixtures are used ONLY to force failures — you cannot break the real tree to prove a refusal.
# Arms 1 and 6 reproduce the two defects that actually killed v0.105.0 and v0.106.0.
#
# CPU only: no cargo, no nvcc, no GPU, no network. Runs in ~1 s.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
census_pub=tools/workspace-publish-census.sh
census_stub=tools/stub-abi-census.py
census_arch=tools/arch-matrix-census.sh
ci=.github/workflows/ci.yml
pub_wf=.github/workflows/publish.yml
rel_wf=.github/workflows/release.yml
build_rs=crates/memra-engine/build.rs

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# ── publish census ────────────────────────────────────────────────────────────────────────
# Arm 1 — THE DEFECT: a publishable member missing from the list (memra-reference, v0.106.0).
sed -E 's/memra-gguf memra-reference/memra-gguf/' "$pub_wf" > "$tmp/no-reference.yml"
# Assert on the LIST, not the file: publish.yml's comments name memra-reference too, so a
# whole-file grep would match the documentation and hide a no-op sed.
listed() { sed -n '/for crate in/,/; do/p' "$1" | tr -d '\\\n'; }
listed "$tmp/no-reference.yml" | grep -q 'memra-reference' \
  && fail "arm 1 setup: memra-reference still in the list, the sed did not bite"
if out=$("$census_pub" Cargo.toml "$tmp/no-reference.yml" 2>&1); then
  fail "arm 1: census accepted a list missing memra-reference: $out"
fi
echo "$out" | grep -q 'memra-reference' \
  || fail "arm 1: refusal does not name the missing member: $out"

# Arm 2 — order: memra-engine listed before its own workspace dep memra-reference.
sed -E 's/memra-gguf memra-reference/memra-gguf/; s/memra-validate memra-engine/memra-validate memra-engine memra-reference/' \
  "$pub_wf" > "$tmp/bad-order.yml"
if out=$("$census_pub" Cargo.toml "$tmp/bad-order.yml" 2>&1); then
  fail "arm 2: census accepted a non-topological list: $out"
fi
echo "$out" | grep -q 'topological' \
  || fail "arm 2: refusal is not the ordering one: $out"

# Arm 3 — a ghost: a listed crate that is not a workspace member.
sed -E 's/memra-gguf memra-reference/memra-gguf memra-ghost memra-reference/' "$pub_wf" \
  > "$tmp/ghost.yml"
if out=$("$census_pub" Cargo.toml "$tmp/ghost.yml" 2>&1); then
  fail "arm 3: census accepted a ghost crate: $out"
fi
echo "$out" | grep -q 'memra-ghost' || fail "arm 3: refusal does not name the ghost: $out"

# Arm 4 — a publish = false crate in the list (memra-probe is the real one).
sed -E 's/memra-gguf memra-reference/memra-gguf memra-probe memra-reference/' "$pub_wf" \
  > "$tmp/probe.yml"
if out=$("$census_pub" Cargo.toml "$tmp/probe.yml" 2>&1); then
  fail "arm 4: census accepted memra-probe (publish = false) in the list: $out"
fi
echo "$out" | grep -q 'publish = false' \
  || fail "arm 4: refusal is not the publish=false one: $out"

# Arm 5 — the REAL repo must pass.
out=$("$census_pub" 2>&1) || fail "arm 5: census refused the real workspace: $out"
echo "$out" | grep -q 'publish-census: OK' || fail "arm 5: no OK line: $out"

# ── stub ABI census ───────────────────────────────────────────────────────────────────────
# Arm 6 — THE OTHER DEFECT: strip the two entry points 58ce746ad3 added, exactly the state that
# made sm_89/sm_90a stop linking and killed two tags.
mirror=$tmp/crate
mkdir -p "$mirror"
cp -r crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs "$mirror/"
python3 - "$mirror/cu/mmq_fp8_blk_stub.cu" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for sym in ("memra_mmq_fp8_blk_quantize_act", "memra_mmq_fp8_blk_grouped"):
    s = re.sub(r'extern "C" int ' + sym + r'\([^{]*\{[^}]*\}\n', "", s, flags=re.S)
    # Assert the DEFINITION is gone, not the name: this stub's header comment names both
    # symbols (it documents the defect), so `sym not in s` would fail on the prose.
    assert f'int {sym}(' not in s, f"teeth setup: failed to strip {sym}"
open(p, "w").write(s)
PY
if out=$("$census_stub" "$mirror" 2>&1); then
  fail "arm 6: census accepted a stub missing two Rust-referenced symbols: $out"
fi
echo "$out" | grep -q 'memra_mmq_fp8_blk_grouped' \
  || fail "arm 6: refusal does not name the missing symbol: $out"

# Arm 7 — a stub whose real twin does not exist.
mirror2=$tmp/crate2
mkdir -p "$mirror2/cu"
cp crates/memra-engine/build.rs "$mirror2/"
cp -r crates/memra-engine/src "$mirror2/"
cp crates/memra-engine/cu/mmq_fp8_blk_stub.cu "$mirror2/cu/orphan_stub.cu"
if out=$("$census_stub" "$mirror2" 2>&1); then
  fail "arm 7: census accepted a stub with no real twin: $out"
fi
echo "$out" | grep -q 'no real twin' || fail "arm 7: wrong refusal: $out"

# Arm 8 — the REAL crate must pass.
out=$("$census_stub" 2>&1) || fail "arm 8: census refused the real crate: $out"
echo "$out" | grep -q 'stub-abi-census: OK' || fail "arm 8: no OK line: $out"

# ── arch matrix census ────────────────────────────────────────────────────────────────────
# Arm 9 — THE HOLE: a ci.yml that does not compile an arch the release matrix builds. This is
# the pre-2026-08-23 state, in which sm_89 was first compiled by the tag.
grep -v 'cuda_arch' "$ci" | grep -v 'MEMRA_CUDA_ARCH' > "$tmp/ci-no-arch.yml"
if out=$("$census_arch" "$tmp/ci-no-arch.yml" "$rel_wf" 2>&1); then
  fail "arm 9: census accepted a ci.yml that compiles no release arch: $out"
fi
echo "$out" | grep -q 'never compiles' || fail "arm 9: wrong refusal: $out"

# Arm 10 — an arch build.rs does not accept (typo protection).
# Inject the bad arch into whatever the matrix currently holds, rather than matching a literal
# list: the list changed (sm_89 was removed from the release matrix on 2026-08-23) and a
# hardcoded pattern silently stopped biting, which the setup assertion below caught.
sed -E 's/^([[:space:]]*cuda_arch: \[)(.*)\]$/\1\2, "77z"]/' "$rel_wf" > "$tmp/rel-bad-arch.yml"
grep -q '77z' "$tmp/rel-bad-arch.yml" || fail "arm 10 setup: sed did not bite"
if out=$("$census_arch" "$ci" "$tmp/rel-bad-arch.yml" 2>&1); then
  fail "arm 10: census accepted an arch build.rs rejects: $out"
fi
echo "$out" | grep -q '77z' || fail "arm 10: refusal does not name the bad arch: $out"

# Arm 11 — the REAL workflows must pass.
out=$("$census_arch" 2>&1) || fail "arm 11: census refused the real workflows: $out"
echo "$out" | grep -q 'arch-census: OK' || fail "arm 11: no OK line: $out"

# Arm 12 — THE INVARIANT THAT MAKES "advisory" SAFE: an arch whose fatbin census is
#          non-blocking must never appear in release.yml's matrix. Without this, the advisory
#          list is just a way to ship a panicking binary quietly — which is what v0.107.0's
#          sm_89 asset did.
printf '90a  # pretend advisory\n' > "$tmp/adv-shipped.txt"
if out=$("$census_arch" "$ci" "$rel_wf" "$build_rs" "$tmp/adv-shipped.txt" 2>&1); then
  fail "arm 12: census allowed an advisory arch to stay in release.yml's matrix: $out"
fi
echo "$out" | grep -q 'panics at Engine::func' \
  || fail "arm 12: refusal does not name the runtime consequence: $out"

# Arm 13 — the other direction: an advisory arch nobody compiles is a dead entry measuring
#          nothing, so it must also refuse.
# The uncompiled arch has to come from a FIXTURE ci.yml: this arm used to lean on the real
# ci.yml never compiling 100a, and the b200-prep lane (2026-09-01) added a 100a compile cell,
# which flipped the arm's premise and failed CI for the right thing happening — the exact
# fixture-pinned-to-transient-content trap arm 14's comment warns about. Same idiom as arms
# 9/10: derive the fixture from the live file and assert the sed actually bit.
sed -e 's/"100a", //' \
    -e 's/MEMRA_CUDA_ARCH: "100a"/MEMRA_CUDA_ARCH: "120a"/' \
    "$ci" > "$tmp/ci-no-100a.yml"
if grep -m1 'cuda_arch:' "$tmp/ci-no-100a.yml" | grep -q '"100a"' \
   || grep -qE 'MEMRA_CUDA_ARCH:[[:space:]]*"?100a"?' "$tmp/ci-no-100a.yml"; then
  fail "arm 13 setup: sed did not remove 100a compile coverage"
fi
printf '100a  # advisory but uncompiled\n' > "$tmp/adv-dead.txt"
if out=$("$census_arch" "$tmp/ci-no-100a.yml" "$rel_wf" "$build_rs" "$tmp/adv-dead.txt" 2>&1); then
  fail "arm 13: census allowed an advisory arch that ci never compiles: $out"
fi
echo "$out" | grep -q 'measures nothing' || fail "arm 13: wrong refusal: $out"

# Arm 14 — the REAL files must satisfy the invariant. Asserts the invariant HOLDS, not a
# particular advisory list: sm_89 was advisory for part of 2026-08-23 and is not any more (its
# census hits turned out to be declared-and-reasoned, not defects), and a fixture pinned to that
# transient content would have failed for the right thing happening.
out=$("$census_arch" 2>&1) || fail "arm 14: real workflows violate the advisory invariant: $out"
echo "$out" | grep -q 'advisory \[' \
  || fail "arm 14: OK line does not report the advisory set at all: $out"

# ── wiring ────────────────────────────────────────────────────────────────────────────────
# Arm 12 — ci.yml must actually invoke all three censuses, this fixture, and the arch mirror
# job. Comment lines are stripped first: a plain substring search would match the explanatory
# comments above each step and pass even after the step itself was deleted.
live=$(grep -vE '^\s*#' "$ci")
for needed in \
  "tools/workspace-publish-census.sh" \
  "tools/stub-abi-census.py" \
  "tools/arch-matrix-census.sh" \
  "tools/test_releasability_census.sh" \
  "research/b200-kernel-twins-dry-20260901/check-layouts.sh" \
  "research/b200-kernel-twins-dry-20260901/check-nvfp4.sh" \
  "research/b200-kernel-twins-dry-20260901/check-fp8.sh" \
  "tools/test_install_b200_policy.sh" \
  "tools/test_b200_phase0_harness.sh" \
  "b200_dry_policy_tests" \
  "b200_runs_static_nvfp4_checks_without_the_sm120_fatbin_cell" \
  "synthetic_fixture_covers_codes_scales_and_exact_activation_blocks" \
  "release-arch-mirror:"
do
  echo "$live" | grep -qF "$needed" \
    || fail "arm 12: $ci does not invoke '$needed' (comments stripped) — the census exists but nothing runs it"
done
# And the two tag workflows must run the structural censuses before they spend a matrix on it.
for wf in "$rel_wf" "$pub_wf"; do
  grep -vE '^\s*#' "$wf" | grep -qF 'tools/workspace-publish-census.sh' \
    || fail "arm 12: $wf does not run the publish census"
done

echo "releasability-census fixture: 15 arms PASS"
