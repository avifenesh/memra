#!/usr/bin/env bash
# Teeth for tools/release-guard.sh — every refusal proven able to fail for the right
# reason, per the GATE-INTEGRITY-20260819 rule that a gate without a fixture rots.
# CPU only: throwaway repos under mktemp, a bare repo standing in for origin, no
# network, no cargo. Wired into ci.yml next to the other fixture rounds.
#
# Arms:
#   1. version mismatch          -> refused, message names both versions
#   2. version match, no claim   -> refused, message names the claim branch
#   3. version match + claim     -> passes
#   4. non-vX.Y.Z tag            -> refused
#   5. wiring census             -> release.yml, publish.yml and ci.yml actually call
#                                   the guard/fixture (A-18: a fixture with no caller
#                                   is born invisible)
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
guard="$here/tools/release-guard.sh"

tmp=$(mktemp -d /tmp/relguard-test.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

# Stage: a bare "origin" and a work repo whose Cargo.toml says 0.103.0.
git init -q --bare "$tmp/origin.git"
git init -q "$tmp/work"
cd "$tmp/work"
git config user.email relguard@test && git config user.name relguard
# The pins matter: release-guard also censuses [workspace.dependencies], and a manifest with
# no pins at all is refused as a vacuous pass (arm 8), so the happy-path manifest must carry
# them exactly as the real Cargo.toml does.
cat > Cargo.toml <<'EOF'
[workspace]
members = []

[workspace.package]
version = "0.103.0"
license = "MIT"

[workspace.dependencies]
memra-gguf = { path = "crates/memra-gguf", version = "=0.103.0" }
memra-engine = { path = "crates/memra-engine", version = "=0.103.0" }
EOF
git add Cargo.toml && git commit -qm seed
git remote add origin "$tmp/origin.git"
git push -q origin HEAD:main

# 1. Mismatch: tag v0.104.0 against workspace 0.103.0 must be refused.
if out=$("$guard" v0.104.0 Cargo.toml origin 2>&1); then
  fail "arm 1: guard accepted v0.104.0 over workspace 0.103.0: $out"
fi
echo "$out" | grep -q "v0.103.0 != tag v0.104.0" \
  || fail "arm 1: refusal does not name both versions: $out"

# 2. Match but unclaimed: v0.103.0 with no claim branch must be refused.
if out=$("$guard" v0.103.0 Cargo.toml origin 2>&1); then
  fail "arm 2: guard accepted an unclaimed v0.103.0: $out"
fi
echo "$out" | grep -q "release/claim-v0.103.0" \
  || fail "arm 2: refusal does not name the claim branch: $out"

# 3. Match + claim: must pass.
git push -q origin HEAD:refs/heads/release/claim-v0.103.0
out=$("$guard" v0.103.0 Cargo.toml origin 2>&1) \
  || fail "arm 3: guard refused a matched, claimed v0.103.0: $out"
echo "$out" | grep -q "v0.103.0 OK" || fail "arm 3: no OK line: $out"

# 4. Garbage tag shape refused.
if out=$("$guard" perf-baseline Cargo.toml origin 2>&1); then
  fail "arm 4: guard accepted non-release tag 'perf-baseline': $out"
fi

# 5. Wiring census: the guard is only teeth if the workflows call it.
grep -q 'tools/release-guard.sh' "$here/.github/workflows/release.yml" \
  || fail "arm 5: release.yml does not call release-guard.sh"
grep -Eq '^\s+needs: guard' "$here/.github/workflows/release.yml" \
  || fail "arm 5: release.yml build matrix does not gate on the guard job"
grep -q 'tools/release-guard.sh' "$here/.github/workflows/publish.yml" \
  || fail "arm 5: publish.yml does not call release-guard.sh"
grep -q 'test_release_guard.sh' "$here/.github/workflows/ci.yml" \
  || fail "arm 5: ci.yml does not run this fixture"

# 6. PARTIAL BUMP: [workspace.package].version moved to 0.104.0 but the pins still say
#    =0.103.0. This passed both tag workflows before 2026-08-23 and passed release.yml's build
#    too, because path deps win locally — the pins only matter once cargo publishes, at which
#    point `=0.103.0` resolves against the LIVE REGISTRY and the tested build is not the
#    shipped one. The guard's own error message had always NAMED the pins without checking them.
sed -i 's/^version = "0.103.0"$/version = "0.104.0"/' Cargo.toml
git commit -qam "partial bump: version only, pins left behind"
git push -q origin HEAD:refs/heads/release/claim-v0.104.0
if out=$("$guard" v0.104.0 Cargo.toml origin 2>&1); then
  fail "arm 6: guard accepted a partial bump (version 0.104.0, pins =0.103.0): $out"
fi
echo "$out" | grep -q 'memra-engine(=0.103.0)' \
  || fail "arm 6: refusal does not name the stale pin: $out"
echo "$out" | grep -q 'live registry' \
  || fail "arm 6: refusal does not explain the registry-resolution consequence: $out"

# 7. Pins bumped to match -> the same tag now passes (proves arm 6 refused the PIN, not the tag).
sed -i 's/version = "=0.103.0"/version = "=0.104.0"/g' Cargo.toml
git commit -qam "finish the bump: pins too"
out=$("$guard" v0.104.0 Cargo.toml origin 2>&1) \
  || fail "arm 7: guard refused a fully bumped, claimed v0.104.0: $out"
echo "$out" | grep -q '2 internal pins match' \
  || fail "arm 7: OK line does not report the pin census: $out"

# 8. VACUOUS PASS: a manifest with no pins at all must refuse rather than report success over
#    an empty set — the failure mode a census is most likely to develop silently.
grep -v 'version = "=0.104.0"' Cargo.toml > nopins.toml
if out=$("$guard" v0.104.0 nopins.toml origin 2>&1); then
  fail "arm 8: guard accepted a manifest with no pins (vacuous pass): $out"
fi
echo "$out" | grep -q 'vacuously' || fail "arm 8: wrong refusal for the no-pin case: $out"

echo "release-guard fixture: 8 arms PASS"
