#!/usr/bin/env bash
# Release-tag guard: refuses a release tag BEFORE any asset is built or published.
#
# Born from the 2026-08-22/23 tag races: three parallel sessions pushed v0.102.0,
# v0.104.0 and v0.104.1 while the workspace Cargo.toml still said an older version.
# publish.yml's inline guard refused the crates (correctly), but release.yml had NO
# guard — it built and published GitHub releases whose binaries self-report a different
# version than the tag they hang on. Three renumbers and one near-bad deploy in a week.
# Those three releases are annotated "[SKIPPED — version mismatch]".
#
# RECORD CORRECTED 2026-08-23: v0.105.0 was originally filed with those three, and that
# was wrong. Its Cargo mismatch was real but incidental — its release run (32581590491)
# died on `rust-lld: error: undefined symbol` in the sm_89 matrix cell, because
# cu/mmq_fp8_blk_stub.cu had lost ABI parity with cu/mmq_fp8_blk.cu at 58ce746ad3. A
# perfectly bumped v0.105.0 would have failed identically, and v0.106.0 proved it by
# doing exactly that with Cargo and tag in agreement. Blaming version discipline stopped
# anyone looking for a day; a wrong postmortem is worse than none. The gate for THAT
# class is tools/stub-abi-census.py plus ci.yml's arch-coverage job — this guard
# never could have caught it, and does not claim to.
#
# Three refusals, all fail-closed:
#   1. workspace version:  [workspace.package].version in the root Cargo.toml must equal
#      the tag with the 'v' stripped. Parsed with awk, not cargo metadata, so the same
#      file runs in the CPU-only CI fixture (tools/test_release_guard.sh) without a
#      CUDA toolchain.
#   2. internal pins:      every [workspace.dependencies] memra-* `version = "=X.Y.Z"` must
#      equal that same version, and there must be at least one (a census over an empty set
#      is not a check). Added 2026-08-23: this guard had always NAMED the pins in refusal 1's
#      error text without ever checking them, so a partial bump passed both tag workflows.
#   3. claim branch:       refs/heads/release/claim-<tag> must exist on the remote.
#      The claim is how parallel sessions serialize version numbers (docs/RELEASING.md
#      "Claiming a version number"): the FIRST push of that branch wins atomically on
#      origin; a session whose claim push is refused picks the next number instead of
#      racing the tag. A tag without a claim is by definition a race product — refuse it.
#
# Usage: tools/release-guard.sh <tag> [manifest] [remote]
#   defaults: manifest=Cargo.toml, remote=origin
# Callers: release.yml (guard job, gates the build matrix), publish.yml (tag runs),
#          tools/test_release_guard.sh (CI teeth — proves both refusals can fail).
set -euo pipefail

tag=${1:?usage: release-guard.sh <tag> [manifest] [remote]}
manifest=${2:-Cargo.toml}
remote=${3:-origin}

case "$tag" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *)
    echo "::error::release-guard: tag '$tag' is not vX.Y.Z"
    exit 1
    ;;
esac

ws_version=$(awk -F'"' '
  /^\[workspace\.package\]/ { s = 1; next }
  /^\[/                     { s = 0 }
  s && /^version[ =]/       { print $2; exit }
' "$manifest")

if [ -z "$ws_version" ]; then
  echo "::error::release-guard: no [workspace.package] version found in $manifest"
  exit 1
fi

if [ "v$ws_version" != "$tag" ]; then
  echo "::error::release-guard: workspace version v$ws_version != tag $tag — bump [workspace.package].version (+ the pinned [workspace.dependencies] versions) BEFORE tagging; this tag would ship $ws_version binaries under a $tag name (docs/RELEASING.md)"
  exit 1
fi

# The PINNED internal versions, which this guard used to name in its own error message
# (above) and then not check — it validated 1 of the 10 version fields the release ritual
# requires. A partial bump (workspace.package.version moved, the `=X.Y.Z` pins left behind)
# passed BOTH tag workflows and passed release.yml's build, because path dependencies win
# locally and cargo only consults the version when it publishes. It then either hard-fails
# `cargo publish --locked` after the 6-cell matrix has already spent its minutes, or — worse —
# resolves `=<old>` against the LIVE REGISTRY, so a green build was built against the previous
# release's crates. Latent, not discharged: every pin agrees today. Checked now so it stays
# that way.
bad_pins=$(awk -F'"' '
  /^\[workspace\.dependencies\]/ { s = 1; next }
  /^\[/                          { s = 0 }
  s && /^memra-/ && /version[ ]*=[ ]*"=/ {
    # field layout: ... version = "=X.Y.Z" }  -> the pinned version is the quoted field
    name = $1; sub(/[ \t].*$/, "", name)
    for (i = 1; i <= NF; i++) if ($i ~ /^=[0-9]/) { v = substr($i, 2); if (v != want) print name "(=" v ")" }
  }
' want="$ws_version" "$manifest")

if [ -n "$bad_pins" ]; then
  echo "::error::release-guard: [workspace.dependencies] pin(s) do not match [workspace.package].version $ws_version: $(echo $bad_pins) — bump the pins in the same sed pass as the version, or cargo publish resolves the stale requirement against the live registry and the build you tested is not the build you ship (docs/RELEASING.md)"
  exit 1
fi
n_pins=$(awk '/^\[workspace\.dependencies\]/{s=1;next} /^\[/{s=0} s && /^memra-/ && /version[ ]*=[ ]*"=/{n++} END{print n+0}' "$manifest")
if [ "$n_pins" -eq 0 ]; then
  echo "::error::release-guard: no pinned [workspace.dependencies] memra-* versions found in $manifest — the pin census would pass vacuously, so it refuses instead"
  exit 1
fi

claim="refs/heads/release/claim-$tag"
if ! git ls-remote --exit-code "$remote" "$claim" >/dev/null 2>&1; then
  echo "::error::release-guard: no claim branch release/claim-$tag on $remote — claim the number first (docs/RELEASING.md 'Claiming a version number'); an unclaimed tag is a session race by definition"
  exit 1
fi

echo "release-guard: $tag OK — workspace $ws_version matches, $n_pins internal pins match, claim branch present on $remote"
