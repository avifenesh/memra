#!/usr/bin/env bash
# Teeth for tools/changelog.sh: the tag walk, the skip list, the explicit FROM, and the
# history-root baseline (2026-09-02), each forced in a throwaway repo under mktemp.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pass=0
ok()  { pass=$((pass+1)); echo "ok   $*"; }
bad() { echo "FAIL $*" >&2; exit 1; }

# A copy of the script with its side files, so the arms can rewrite the baseline and skip
# list without touching the real ones.
mkdir -p "$tmp/tools" "$tmp/docs/archive"
cp "$here/tools/changelog.sh" "$tmp/tools/"
: > "$tmp/tools/changelog-skip-tags.txt"
cl=$tmp/tools/changelog.sh

repo=$tmp/repo; git init -q "$repo"
g() { git -C "$repo" "$@"; }
g config user.email t@t; g config user.name t
c() { echo "$RANDOM" >> "$repo/f"; g add f; g commit -q -m "$1"; }
c "chore: snapshot root"
root=$(g rev-parse HEAD)
c "feat: first new thing"
c "fix: second new thing"

# arm 1: no tag, no baseline row -> plain root header, no archive section
: > "$tmp/tools/changelog-baseline.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q "^Changes since $root:" || bad "arm1: header should name the root: $out"
printf '%s\n' "$out" | grep -q 'Before the snapshot' && bad "arm1: archive section without a baseline row"
printf '%s\n' "$out" | grep -q '^- first new thing' || bad "arm1: feat missing"
ok "arm1 tagless, no baseline: root header"

# arm 2: no tag, baseline row for THIS root -> predecessor header + archive appended, demoted
cat > "$tmp/docs/archive/PRE.md" <<ARCH
# archive preamble that must NOT be echoed
Changes since v0.123.0:
Changes since v0.123.0:

## Fixes
- old fix that shipped untagged
ARCH
printf '# comment\n%s\tv0.123.0\t469c8898e\tdocs/archive/PRE.md\n' "$root" > "$tmp/tools/changelog-baseline.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q '^Changes since v0.123.0 (the last release on the old history' || bad "arm2: header should name the predecessor tag: $out"
printf '%s\n' "$out" | grep -q '^## Before the snapshot: v0.123.0 to 469c8898e' || bad "arm2: archive section missing: $out"
printf '%s\n' "$out" | grep -q '^### Fixes' || bad "arm2: archive sections must be demoted one level: $out"
printf '%s\n' "$out" | grep -q '^- old fix that shipped untagged' || bad "arm2: archive bullets missing"
printf '%s\n' "$out" | grep -q 'archive preamble' && bad "arm2: the archive preamble leaked into the draft"
[ "$(printf '%s\n' "$out" | grep -c '^Changes since ')" -eq 1 ] || bad "arm2: a 'Changes since' line from the archive leaked into the draft (revuto, PR #63): $out"
printf '%s\n' "$out" | grep -q '^- first new thing' || bad "arm2: new-history commits missing"
ok "arm2 tagless, baseline row: predecessor header + archive"

# arm 3: baseline row for a DIFFERENT root is ignored
printf '%s\tv0.1.0\tdeadbeef\tdocs/archive/PRE.md\n' 0000000000000000000000000000000000000000 > "$tmp/tools/changelog-baseline.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q "^Changes since $root:" || bad "arm3: foreign root row must be ignored: $out"
ok "arm3 foreign baseline row ignored"

# arm 4: a reachable tag wins over the baseline
printf '%s\tv0.123.0\t469c8898e\tdocs/archive/PRE.md\n' "$root" > "$tmp/tools/changelog-baseline.txt"
g tag -a v9.0.0 -m t "$(g rev-parse HEAD~1)"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q '^Changes since v9.0.0:' || bad "arm4: tag should win: $out"
printf '%s\n' "$out" | grep -q 'Before the snapshot' && bad "arm4: archive appended although a tag was reachable"
printf '%s\n' "$out" | grep -q '^- second new thing' || bad "arm4: commit after the tag missing"
printf '%s\n' "$out" | grep -q '^- first new thing' && bad "arm4: commit AT the tag must not be listed"
ok "arm4 reachable tag wins"

# arm 5: a skipped tag is walked past (existing behaviour, kept)
echo v9.0.0 > "$tmp/tools/changelog-skip-tags.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q '^Changes since v0.123.0' || bad "arm5: skipped tag should fall through to the baseline: $out"
ok "arm5 skip list honoured, falls through to baseline"

# arm 6: explicit FROM bypasses everything
out=$(cd "$repo" && "$cl" "$root")
printf '%s\n' "$out" | grep -q "^Changes since $root:" || bad "arm6: explicit FROM must win: $out"
printf '%s\n' "$out" | grep -q 'Before the snapshot' && bad "arm6: explicit FROM must not append the archive"
ok "arm6 explicit FROM wins"

# arm 7: baseline names a missing archive -> loud RECORD MISSING, never silent
: > "$tmp/tools/changelog-skip-tags.txt"; g tag -d v9.0.0 >/dev/null
printf '%s\tv0.123.0\t469c8898e\tdocs/archive/NOPE.md\n' "$root" > "$tmp/tools/changelog-baseline.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q 'RECORD MISSING: docs/archive/NOPE.md' || bad "arm7: missing archive must be loud: $out"
ok "arm7 missing archive is loud"

# arm 8: the REAL baseline row points at the REAL root of this history and a tracked file
# (a shallow checkout, the ci.yml default, cannot see the real root: the row's shape and its
# archive file are checked there, and the root identity is checked wherever history is full)
if [ "$(git -C "$here" rev-parse --is-shallow-repository)" = "true" ]; then
    row=$(grep -v '^#' "$here/tools/changelog-baseline.txt" | grep -E '^[0-9a-f]{40}\s' | head -1 || true)
    [ -n "$row" ] || bad "arm8: tools/changelog-baseline.txt has no full-sha row"
    echo "note arm8: shallow checkout, root identity not verifiable here (release.yml uses fetch-depth 0)"
else
    real_root=$(git -C "$here" rev-list --max-parents=0 HEAD | tail -1)
    row=$(grep -v '^#' "$here/tools/changelog-baseline.txt" | grep "^$real_root" || true)
    [ -n "$row" ] || bad "arm8: tools/changelog-baseline.txt has no row for this history's root $real_root"
fi
notes=$(printf '%s\n' "$row" | cut -f4)
[ -f "$here/$notes" ] || bad "arm8: baseline names $notes, which is not in the tree"
git -C "$here" ls-files --error-unmatch "$notes" >/dev/null 2>&1 || bad "arm8: $notes is not tracked"
ok "arm8 real baseline row resolves ($notes)"

# arm 9: emission stops at the bookkeeping marker (rig timing rows stay in the archive, never
# on a release page)
cat > "$tmp/docs/archive/PRE.md" <<ARCH
preamble
Changes since v0.123.0:

## Fixes
- public fix

<!-- bookkeeping -->
## Dropped as lane bookkeeping
- perf-ci row: 138.22 tok/s on the rig
ARCH
printf '%s\tv0.123.0\t469c8898e\tdocs/archive/PRE.md\n' "$root" > "$tmp/tools/changelog-baseline.txt"
out=$(cd "$repo" && "$cl")
printf '%s\n' "$out" | grep -q '^- public fix' || bad "arm9: public bullet missing: $out"
printf '%s\n' "$out" | grep -q 'tok/s' && bad "arm9: bookkeeping leaked past the marker: $out"
printf '%s\n' "$out" | grep -q 'Dropped as lane bookkeeping' && bad "arm9: bookkeeping heading leaked"
ok "arm9 emission stops at the bookkeeping marker"

# arm 10: a shallow clone says so out loud and does not apply the baseline (no silent loss)
printf '%s\tv0.123.0\t469c8898e\tdocs/archive/PRE.md\n' "$root" > "$tmp/tools/changelog-baseline.txt"
git clone -q --depth 1 "file://$repo" "$tmp/shallow"
out=$(cd "$tmp/shallow" && "$cl" 2>"$tmp/shallow.err")
grep -q 'shallow checkout' "$tmp/shallow.err" || bad "arm10: no shallow NOTE on stderr: $(cat "$tmp/shallow.err")"
printf '%s\n' "$out" | grep -q 'Before the snapshot' && bad "arm10: baseline applied in a shallow clone"
ok "arm10 shallow clone announces itself, baseline not applied"

echo "test_changelog: $pass arms PASS"
