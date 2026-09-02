#!/bin/bash
# Draft release notes from conventional-commit history: changelog.sh [FROM_TAG] [TO_REF]
# Defaults: FROM = previous tag, TO = HEAD. Groups by prefix; data()/chore() and merge
# commits are dropped (research log rows and plumbing are not user-facing changes).
set -euo pipefail
TO=${2:-HEAD}

# The baseline must be the last tag that actually BECAME A RELEASE. `git describe` returns the
# nearest tag, shipped or not, and on 2026-08-23 that produced a v0.106.0 draft headed "Changes
# since v0.105.0" covering 48 commits — while the last real release, v0.103.0, was 236 non-merge
# commits back. Four fifths of the release would have been missing from the public record.
# tools/changelog-skip-tags.txt names the tags that never shipped (git cannot know this), and
# this walk skips them. An explicit FROM argument always wins and is never filtered.
skip_file=$(dirname "$0")/changelog-skip-tags.txt
pick_from() {
    local candidate skiplist=""
    [ -f "$skip_file" ] && skiplist=$(grep -vE '^\s*(#|$)' "$skip_file" | tr -d '\r')
    # Tags reachable from TO^, newest first by version order.
    for candidate in $(git tag --merged "$TO^" --sort=-v:refname 2>/dev/null); do
        case "
$skiplist
" in
            *"
$candidate
"*) continue ;;
        esac
        printf '%s\n' "$candidate"
        return 0
    done
    git rev-list --max-parents=0 HEAD | tail -1
}
FROM=${1:-$(pick_from)}

# HISTORY ROOT BASELINE (2026-09-02). The fallback above returns the root commit when no
# release tag is reachable. On a history rebuilt from a content snapshot (this repo,
# 2026-09-01) the root is not the beginning: tools/changelog-baseline.txt says which release
# it stands past and where the pre-snapshot notes live. Without this, the first tag on the
# new history reads "Changes since <root>" and drops 108 shipped-but-untagged commits from the
# public record. An explicit FROM argument bypasses the baseline like it bypasses the skips.
baseline_file=$(dirname "$0")/changelog-baseline.txt
baseline_tag=""; baseline_old=""; baseline_notes=""
if [ -z "${1:-}" ] && [ -f "$baseline_file" ] && [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
    # A shallow checkout's "root" is the graft boundary, not the history root, so the baseline
    # cannot be matched and the pre-snapshot notes would silently vanish. Say so; release.yml
    # checks out with fetch-depth 0 exactly for this step.
    echo "changelog: NOTE — shallow checkout: the history root is unknown here, so the baseline in $baseline_file is NOT applied (fetch-depth 0 / --unshallow for release notes)" >&2
elif [ -z "${1:-}" ] && [ -f "$baseline_file" ]; then
    root=$(git rev-list --max-parents=0 "$TO" 2>/dev/null | tail -1)
    if [ -n "$root" ] && [ "$(git rev-parse "$FROM" 2>/dev/null)" = "$root" ]; then
        while IFS=$'\t' read -r b_root b_tag b_old b_notes; do
            case "$b_root" in ''|\#*) continue ;; esac
            if [ "$b_root" = "$root" ]; then
                baseline_tag=$b_tag; baseline_old=$b_old; baseline_notes=$b_notes
            fi
        done < "$baseline_file"
    fi
fi

section() { # section <title> <grep-prefix-regex>
  local body
  body=$(git log --no-merges --format='- %s' "$FROM..$TO" | grep -E "^- $2" \
         | sed -E "s/^- $2(\([^)]*\))?!?: /- /" || true)
  [ -n "$body" ] && printf '## %s\n%s\n\n' "$1" "$body"
  return 0   # an empty section is not an error (set -e)
}

if [ -n "$baseline_tag" ]; then
    echo "Changes since ${baseline_tag} (the last release on the old history; this history starts at snapshot ${FROM:0:9} = old ${baseline_old}):"
else
    echo "Changes since ${FROM}:"
fi
echo
section "Performance"    "perf"
section "Features"       "feat"
section "Fixes"          "fix"
section "Configuration"  "config"
section "Documentation"  "docs"
# anything not matching a known prefix (and not data/chore) lands under Other
other=$(git log --no-merges --format='- %s' "$FROM..$TO" \
        | grep -vE '^- (perf|feat|fix|config|docs|data|chore|wip|probe)(\([^)]*\))?!?:' || true)
[ -n "$other" ] && printf '## Other\n%s\n\n' "$other" || true

if [ -n "$baseline_tag" ]; then
    notes_path=$(dirname "$0")/../$baseline_notes
    if [ -f "$notes_path" ]; then
        printf '## Before the snapshot: %s to %s (old history)\n' "$baseline_tag" "$baseline_old"
        # The archive file's own preamble ends at its "Changes since" line; emit the sections
        # that follow it, demoted one heading level so they nest under this one. EVERY
        # "Changes since" line is dropped, not only the first: a duplicated header in the
        # archive leaked into the v0.124.0 draft (revuto finding on PR #63).
        awk 'found && /^<!-- bookkeeping -->/ { exit } /^Changes since / { found = 1; next } found { print }' "$notes_path" | sed -E 's/^## /### /'
        printf 'Full pre-snapshot record: %s\n\n' "$baseline_notes"
    else
        printf '## Before the snapshot: %s to %s (old history)\nRECORD MISSING: %s is named by tools/changelog-baseline.txt but is not in the tree.\n\n' "$baseline_tag" "$baseline_old" "$baseline_notes"
    fi
fi
echo "Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full experiment log in research/tune-data/"
