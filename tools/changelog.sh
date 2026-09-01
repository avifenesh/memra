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

section() { # section <title> <grep-prefix-regex>
  local body
  body=$(git log --no-merges --format='- %s' "$FROM..$TO" | grep -E "^- $2" \
         | sed -E "s/^- $2(\([^)]*\))?!?: /- /" || true)
  [ -n "$body" ] && printf '## %s\n%s\n\n' "$1" "$body"
  return 0   # an empty section is not an error (set -e)
}

echo "Changes since ${FROM}:"
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

echo "Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full experiment log in research/tune-data/"
