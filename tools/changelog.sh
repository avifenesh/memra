#!/bin/bash
# Draft release notes from conventional-commit history: changelog.sh [FROM_TAG] [TO_REF]
# Defaults: FROM = previous tag, TO = HEAD. Groups by prefix; data()/chore() and merge
# commits are dropped (research log rows and plumbing are not user-facing changes). The draft
# is capped under GitHub's release-notes limit (CHANGELOG_MAX_BYTES, see the cap below).
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

collect() { # collect <grep-prefix-regex>: the section's bullets, one per line, no trailing newline
  git log --no-merges --format='- %s' "$FROM..$TO" | grep -E "^- $1" \
    | sed -E "s/^- $1(\([^)]*\))?!?: /- /" || true   # an empty section is not an error (set -e)
}
blen() { local LC_ALL=C; REPLY=${#1}; }  # bytes, not characters: a byte bound also bounds characters

titles=(Performance Features Fixes Configuration Documentation Other)
bodies=("$(collect perf)" "$(collect feat)" "$(collect fix)" "$(collect config)" "$(collect docs)"
        # anything not matching a known prefix (and not data/chore) lands under Other
        "$(git log --no-merges --format='- %s' "$FROM..$TO" \
           | grep -vE '^- (perf|feat|fix|config|docs|data|chore|wip|probe)(\([^)]*\))?!?:' || true)")

if [ -n "$baseline_tag" ]; then
    header="Changes since ${baseline_tag} (the last release on the old history; this history starts at snapshot ${FROM:0:9} = old ${baseline_old}):"$'\n\n'
else
    header="Changes since ${FROM}:"$'\n\n'
fi
footer="Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full experiment log in research/tune-data/"$'\n'

archive=""; archive_short=""
if [ -n "$baseline_tag" ]; then
    notes_path=$(dirname "$0")/../$baseline_notes
    if [ -f "$notes_path" ]; then
        # The archive file's own preamble ends at its "Changes since" line; emit the sections
        # that follow it, demoted one heading level so they nest under this one. EVERY
        # "Changes since" line is dropped, not only the first: a duplicated header in the
        # archive leaked into the v0.124.0 draft (revuto finding on PR #63). The trailing x keeps
        # the block's final newlines through the command substitution.
        archive=$(printf '## Before the snapshot: %s to %s (old history)\n' "$baseline_tag" "$baseline_old"
                  awk 'found && /^<!-- bookkeeping -->/ { exit } /^Changes since / { found = 1; next } found { print }' "$notes_path" | sed -E 's/^## /### /'
                  printf 'Full pre-snapshot record: %s\n\nx' "$baseline_notes")
        archive=${archive%x}
        archive_short=$(printf '## Before the snapshot: %s to %s (old history)\nNot listed here. Full pre-snapshot record: %s\n\nx' "$baseline_tag" "$baseline_old" "$baseline_notes")
        archive_short=${archive_short%x}
    else
        archive=$(printf '## Before the snapshot: %s to %s (old history)\nRECORD MISSING: %s is named by tools/changelog-baseline.txt but is not in the tree.\n\nx' "$baseline_tag" "$baseline_old" "$baseline_notes")
        archive=${archive%x}; archive_short=$archive
    fi
fi

out=$header
for i in "${!titles[@]}"; do
    [ -n "${bodies[$i]}" ] && out+="## ${titles[$i]}"$'\n'"${bodies[$i]}"$'\n\n'
done
out+=$archive$footer

# GitHub refuses release notes over 125,000 characters, and `gh release create` then fails the
# whole release step: the v0.139.0 draft was 328,884 for 3,866 commits. The draft is capped at
# CHANGELOG_MAX_BYTES (default 120,000 bytes, so at most 120,000 characters), which leaves room
# for the headline and context an owner adds on GitHub afterwards (docs/RELEASING.md). A draft
# that fits is emitted unchanged. One that does not keeps whole sections in priority order
# (configuration, performance, features, fixes, documentation, other: defaults and speed first,
# the research-log tail last), lists as many bullets of the first section that does not fit as
# the budget allows, names the count of every bullet it leaves out, and links the full range.
cap=${CHANGELOG_MAX_BYTES:-120000}
blen "$out"
if [ "$REPLY" -gt "$cap" ]; then
    to_ref=$(git tag --points-at "$TO" --sort=-v:refname 2>/dev/null | head -1)
    [ -n "$to_ref" ] || to_ref=$(git rev-parse "$TO")
    compare="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-avifenesh/memra}/compare/${FROM}...${to_ref}"
    total=0; declare -a count
    for i in "${!titles[@]}"; do
        if [ -n "${bodies[$i]}" ]; then count[$i]=$(printf '%s\n' "${bodies[$i]}" | wc -l); else count[$i]=0; fi
        total=$((total + count[i]))
    done
    notice() { # notice <listed>: sets NOTICE
        printf -v NOTICE 'These notes are cut to fit the GitHub release limit of 125,000 characters: %s of %s entries are listed, whole sections in priority order (configuration, performance, features, fixes, documentation, other). Every commit: %s or `git log --no-merges %s..%s`.\n\n' \
            "$1" "$total" "$compare" "$FROM" "$to_ref"
    }
    omitted() { printf -v OMIT '## %s\n- %s entries not listed (full list linked above)\n\n' "$1" "$2"; }
    # Reserve the smallest form of everything (the notice sized for its largest count), then
    # upgrade sections in priority order while the budget lasts.
    notice "$total"; blen "$header$NOTICE$footer"; used=$REPLY
    declare -a state small full
    for i in "${!titles[@]}"; do
        state[$i]=empty; small[$i]=0; full[$i]=0
        [ "${count[$i]}" -gt 0 ] || continue
        state[$i]=omitted
        omitted "${titles[$i]}" "${count[$i]}"; blen "$OMIT"; small[$i]=$REPLY
        blen "## ${titles[$i]}"$'\n'"${bodies[$i]}"$'\n\n'; full[$i]=$REPLY
        used=$((used + small[i]))
    done
    if [ -n "$archive" ]; then blen "$archive_short"; used=$((used + REPLY)); fi
    left=$((cap - used))
    [ "$left" -ge 0 ] || { echo "changelog: the section headings alone exceed $cap bytes" >&2; exit 1; }
    partial=-1; keep=0; archive_out=$archive_short
    for i in 3 0 1 2 4 5; do
        [ "${state[$i]}" = omitted ] || continue
        if [ $((full[i] - small[i])) -le "$left" ]; then
            state[$i]=full; left=$((left - full[i] + small[i])); continue
        fi
        # The first section that does not fit lists what fits; every later one is omitted.
        blen "## ${titles[$i]}"$'\n'"- ${count[$i]} more not listed (full list linked above)"$'\n\n'
        budget=$((left + small[i] - REPLY))
        while IFS= read -r line; do
            blen "$line"
            [ $((REPLY + 1)) -le "$budget" ] || break
            budget=$((budget - REPLY - 1)); keep=$((keep + 1))
        done <<< "${bodies[$i]}"
        [ "$keep" -gt 0 ] && { partial=$i; state[$i]=partial; }
        break
    done
    if [ "$partial" -lt 0 ] && [ -n "$archive" ]; then
        blen "$archive"; a=$REPLY; blen "$archive_short"
        [ $((a - REPLY)) -le "$left" ] && archive_out=$archive
    fi
    listed=0; body=""
    for i in "${!titles[@]}"; do
        case "${state[$i]}" in
            full) body+="## ${titles[$i]}"$'\n'"${bodies[$i]}"$'\n\n'; listed=$((listed + count[i])) ;;
            partial) body+="## ${titles[$i]}"$'\n'"$(printf '%s\n' "${bodies[$i]}" | head -n "$keep")"$'\n'
                     body+="- $((count[i] - keep)) more not listed (full list linked above)"$'\n\n'
                     listed=$((listed + keep)) ;;
            omitted) omitted "${titles[$i]}" "${count[$i]}"; body+=$OMIT ;;
        esac
    done
    notice "$listed"
    out=$header$NOTICE$body$archive_out$footer
    blen "$out"
    [ "$REPLY" -le "$cap" ] || { echo "changelog: capped draft is still $REPLY bytes (cap $cap)" >&2; exit 1; }
fi
printf '%s' "$out"
