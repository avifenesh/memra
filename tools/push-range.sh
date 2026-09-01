#!/bin/sh
# Prints the base commit of the current branch — the commit it FORKED FROM — for the
# pre-push gates to diff against. SHA on stdout, diagnostics on stderr, exit 1 with an
# actionable message when the fork point cannot be named.
#
# Usage: tools/push-range.sh [remote-name]        (git hands the pre-push hook $1 = remote)
#
# WHY THIS IS ITS OWN SCRIPT (2026-08-19)
# It used to be one line inside tools/hooks/pre-push:
#     range=$(git rev-parse '@{u}' 2>/dev/null || echo HEAD~20)
# `@{u}` is unset on a branch's FIRST push, so every fresh lane fell through to an
# arbitrary depth of 20 commits, which on a busy day is OTHER LANES' work. Measured on
# origin/main @ 43bd1afb84: HEAD~20..HEAD names 10 memra-engine files. On the live
# lane/gate-coverage-20260819 the fallback indicted 7 engine files where the branch adds 2.
# Lanes touching zero engine files got a red perf gate, verified by hand that it was
# spurious, and pushed with MEMRA_SKIP_PERF_CI=1 — at least four times in one day. A gate
# that is LOUDEST WHEN IT HAS LEAST INFORMATION trains everyone to override it, and the
# trained habit is worse damage than the false positive. So: no arbitrary fallback. Either
# we can name the fork point or we say what to run and stop.
#
# It lives in its own file so it can have a FIXTURE. In the hook it was a branch no test
# could reach without a real remote; tools/test_push_range.sh now drives all four arms
# (upstream present, first push, base ref absent, unrelated histories) in throwaway repos.

set -eu

push_remote=${1:-}

# MERGE-BASE, not the ref itself, in BOTH arms. `git diff A..HEAD` is a two-dot TREE diff:
# when the base ref is not an ancestor of HEAD (diverged upstream, or an upstream that is
# literally origin/main and has since moved on) everything the BASE changed also shows up
# as "changed by this branch" — the same false positive by another door. A merge-base is an
# ancestor of HEAD by construction, so the diff is exactly what this branch adds.
base_ref=$(git rev-parse --symbolic-full-name '@{u}' 2>/dev/null || true)

if [ -z "$base_ref" ]; then
    # No upstream: first push. Fork point vs the default branch on the remote being pushed
    # to. $1 may be a bare URL rather than a configured remote name.
    if [ -z "$push_remote" ] || ! git remote get-url "$push_remote" >/dev/null 2>&1; then
        push_remote=origin
    fi
    base_ref="refs/remotes/$push_remote/main"
    if ! git rev-parse --verify --quiet "$base_ref" >/dev/null; then
        echo "push-range: cannot tell what this branch adds — it has no upstream and" >&2
        echo "    $base_ref is not present in this checkout." >&2
        echo "Fix by giving the gate a base, not by skipping the gate:" >&2
        echo "    git fetch $push_remote main        # then push again" >&2
        echo "    (or push with -u so @{u} is set: git push -u $push_remote HEAD)" >&2
        exit 1
    fi
fi

base=$(git merge-base HEAD "$base_ref" 2>/dev/null || true)
if [ -z "$base" ]; then
    echo "push-range: no merge-base between HEAD and $base_ref, so the change set for" >&2
    echo "    this branch is undefined and the pre-push gates cannot be evaluated." >&2
    echo "    This means unrelated histories (a fresh init, a grafted or filtered" >&2
    echo "    checkout), not a stale fetch." >&2
    echo "Rebase this work onto the default branch before pushing." >&2
    exit 1
fi

printf '%s\n' "$base"
