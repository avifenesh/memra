#!/usr/bin/env bash
# Fixture for tools/push-range.sh — the pre-push gates' base-commit derivation.
#
# The defect this exists to prevent (2026-08-19): the derivation was one inline line in
# tools/hooks/pre-push with an arbitrary `|| echo HEAD~20` fallback, and there was no way to
# test it without a real remote, so nobody ever did. It false-positived on every branch's
# FIRST push by sweeping ~10 unrelated engine files into the diff, and lanes learned to push
# with MEMRA_SKIP_PERF_CI=1. Every arm below therefore asserts in BOTH directions: the base
# is right AND the wrong answer is proven wrong on the same fixture.
#
# No network, no GPU, no models. Builds throwaway repos under a mktemp dir and removes them.
#   bash tools/test_push_range.sh

set -uo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
# MEMRA_PUSH_RANGE points the fixture at an alternative implementation. It exists so the
# fixture's DECISIVENESS can be demonstrated: point it at a script that restores the old
# `git rev-parse @{u} || echo HEAD~20` behaviour and arms 1, 3, 4 and 5 must go red. A
# fixture nobody has ever seen fail is not evidence.
subject=${MEMRA_PUSH_RANGE:-$repo_root/tools/push-range.sh}
hook="$repo_root/tools/hooks/pre-push"

# The pre-push hook's engine-file trigger. Asserted below to be byte-present in the hook so
# this fixture cannot drift away from the pattern it claims to exercise.
ENGINE_RE='^crates/memra-engine/(cu/|src/.*\.rs$)'

tmp=$(mktemp -d "${TMPDIR:-/tmp}/memra-push-range-test.XXXXXX")
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT HUP INT TERM

# Verdicts go to a FILE, not to shell variables. Every arm below runs in a ( subshell ) so it
# can cd freely, and a counter incremented in a subshell is discarded by the parent — the
# first draft of this fixture reported "1 passed / 0 failed" while an arm was visibly FAILing
# on the same screen. That is the exact shape (an accounting path that cannot report failure)
# this file exists to catch elsewhere, so it is worth the file.
results="$tmp/results"
: > "$results"

ok()   { printf 'ok   %s\n' "$1"; printf 'ok\n' >> "$results"; }
bad()  { printf 'FAIL %s\n     %s\n' "$1" "${2:-}"; printf 'FAIL\n' >> "$results"; }
check(){ # check <name> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "expected [$2] got [$3]"; fi
}

git_q() { git -c advice.detachedHead=false -c init.defaultBranch=main "$@"; }

# Builds an upstream repo with N commits, one of which touches an engine file, then a clone.
# $1 = dir name
make_pair() {
    local up="$tmp/$1-remote" wt="$tmp/$1"
    git_q init -q --bare "$up"
    git_q init -q "$wt"
    (
        cd "$wt"
        git config user.email t@example.invalid
        git config user.name test
        mkdir -p crates/memra-engine/src crates/memra-engine/cu crates/memra-gguf/src docs
        # Base history DEEPER THAN 20 commits, so `HEAD~20` resolves and the old fallback's
        # false positive is reachable on this fixture. Engine files that belong to "other
        # lanes" sit inside that window, which is the whole point.
        for i in $(seq 1 24); do
            if [ $((i % 4)) -eq 0 ]; then
                echo "other lane $i" > "crates/memra-engine/src/other$i.rs"
            else
                echo "other lane $i" > "docs/other$i.md"
            fi
            git add -A && git commit -q -m "other lane commit $i"
        done
        echo k > crates/memra-engine/cu/other_kernel.cu
        git add -A && git commit -q -m "other lane kernel"
        git remote add origin "$up"
        git push -q origin main
    )
    printf '%s\n' "$wt"
}

engine_files_for() { # engine_files_for <base>  (run inside a repo)
    git diff --name-only "$1"..HEAD | grep -E "$ENGINE_RE" || true
}

# ---------------------------------------------------------------- arm 0: pattern is real
if grep -qF "$ENGINE_RE" "$hook"; then
    ok "engine-file trigger in this fixture is byte-present in tools/hooks/pre-push"
else
    bad "engine-file trigger drifted from tools/hooks/pre-push" \
        "pattern not found in hook: $ENGINE_RE"
fi

# ------------------------------------------------- arm 1: FIRST push, no upstream, no engine
# The reported false positive. Branch adds one non-engine file; the old HEAD~20 fallback
# swept the base history's engine files in.
wt=$(make_pair first-push-clean)
(
    cd "$wt"
    git checkout -q --no-track -b lane/docs-only
    echo "doc" > docs/NOTE.md
    echo "cfg" > crates/memra-gguf/src/config.rs   # crates/, but NOT an engine file
    git add -A && git commit -q -m "docs + gguf config, zero engine files"

    upstream=$(git rev-parse --symbolic-full-name '@{u}' 2>/dev/null || echo NONE)
    check "arm1: branch really has no upstream" "NONE" "$upstream"

    base=$(bash "$subject" origin 2>/dev/null)
    expect=$(git rev-parse origin/main)
    check "arm1: base is the fork point (origin/main)" "$expect" "$base"

    n_new=$(engine_files_for "$base" | grep -c . || true)
    check "arm1: engine files under the fix = 0 (gate stays quiet)" "0" "$n_new"

    # Negative control on the same fixture: the old fallback is loud here.
    n_old=$(engine_files_for "HEAD~20" 2>/dev/null | grep -c . || true)
    if [ "$n_old" -gt 0 ]; then
        ok "arm1: old HEAD~20 fallback would have indicted $n_old unrelated engine files"
    else
        bad "arm1: negative control did not reproduce the false positive" \
            "HEAD~20 named no engine files; fixture is not representative"
    fi
)

# ------------------------------------------- arm 2: FIRST push that DOES touch an engine file
# Teeth. A gate that stops firing is worse than a false positive.
wt=$(make_pair first-push-engine)
(
    cd "$wt"
    git checkout -q --no-track -b lane/kernel
    echo "fn mine() {}" > crates/memra-engine/src/mine.rs
    git add -A && git commit -q -m "real engine change"

    base=$(bash "$subject" origin 2>/dev/null)
    check "arm2: base is the fork point" "$(git rev-parse origin/main)" "$base"
    check "arm2: the branch's own engine file IS reported" \
        "crates/memra-engine/src/mine.rs" "$(engine_files_for "$base")"

    # And a .cu file trips it too (the other half of the trigger regex).
    echo "// kernel" > crates/memra-engine/cu/mine.cu
    git add -A && git commit -q -m "real kernel change"
    got=$(engine_files_for "$base" | sort | tr '\n' ' ')
    check "arm2: cu/ half of the trigger also fires" \
        "crates/memra-engine/cu/mine.cu crates/memra-engine/src/mine.rs " "$got"
)

# ------------------------------------- arm 3: upstream SET and DIVERGED -> two-dot tree diff
# `git diff <upstream>..HEAD` is a tree comparison, so an upstream that moved on has its own
# files appear as "changed by this branch". merge-base removes that without going quiet.
wt=$(make_pair diverged)
(
    cd "$wt"
    git checkout -q -b lane/diverged --track origin/main
    fork=$(git rev-parse HEAD)
    echo "doc" > docs/NOTE.md
    git add -A && git commit -q -m "lane: docs only"
    # Upstream advances with somebody else's engine work.
    git checkout -q main
    echo "fn theirs() {}" > crates/memra-engine/src/theirs.rs
    git add -A && git commit -q -m "other lane advances main"
    git push -q origin main
    git checkout -q lane/diverged

    upstream=$(git rev-parse --symbolic-full-name '@{u}')
    check "arm3: upstream is set" "refs/remotes/origin/main" "$upstream"

    base=$(bash "$subject" origin 2>/dev/null)
    check "arm3: base is the merge-base, not the moved upstream tip" "$fork" "$base"

    n_two_dot=$(engine_files_for "origin/main" | grep -c . || true)
    n_fixed=$(engine_files_for "$base" | grep -c . || true)
    if [ "$n_two_dot" -gt 0 ]; then
        ok "arm3: raw upstream..HEAD tree diff falsely names $n_two_dot engine file(s)"
    else
        bad "arm3: negative control failed" "diverged upstream named no engine files"
    fi
    check "arm3: merge-base base names 0 engine files" "0" "$n_fixed"
)

# --------------------------------- arm 4: no upstream AND no remote default branch -> refuse
wt=$(make_pair no-base-ref)
(
    cd "$wt"
    git checkout -q --no-track -b lane/orphaned
    echo "doc" > docs/NOTE.md
    git add -A && git commit -q -m "docs"
    git update-ref -d refs/remotes/origin/main      # simulate a never-fetched default branch

    out=$(bash "$subject" origin 2>&1); rc=$?
    check "arm4: refuses instead of guessing" "1" "$rc"
    case "$out" in
        *"git fetch origin main"*) ok "arm4: message names the fetch that fixes it" ;;
        *) bad "arm4: message is not actionable" "$out" ;;
    esac
    case "$out" in
        *HEAD~*) bad "arm4: still mentions an arbitrary depth" "$out" ;;
        *) ok "arm4: no arbitrary-depth fallback in the failure path" ;;
    esac
)

# ------------------------------------------- arm 5: unrelated histories -> refuse, distinctly
wt=$(make_pair unrelated)
(
    cd "$wt"
    # A branch with no common ancestor at all.
    git checkout -q --orphan lane/orphan-root
    git rm -rq --cached . 2>/dev/null || true
    rm -rf crates docs
    mkdir -p docs && echo x > docs/x.md
    git add -A && git commit -q -m "orphan root"

    out=$(bash "$subject" origin 2>&1); rc=$?
    check "arm5: refuses on unrelated histories" "1" "$rc"
    case "$out" in
        *"no merge-base"*) ok "arm5: names the real cause (no merge-base), not a stale fetch" ;;
        *) bad "arm5: wrong diagnosis" "$out" ;;
    esac
)

# ------------------------------------------------ arm 6: remote arg that is a URL, not a name
wt=$(make_pair url-remote)
(
    cd "$wt"
    git checkout -q --no-track -b lane/url
    echo doc > docs/NOTE.md
    git add -A && git commit -q -m docs
    base=$(bash "$subject" "$tmp/url-remote-remote" 2>/dev/null)
    check "arm6: a URL in \$1 falls back to origin, not to a guess" \
        "$(git rev-parse origin/main)" "$base"
)

pass=$(grep -c '^ok$'   "$results" || true)
fail=$(grep -c '^FAIL$' "$results" || true)
total=$(grep -c . "$results" || true)
printf '\n%s passed / %s failed  (%s assertions recorded)\n' "$pass" "$fail" "$total"
# A run that recorded nothing is a broken fixture, not a green one.
if [ "$total" -lt 18 ]; then
    printf 'FAIL fixture recorded only %s assertions; expected >= 18 — arms did not run\n' \
        "$total"
    exit 1
fi
[ "$fail" -eq 0 ]
