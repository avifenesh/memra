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

CDPATH=
repo_root=$(cd "$(dirname "$0")/.." && pwd -P) || exit 2
# Git hooks may export repository-local context; fixtures must never inherit it.
git_local_env=$(git rev-parse --local-env-vars) || exit 2
for name in $git_local_env; do unset "$name"; done
# MEMRA_PUSH_RANGE points the fixture at an alternative implementation. It exists so the
# fixture's DECISIVENESS can be demonstrated: point it at a script that restores the old
# `git rev-parse @{u} || echo HEAD~20` behaviour and arms 1, 3, 4 and 5 must go red. A
# fixture nobody has ever seen fail is not evidence.
subject=${MEMRA_PUSH_RANGE:-$repo_root/tools/push-range.sh}
hook="$repo_root/tools/hooks/pre-push"

# Engine paths are sample data for fork-point arithmetic, not the current hardware
# applicability rule. Full dependency selection is tested by test_check_hardware_gate.py
# and test_release_qualification.py; this helper remains in the additional Step hook arm.
ENGINE_RE='^crates/memra-engine/(cu/|src/.*\.rs$)'

tmp_base=$(cd -- "${TMPDIR:-/tmp}" && pwd -P) || exit 2
tmp=$(mktemp -d "$tmp_base/memra-push-range-test.XXXXXX") || exit 2
case "$tmp" in "$tmp_base"/memra-push-range-test.*) ;; *) exit 2 ;; esac
[ -d "$tmp" ] && [ ! -L "$tmp" ] || exit 2
tmp=$(cd -- "$tmp" && pwd -P) || exit 2
[ "$(dirname -- "$tmp")" = "$tmp_base" ] && [ "$tmp" != "$repo_root" ] || exit 2
owner_file="$tmp/.fixture-owner"
printf '%s\n' "$$" > "$owner_file" || exit 2

owned_root() {
    [ -d "$tmp" ] && [ ! -L "$tmp" ] &&
        [ -f "$owner_file" ] && [ ! -L "$owner_file" ] &&
        [ "$(cat "$owner_file")" = "$$" ] &&
        [ "$(cd -- "$tmp" && pwd -P)" = "$tmp" ]
}
cleanup() (
    owned_root || { echo "push-range fixture: refusing unowned cleanup" >&2; exit 2; }
    cd -- "$tmp" || exit 2
    [ "$(pwd -P)" = "$tmp" ] || exit 2
    rm -rf -- "$tmp"
)
trap 'fixture_exit=$?; cleanup || exit 2; exit "$fixture_exit"' EXIT
trap 'exit 2' HUP INT TERM

fixture_cwd() {
    owned_root || return 2
    case "$1" in "$tmp"/*) ;; *) return 2 ;; esac
    [ -d "$1/.git" ] && [ ! -L "$1" ] && [ ! -L "$1/.git" ] &&
        [ "$(pwd -P)" = "$1" ] &&
        [ "$(git rev-parse --show-toplevel)" = "$1" ] &&
        [ "$(git rev-parse --absolute-git-dir)" = "$1/.git" ]
}
enter_fixture() {
    cd -- "$1" || return 2
    fixture_cwd "$1"
}

# Verdicts go to a FILE, not to shell variables. Every arm below runs in a ( subshell ) so it
# can cd freely, and a counter incremented in a subshell is discarded by the parent — the
# first draft of this fixture reported "1 passed / 0 failed" while an arm was visibly FAILing
# on the same screen. That is the exact shape (an accounting path that cannot report failure)
# this file exists to catch elsewhere, so it is worth the file.
results="$tmp/results"
: > "$results" || exit 2

ok()   { printf 'ok   %s\n' "$1"; printf 'ok\n' >> "$results"; }
bad()  { printf 'FAIL %s\n     %s\n' "$1" "${2:-}"; printf 'FAIL\n' >> "$results"; }
check(){ # check <name> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "expected [$2] got [$3]"; fi
}

git_q() { git -c advice.detachedHead=false -c init.defaultBranch=main "$@"; }

# Builds an upstream repo with N commits, one of which touches an engine file, then a clone.
# Fail before returning a path if scratch setup fails: continuing after a failed cd
# would run the unrelated-history cleanup in the caller checkout.
# $1 = dir name
make_pair() {
    case "$1" in ''|*[!a-z0-9-]*) return 2 ;; esac
    owned_root || return 2
    local up="$tmp/$1-remote" wt="$tmp/$1"
    git_q init -q --bare "$up" || return 2
    git_q init -q "$wt" || return 2
    (
        enter_fixture "$wt" || exit 2
        git config user.email t@example.invalid || exit 2
        git config user.name test || exit 2
        git config core.hooksPath /dev/null || exit 2
        mkdir -p crates/memra-engine/src crates/memra-engine/cu crates/memra-gguf/src docs || exit 2
        # Base history DEEPER THAN 20 commits, so `HEAD~20` resolves and the old fallback's
        # false positive is reachable on this fixture. Engine files that belong to "other
        # lanes" sit inside that window, which is the whole point.
        for i in $(seq 1 24); do
            if [ $((i % 4)) -eq 0 ]; then
                echo "other lane $i" > "crates/memra-engine/src/other$i.rs" || exit 2
            else
                echo "other lane $i" > "docs/other$i.md" || exit 2
            fi
            git add -A && git commit -q -m "other lane commit $i" || exit 2
        done
        echo k > crates/memra-engine/cu/other_kernel.cu || exit 2
        git add -A && git commit -q -m "other lane kernel" || exit 2
        git remote add origin "$up" || exit 2
        git push -q origin main || exit 2
    ) || return 2
    printf '%s\n' "$wt"
}

engine_files_for() { # engine_files_for <base>  (run inside a repo)
    git diff --name-only "$1"..HEAD | grep -E "$ENGINE_RE" || true
}

# ---------------------------------------------------------------- arm 0: caller is live
if grep -qE '^[[:space:]]*range=\$\(tools/push-range\.sh' "$hook"; then
    ok "pre-push invokes the tested fork-point helper for the additional Step gate"
else
    bad "push-range helper has no live pre-push caller" \
        "expected the additional Step gate to invoke tools/push-range.sh"
fi

# ------------------------------------------------- arm 1: FIRST push, no upstream, no engine
# The reported false positive. Branch adds one non-engine file; the old HEAD~20 fallback
# swept the base history's engine files in.
wt=$(make_pair first-push-clean) || exit 2
(
    enter_fixture "$wt" || exit 2
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
    check "arm1: fork-point diff contains 0 unrelated engine samples" "0" "$n_new"

    # Negative control on the same fixture: the old fallback is loud here.
    n_old=$(engine_files_for "HEAD~20" 2>/dev/null | grep -c . || true)
    if [ "$n_old" -gt 0 ]; then
        ok "arm1: old HEAD~20 fallback would have indicted $n_old unrelated engine files"
    else
        bad "arm1: negative control did not reproduce the false positive" \
            "HEAD~20 named no engine files; fixture is not representative"
    fi
) || exit 2

# ------------------------------------------- arm 2: FIRST push that DOES touch an engine file
# Teeth. A gate that stops firing is worse than a false positive.
wt=$(make_pair first-push-engine) || exit 2
(
    enter_fixture "$wt" || exit 2
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
) || exit 2

# ------------------------------------- arm 3: upstream SET and DIVERGED -> two-dot tree diff
# `git diff <upstream>..HEAD` is a tree comparison, so an upstream that moved on has its own
# files appear as "changed by this branch". merge-base removes that without going quiet.
wt=$(make_pair diverged) || exit 2
(
    enter_fixture "$wt" || exit 2
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
) || exit 2

# --------------------------------- arm 4: no upstream AND no remote default branch -> refuse
wt=$(make_pair no-base-ref) || exit 2
(
    enter_fixture "$wt" || exit 2
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
) || exit 2

# ------------------------------------------- arm 5: unrelated histories -> refuse, distinctly
wt=$(make_pair unrelated) || exit 2
(
    enter_fixture "$wt" || exit 2
    # A branch with no common ancestor at all.
    git checkout -q --orphan lane/orphan-root || exit 2
    fixture_cwd "$wt" || exit 2
    git rm -rq --cached . || exit 2
    rm -rf -- crates docs || exit 2
    mkdir -p docs || exit 2
    echo x > docs/x.md || exit 2
    git add -A && git commit -q -m "orphan root"

    out=$(bash "$subject" origin 2>&1); rc=$?
    check "arm5: refuses on unrelated histories" "1" "$rc"
    case "$out" in
        *"no merge-base"*) ok "arm5: names the real cause (no merge-base), not a stale fetch" ;;
        *) bad "arm5: wrong diagnosis" "$out" ;;
    esac
) || exit 2

# ------------------------------------------------ arm 6: remote arg that is a URL, not a name
wt=$(make_pair url-remote) || exit 2
(
    enter_fixture "$wt" || exit 2
    git checkout -q --no-track -b lane/url
    echo doc > docs/NOTE.md
    git add -A && git commit -q -m docs
    base=$(bash "$subject" "$tmp/url-remote-remote" 2>/dev/null)
    check "arm6: a URL in \$1 falls back to origin, not to a guess" \
        "$(git rev-parse origin/main)" "$base"
) || exit 2

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
