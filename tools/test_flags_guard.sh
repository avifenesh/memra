#!/usr/bin/env bash
# Teeth for the pre-push MEMRA_* flags-census arm in tools/hooks/pre-push (2026-08-23).
#
# Why this fixture exists at all: the census (tools/check-flags.sh) was already correct and
# already had its own coverage fixture (tools/test_check_flags.sh, 19 arms). What was missing
# was a caller EARLY enough to matter — the only local one was tools/local-ci.sh, where it is a
# non-fatal WARNING behind a cargo build and a GPU. main went red three times on 2026-08-23
# (61e99d5337, d7fed25562, plus two lanes that inherited the red) because the first thing that
# actually refused was a CI job, i.e. after landing. The hook arm is the fix; this file is what
# keeps the hook arm from becoming another gate nobody has watched fail.
#
# It drives a REAL `git push` through the REAL tools/hooks/pre-push into a bare local origin, in
# a throwaway repo whose OTHER hook arms are stubbed green — so a refusal here can only be the
# census, and a success here proves the census did not merely fail to run. That is deliberately
# stronger than a grep for the call site: this is the same distinction
# tools/test_release_guard.sh draws with its wiring arm, and the same lesson
# GATE-INTEGRITY-20260819 A-18 wrote down (a gate with no exercised caller reads as coverage
# while providing none).
#
# Arms:
#   1. uncovered MEMRA_* read, committed      -> push REFUSED, message names the flag
#   2. refusal is the CENSUS, not a stub      -> the refusal text names this arm, not another
#   3. same read + a docs/FLAGS.md row        -> push SUCCEEDS
#   4. same read + a PREFIX row (MEMRA_X_*)   -> push SUCCEEDS
#   5. MEMRA_SKIP_FLAGS_CENSUS=1              -> push succeeds, skip PRINTED and LOGGED
#   6. rg absent                              -> WARNS and continues (condition self-checked)
#   7. wiring census                          -> the hook calls check-flags.sh, ci.yml still
#                                                runs the self-test backstop and this fixture
#   8. MEMRA_SKIP_PERF_CI=1 on an engine file -> perf gate skip PRINTED and LOGGED too
#   9. rewritten topic based on newer main    -> boundary scan excludes already-remote commits
#
# SCOPE, since the filename is narrower than the content: arms 1-4, 6 and 7 are the census arm;
# arms 5 and 8 cover the hook's ESCAPE HATCHES, which are a property of the hook rather than of
# any one gate in it. They live here because this is the only fixture that drives the real
# tools/hooks/pre-push end to end, and a near-duplicate file to hold two arms would be a second
# idiom for one fact. If a third skip env is ever added to that hook, its arm belongs here.
#
# CPU only: throwaway repos under mktemp, a bare repo standing in for origin, no network, no
# cargo, no GPU. Needs rg and python3, same as the census itself.
set -uo pipefail

here=$(cd -- "$(dirname -- "$0")/.." && pwd)
hook=$here/tools/hooks/pre-push
census=$here/tools/check-flags.sh
range=$here/tools/push-range.sh

for required in "$hook" "$census" "$range"; do
    [[ -f "$required" ]] || { echo "test_flags_guard: missing $required" >&2; exit 2; }
done
command -v rg >/dev/null || { echo "test_flags_guard: rg is required" >&2; exit 2; }

tmp=$(mktemp -d /tmp/flagsguard-test.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

pass=0
fail=0

ok()   { printf 'ok   %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf 'FAIL %s\n' "$1" >&2; fail=$((fail+1)); }

# The read the fixture plants. A MEMRA_FIXTURE_ prefix so a stray copy of this name can never
# collide with a real flag, and it goes in memra-tokenizer — NOT memra-engine — so the hook's
# engine-file hardware/perf block stays out of the way and cannot mask or mimic a census verdict.
FIXTURE_FLAG=MEMRA_FIXTURE_GUARD_ESCAPE
FIXTURE_READ="pub fn escape() -> bool {
    std::env::var(\"$FIXTURE_FLAG\").as_deref() == Ok(\"1\")
}"

stage() {
    # stage <name> -> sets the global $work to the work-repo path. A throwaway repo carrying the
    # REAL pre-push hook, the REAL census and the REAL push-range helper, with every other hook
    # arm stubbed green. The stubs are the point: a red push in this repo has exactly one
    # possible cause.
    #
    # It sets a GLOBAL rather than printing the path, because `work=$(stage arm1)` puts staging
    # inside a command substitution where `exit 2` kills only the substitution — the first draft
    # forgot `git remote add origin` and every arm then failed on "'origin' does not appear to be
    # a git repository" while staging reported nothing. A broken stage must abort the run, not
    # hand the arms a repo that cannot push.
    # Two statements, not one `local name=$1 root=$tmp/$name`: bash expands every word of a
    # `local` before performing any of its assignments, so $name is still unset in that form.
    local name=$1
    local root=$tmp/$name
    mkdir -p "$root"
    git init -q --bare "$root/origin.git"
    git init -q "$root/work"
    (
        cd "$root/work" || exit 1
        git config user.email flagsguard@test
        git config user.name flagsguard
        git remote add origin "$root/origin.git"
        mkdir -p tools/hooks docs crates/memra-tokenizer/src

        cp "$hook" tools/hooks/pre-push
        cp "$census" tools/check-flags.sh
        cp "$range" tools/push-range.sh
        chmod +x tools/hooks/pre-push tools/check-flags.sh tools/push-range.sh

        # Stubs for the hook's other arms. Python, because the hook invokes them as
        # `python3 tools/<name>.py` — a shell stub under that name would fail as a syntax error
        # and every arm below would "pass" for the wrong reason.
        printf 'import sys\nsys.exit(0)\n' > tools/update-perf-board.py
        printf 'import sys\nsys.exit(0)\n' > tools/check-public-boundary.py

        # The three releasability censuses (landed on main 2026-08-23 in 7f342b42b6). They are
        # deliberately skip-less and fail CLOSED on a missing script, so an unstubbed one refuses
        # every push in this fixture — which is how their arrival announced itself here within
        # minutes rather than silently. That is the cost of driving the REAL hook, and it is the
        # point: a stub list that has to be maintained is a fixture that notices new arms. These
        # are invoked directly (`"$census"`), so each stub's own shebang governs, and each needs
        # the executable bit the hook checks for.
        printf '#!/bin/sh\nexit 0\n' > tools/workspace-publish-census.sh
        printf '#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n' > tools/stub-abi-census.py
        printf '#!/bin/sh\nexit 0\n' > tools/arch-matrix-census.sh
        chmod +x tools/workspace-publish-census.sh tools/stub-abi-census.py \
            tools/arch-matrix-census.sh

        # docs/FLAGS.md only. The census's grandfather baseline was DELETED 2026-08-23 and its
        # absence is now asserted, so this used to create
        # `research/docsync3-20260811/flags-drift.txt` and creating it here would make the gate
        # refuse (rc=2) on every arm below — the fixture would report a census failure that is
        # really a re-grant refusal. Nothing else here needed the file: it was always empty.
        printf '# Flags\n\n' > docs/FLAGS.md
        # An empty crate src keeps the census non-vacuous-by-construction (it refuses outright
        # when no crates/*/src exists) while reading zero flags, so the seed push is green.
        : > crates/memra-tokenizer/src/lib.rs

        # Pathspec staging, never -A, even in a fixture repo.
        git add tools docs crates
        git commit -qm seed
        # Seed push runs with hooks UNCONFIGURED: it is staging, not the thing under test, and
        # push-range.sh correctly refuses a first push with no upstream and no origin/main.
        git push -q -u origin HEAD:main
        git config core.hooksPath tools/hooks
    ) || { echo "test_flags_guard: staging $name FAILED — aborting" >&2; exit 2; }
    work=$root/work
    # Prove the stage is pushable BEFORE an arm reads a push failure as a census verdict.
    ( cd "$work" && git ls-remote --exit-code origin >/dev/null 2>&1 ) \
        || { echo "test_flags_guard: staging $name produced no working origin — aborting" >&2
             exit 2; }
}

document() {
    # document <work> <row text> — append a FLAGS.md row and commit it.
    local work=$1 row=$2
    ( cd "$work" && printf '| `%s` | documented by the fixture |\n' "$row" >> docs/FLAGS.md \
        && git add docs/FLAGS.md && git commit -qm "flags row: $row" ) >/dev/null
}

plant_read() {
    # plant_read <work> — commit the uncovered MEMRA_* read.
    local work=$1
    ( cd "$work" && printf '%s\n' "$FIXTURE_READ" > crates/memra-tokenizer/src/lib.rs \
        && git add crates/memra-tokenizer/src/lib.rs \
        && git commit -qm "plant an undocumented $FIXTURE_FLAG read" ) >/dev/null
}

push() {
    # push <work> [env assignments...] -> prints combined output, returns push's exit code
    local work=$1; shift
    ( cd "$work" && env "$@" git push origin HEAD:main 2>&1 )
}

# ---------------------------------------------------------------------------
# Arms 1 + 2: an undocumented read must REFUSE the push, and the refusal must be the census.
# ---------------------------------------------------------------------------
stage arm1
plant_read "$work"
if out=$(push "$work"); then
    bad "arm1: push ACCEPTED an undocumented $FIXTURE_FLAG read
$out"
    bad "arm2: (not evaluated — arm1 accepted the push)"
else
    if [[ "$out" == *"$FIXTURE_FLAG"* ]]; then
        ok "arm1: push REFUSED and the message names $FIXTURE_FLAG"
    else
        bad "arm1: push refused but the message never names $FIXTURE_FLAG
$out"
    fi
    # The refusal has to be attributable. "some arm said no" is not evidence that THIS arm
    # works, and the perf-board / boundary stubs are precisely the confusable neighbours.
    if [[ "$out" == *"flags census is red"* && "$out" == *"docs/FLAGS.md"* ]]; then
        ok "arm2: the refusal is the flags census (names itself and docs/FLAGS.md)"
    else
        bad "arm2: refusal is not attributable to the flags census
$out"
    fi
fi

# ---------------------------------------------------------------------------
# Arm 3: the same read passes once the row is committed. Without this the arm above would be
# satisfied by a hook that refuses everything.
# ---------------------------------------------------------------------------
stage arm3
plant_read "$work"
document "$work" "$FIXTURE_FLAG"
if out=$(push "$work"); then
    ok "arm3: documented read PASSES the hook"
else
    bad "arm3: hook refused a documented read
$out"
fi

# ---------------------------------------------------------------------------
# Arm 4: a prefix row. The refusal message advertises MEMRA_TCOL_*-style family rows, so the
# advice has to actually work — a message that recommends a fix that does not fix it is how a
# gate teaches people to reach for the override instead.
# ---------------------------------------------------------------------------
stage arm4
plant_read "$work"
document "$work" "MEMRA_FIXTURE_GUARD_*"
if out=$(push "$work"); then
    ok "arm4: prefix row MEMRA_FIXTURE_GUARD_* covers the read"
else
    bad "arm4: prefix row did not cover the read
$out"
fi

# ---------------------------------------------------------------------------
# Arm 5: the escape hatch. Fail-closed but escapable ON THE RECORD — an emergency push must be
# possible and must leave a trace, so both halves are asserted: the printed line AND the log.
# ---------------------------------------------------------------------------
stage arm5
plant_read "$work"
if out=$(push "$work" MEMRA_SKIP_FLAGS_CENSUS=1); then
    if [[ "$out" == *"SKIPPED"* && "$out" == *"MEMRA_SKIP_FLAGS_CENSUS=1"* ]]; then
        ok "arm5: MEMRA_SKIP_FLAGS_CENSUS=1 lets the push through and ANNOUNCES itself"
    else
        bad "arm5: skip was silent — no announcement in the push output
$out"
    fi
    skip_log=$work/.git/memra-gate-skips.log
    if [[ -s "$skip_log" ]] && grep -q 'flags-census.*MEMRA_SKIP_FLAGS_CENSUS=1' "$skip_log"; then
        ok "arm5: the skip is on the record in $(basename "$skip_log")"
    else
        bad "arm5: no durable trace at $skip_log (a printed-only skip dies with the terminal)"
    fi
else
    bad "arm5: MEMRA_SKIP_FLAGS_CENSUS=1 did not let the push through
$out"
    bad "arm5: (log not evaluated — the escape hatch did not work)"
fi

# ---------------------------------------------------------------------------
# Arm 6: rg absent -> WARN and continue. This is the arm's only fail-OPEN branch, so it gets a
# test; and the condition is SELF-CHECKED, because a PATH that failed to hide rg would make this
# arm pass by testing nothing at all — the "constant-true conjunct" defect A-9 found in
# tools/test_check_flags.sh's own live block.
# ---------------------------------------------------------------------------
stage arm6
plant_read "$work"
norg=$tmp/norg
mkdir -p "$norg"
missing=""
for bin in git python3 mktemp rm sort grep sed cat date tr wc stat find comm awk env dirname \
           basename head tail cut; do
    path=$(type -P "$bin" 2>/dev/null || true)
    if [[ -n "$path" ]]; then ln -sf "$path" "$norg/$bin"; else missing="$missing $bin"; fi
done
if [[ -n "$missing" ]]; then
    bad "arm6: cannot build an rg-less PATH — missing binaries:$missing"
elif PATH="$norg" type -P rg >/dev/null 2>&1; then
    bad "arm6: BROKEN CONDITION — rg is still visible on the restricted PATH, so this arm
     would pass without ever exercising the missing-rg branch"
else
    out=$(cd "$work" && PATH="$norg" git push origin HEAD:main 2>&1); rc=$?
    if (( rc == 0 )) && [[ "$out" == *"rg is not on PATH"* ]]; then
        ok "arm6: missing rg WARNS by name and does not block"
    else
        bad "arm6: missing rg did not warn-and-continue (rc=$rc)
$out"
    fi
fi

# ---------------------------------------------------------------------------
# Arm 7: wiring census. The arms above prove the hook refuses; this proves the hook in the REPO
# is the one they exercised, and that nothing here traded the CI backstop away for it.
# ---------------------------------------------------------------------------
# The INVOCATION, not any mention of it. A bare `grep tools/check-flags.sh` passed while the
# teeth run below had the call replaced by `elif false; then` — the hook's own rationale comment
# names the script three times, so the loose pattern was matching prose. Anchoring on the
# command-substitution form is the difference between "the file talks about the census" and
# "the file runs it".
if grep -qE '^[[:space:]]*elif ! flags_out=\$\(tools/check-flags\.sh' "$hook"; then
    ok "arm7: tools/hooks/pre-push INVOKES tools/check-flags.sh (not just mentions it)"
else
    bad "arm7: tools/hooks/pre-push has no live tools/check-flags.sh invocation"
fi
if grep -q 'test_check_flags.sh' "$here/.github/workflows/ci.yml"; then
    ok "arm7: ci.yml still runs the self-test backstop (not weakened, not duplicated)"
else
    bad "arm7: ci.yml no longer runs test_check_flags.sh — the backstop was removed"
fi
if grep -q 'test_flags_guard.sh' "$here/.github/workflows/ci.yml"; then
    ok "arm7: ci.yml runs this fixture (A-18: a fixture with no caller is born invisible)"
else
    bad "arm7: ci.yml does not run this fixture"
fi

# ---------------------------------------------------------------------------
# Arm 8: the OTHER escape hatch. MEMRA_SKIP_PERF_CI is the most-used override in this repo (at
# least four uses on 2026-08-19 alone) and until 2026-08-23 it was a NEGATIVE condition, so
# setting it took no branch and produced no output and no trace whatsoever. It now announces and
# logs like the census skip.
#
# It only engages when the push touches an engine file, so the arm has to commit one — that is
# the branch's real precondition and an arm that skipped it would be testing nothing. The engine
# file is deliberately FLAG-FREE so the census stays green and a failure here can only be the
# perf-gate skip. MEMRA_MODELS_DIR is irrelevant: the skip branch is taken before the model dir
# is consulted, which the assertions below prove by never providing one.
# ---------------------------------------------------------------------------
stage arm8
( cd "$work" \
    && mkdir -p crates/memra-engine/src \
    && printf '// engine change with no MEMRA_ read, so only the perf gate can object.\n' \
        > crates/memra-engine/src/lib.rs \
    && git add crates/memra-engine/src/lib.rs \
    && git commit -qm 'engine file: engages the perf-ci freshness gate' ) >/dev/null

# MEMRA_MODELS_DIR IS PINNED TO A PATH THAT CANNOT EXIST, and that pin is the fix for a real
# defect this arm shipped with (main red at 5ffa711c32, repaired here). Unpinned, the precondition
# push below takes a DIFFERENT branch on each machine:
#   * rig      — /data/ai-ml/hf-models exists, so the gate looks for perf-ci.jsonl, does not find
#                it in a throwaway repo, and REFUSES the push (exit 1).
#   * runner   — no model dir, so the gate prints its NOTE and the push SUCCEEDS.
# Both satisfied the assertion below, so the precondition arm went green in both places — for
# opposite reasons. The consequence landed on the NEXT push: on the runner the precondition push
# had already advanced the bare origin, so the override push had nothing to send, git printed
# "Everything up-to-date" and NEVER RAN THE HOOK. Two arms then failed for a reason that had
# nothing to do with the code under test. An environment-dependent fixture is a fixture that
# reports on the machine instead of the change.
export MEMRA_MODELS_DIR="$tmp/no-such-models-dir"

# PRECONDITION: without the override, this push must reach the perf-ci gate at all. Otherwise
# "the skip announced itself" would be a claim about a branch nothing else can reach.
out=$(push "$work" "MEMRA_MODELS_DIR=$MEMRA_MODELS_DIR")
if [[ "$out" == *"perf-ci"* || "$out" == *"perf_ci"* || "$out" == *"model dir"* ]]; then
    ok "arm8: the engine file DOES engage the perf-ci gate (precondition, not assumed)"
else
    bad "arm8: engine file did not engage the perf-ci gate — arm 8 would prove nothing
$out"
fi

# A SECOND commit, so the override push always has work to send no matter whether the
# precondition push landed. A no-op push exits 0 without running any hook, which is the trap
# above: green-looking plumbing, zero coverage.
( cd "$work" \
    && printf '// second engine change, so the override push is never a no-op.\n' \
        >> crates/memra-engine/src/lib.rs \
    && git add crates/memra-engine/src/lib.rs \
    && git commit -qm 'engine file: second change for the override push' ) >/dev/null

if out=$(push "$work" MEMRA_SKIP_PERF_CI=1 "MEMRA_MODELS_DIR=$MEMRA_MODELS_DIR"); then
    # Assert the push was NOT a no-op before reading anything into its output. This is the
    # permanent guard for the defect described above: `git push` with nothing to send succeeds,
    # prints "Everything up-to-date", and runs no hook — so every downstream assertion about
    # hook behaviour would be measuring silence.
    if [[ "$out" == *"Everything up-to-date"* ]]; then
        bad "arm8: the override push was a NO-OP (no hook ran) — the arm proved nothing
$out"
        bad "arm8: (log not evaluated — nothing was pushed)"
    else
        if [[ "$out" == *"SKIPPED"* && "$out" == *"MEMRA_SKIP_PERF_CI=1"* ]]; then
            ok "arm8: MEMRA_SKIP_PERF_CI=1 ANNOUNCES itself (it used to be silent)"
        else
            bad "arm8: perf-ci skip was silent — no announcement in the push output
$out"
        fi
        skip_log=$work/.git/memra-gate-skips.log
        # The row must NAME WHAT IT LET THROUGH, not merely that a skip happened.
        if [[ -s "$skip_log" ]] \
            && grep -q 'perf-ci.*MEMRA_SKIP_PERF_CI=1.*engine_files=.*memra-engine' "$skip_log"
        then
            ok "arm8: the row names the engine files it waved past"
        else
            bad "arm8: no durable row naming the engine files at $skip_log
$(cat "$skip_log" 2>/dev/null)"
        fi
    fi
else
    bad "arm8: MEMRA_SKIP_PERF_CI=1 did not let the push through
$out"
    bad "arm8: (log not evaluated — the escape hatch did not work)"
fi

# ---------------------------------------------------------------------------
# Arm 9: a force-pushed topic rebased onto newer main must only scan commits the push actually
# introduces.  `old_topic..rewritten_topic` also contains main commits that are already on the
# destination remote; the old hook fed those historical blobs to the boundary scanner, where a
# final-tree hash allowlist cannot legitimately cover them.  The scanner stub below has teeth in
# both directions: it requires the rewritten topic commit and rejects the already-remote main
# commit.  This drives a REAL force-push through the REAL hook.
# ---------------------------------------------------------------------------
stage arm9
(
    cd "$work" || exit 1
    seed_branch=$(git branch --show-current)
    git switch -qc topic
    printf 'old topic\n' > topic.txt
    git add topic.txt
    git commit -qm 'old topic tip'
    git push -q -u origin HEAD:topic

    git switch -q "$seed_branch"
    printf 'already published on main\n' > main-only.txt
    git add main-only.txt
    git commit -qm 'main-only boundary history sentinel'
    git push -q origin HEAD:main
    git fetch -q origin main topic
    main_only=$(git rev-parse HEAD)

    git switch -qc rewritten
    printf 'rewritten topic\n' > rewritten.txt
    git add rewritten.txt
    git commit -qm 'rewritten topic tip'
    rewritten=$(git rev-parse HEAD)

    # This stub replaces only the boundary evaluator.  Every other hook arm remains real or is
    # staged by stage() above.  It refuses the exact false-positive commit and also refuses an
    # empty/incorrect pushed-commit list.
    printf '%s\n' \
        'import os, pathlib, sys' \
        'path = pathlib.Path(sys.argv[sys.argv.index("--commits-file") + 1])' \
        'seen = set(path.read_text().split())' \
        'forbidden = os.environ["BOUNDARY_FORBIDDEN_COMMIT"]' \
        'expected = os.environ["BOUNDARY_EXPECTED_COMMIT"]' \
        'if forbidden in seen:' \
        '    print("boundary fixture: already-remote main commit was rescanned")' \
        '    raise SystemExit(1)' \
        'if expected not in seen:' \
        '    print("boundary fixture: rewritten topic commit was not scanned")' \
        '    raise SystemExit(1)' \
        'print("boundary fixture: pushed-only commit range")' \
        > tools/check-public-boundary.py

    BOUNDARY_FORBIDDEN_COMMIT=$main_only \
    BOUNDARY_EXPECTED_COMMIT=$rewritten \
        git push --force origin HEAD:topic > push.out 2>&1
) && out=$(cat "$work/push.out") && [[ "$out" == *"pushed-only commit range"* ]]
if (( $? == 0 )); then
    ok "arm9: force-push boundary range excludes commits already published on origin/main"
else
    bad "arm9: force-push boundary range rescanned remote history or missed the new topic
${out:-}"
fi

# ---------------------------------------------------------------------------
# The fixture's own floor, the shape test_check_flags.sh uses: a run that records FEWER
# assertions than it should is BROKEN, not green. Every `bad` path above is paired so the count
# is invariant to which branch was taken.
#   arms 1-2: 2 | arm 3: 1 | arm 4: 1 | arm 5: 2 | arm 6: 1 | arm 7: 3 | arm 8: 3
#   arm 9: 1  = 14
# ---------------------------------------------------------------------------
EXPECTED_ASSERTIONS=14
total=$((pass + fail))
printf '\ntest_flags_guard: %d passed, %d failed (%d assertions, expected %d)\n' \
    "$pass" "$fail" "$total" "$EXPECTED_ASSERTIONS"
if (( total != EXPECTED_ASSERTIONS )); then
    printf 'test_flags_guard: BROKEN FIXTURE — recorded %d assertions, expected %d\n' \
        "$total" "$EXPECTED_ASSERTIONS" >&2
    exit 3
fi
(( fail == 0 ))
