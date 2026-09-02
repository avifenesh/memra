#!/usr/bin/env bash
# Teeth fixture for the round-2 gate-integrity fixes (GATE-INTEGRITY-20260819, A-9/
# A-10/A-16). One arm per fix, and every arm FORCES the failure so the check is proven able to
# fail for the RIGHT reason. A gate nobody has seen go red is not evidence.
#
# A-1/A-2 (the discarded-suite and kernel-check-skip verdicts in tools/validate-h100.sh) were
# retired with that battery on 2026-09-02, when the sm_90a lane left CI by owner ruling. Their
# shapes live on where they were copied to: tools/skip-census.py ("THE SHAPE, copied
# deliberately from round 2's kernel-check fix in tools/validate-h100.sh") and local-ci.sh's
# kernel-check verdict line.
#
# What it does NOT need: a GPU, a model, a network, or the CUDA toolkit. Throwaway repos under
# mktemp, stub binaries on PATH, and a real TCP listener for the port arms.
#
# VERDICT DISCIPLINE (round 1's lesson, GATE-INTEGRITY-20260819 §3). Round 1's first fixture
# draft printed "1 passed / 0 failed" while an arm was visibly FAILing, because every arm ran in
# a ( subshell ) and the pass/fail counters incremented inside were discarded. So: verdicts are
# appended to a FILE, the summary counts the file, and a run that records fewer assertions than
# EXPECTED_ASSERTIONS fails as a BROKEN FIXTURE rather than passing as a green one.
set -uo pipefail
cd -- "$(dirname -- "$0")/.."
REPO=$PWD
# TEETH HARNESS ONLY. Point this at a directory holding the PRE-FIX copies of the gates
# (`git show origin/main:tools/<f> > $dir/<f>`) and every arm below must go red — that run is the
# fixture's own decisiveness receipt, the same method round 1 used via MEMRA_PUSH_RANGE. Default
# is the live tree, so it cannot be used to make a real run pass.
SRC=${MEMRA_GATE_SRC_DIR:-$REPO/tools}

VERDICTS=$(mktemp "${TMPDIR:-/tmp}/gate-integrity-r2-verdicts-XXXXXX")
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/gate-integrity-r2-XXXXXX")
# /tmp hygiene law: the task that creates scratch deletes it.
trap 'rm -rf "$SCRATCH"; rm -f "$VERDICTS"' EXIT

# A-9: 5 · A-16: 6 teeth/control + 7 wiring · A-10: 4 lock-gate + 5 prime-split
# + 2 comparator-evidence = 29 (was 40 before the A-1/A-2 arms left with validate-h100.sh,
# 2026-09-02). It is an EQUALITY, not a floor: this constant has already caught
# three of its own miscounts (19 vs 23 in test_check_flags; 24 then 38 here) and it is what turns
# an arm that silently stops running — a `bad` branch replacing four assertions with one, a
# `continue` added to a loop — into a red run instead of a smaller green one.
EXPECTED_ASSERTIONS=29

ok()   { printf 'ok   %s\n' "$1"; printf 'PASS %s\n' "$1" >> "$VERDICTS"; }
bad()  { printf 'FAIL %s\n' "$1" >&2; printf 'FAIL %s\n' "$1" >> "$VERDICTS"; }

# assert_has <label> <text> <needle>
assert_has() {
    local label=$1 text=$2 needle=$3
    if [[ "$text" == *"$needle"* ]]; then ok "$label"; else
        bad "$label (missing: $needle)"
        printf '%s\n' "$text" | sed 's/^/      /' | head -25 >&2
    fi
}
# assert_not <label> <text> <needle>
assert_not() {
    local label=$1 text=$2 needle=$3
    if [[ "$text" != *"$needle"* ]]; then ok "$label"; else
        bad "$label (unexpectedly present: $needle)"
        printf '%s\n' "$text" | sed 's/^/      /' | head -25 >&2
    fi
}
# assert_rc <label> <want> <got>
assert_rc() {
    local label=$1 want=$2 got=$3
    if [ "$got" = "$want" ]; then ok "$label"; else bad "$label (rc=$got want=$want)"; fi
}

echo "=== A-9: the flags fixture asserts the CENSUS, not just the doc ==="
# TEETH: a census that cannot SEE the flag, while docs/FLAGS.md still documents it — the exact
# v0.94.0 state. The pre-fix live block's first conjunct never mentioned the flag name, so it
# collapsed to "the name appears in FLAGS.md" and passed.
mkdir -p "$SCRATCH/blindcensus"
cat > "$SCRATCH/blindcensus/check-flags.sh" <<'STUB'
#!/usr/bin/env bash
# Simulates the pre-v0.94.0 census: identical to the real gate except that one flag is
# invisible to it — which is what a hand-written runtime_dirs list and a const-indirected read
# did for real.
out=$(tools/check-flags.sh "$@" 2>&1); rc=$?
printf '%s\n' "$out" | grep -v '^MEMRA_ALLOW_UNKNOWN_PRETOKENIZER$'
exit $rc
STUB
chmod +x "$SCRATCH/blindcensus/check-flags.sh"
out=$(MEMRA_CHECK_FLAGS="$SCRATCH/blindcensus/check-flags.sh" tools/test_check_flags.sh 2>&1)
RC=$?
assert_has "A-9 teeth: a blind census fails the live block" \
    "$out" "MEMRA_ALLOW_UNKNOWN_PRETOKENIZER is NOT in the census"
assert_has "A-9 teeth: the doc-only conjunct still passes, proving it was never the test" \
    "$out" "MEMRA_ALLOW_UNKNOWN_PRETOKENIZER is documented"
assert_rc "A-9 teeth: the fixture exits nonzero" 1 "$RC"
# CONTROL: the real census passes, and the assertion FLOOR is intact.
out=$(tools/test_check_flags.sh 2>&1); RC=$?
assert_rc "A-9 control: the live tree is green" 0 "$RC"
# PINNED ON PURPOSE, and it is a deliberate two-file coupling: a hardcoded number means shrinking
# the flags fixture's floor cannot pass quietly here. It went 19 -> 24 on 2026-08-23 when the
# census's 75-name grandfather list was deleted and four retirement arms plus a
# baseline-still-absent live arm were added. If you change EXPECTED_ASSERTIONS in
# tools/test_check_flags.sh, change this line in the same commit — this arm exists precisely so
# that is not optional. (It caught that bump within one run, which is the pin working.)
assert_has "A-9 control: the fixture reports its assertion count" "$out" "expected 24"

echo "=== A-16: the port guard refuses rather than measuring a stranger ==="
# TEETH: a real listener on the port. `python3 -c` binds, prints the port, and holds it until
# killed — no model, no server, no GPU.
LISTEN_OUT=$SCRATCH/listener.port
python3 - "$LISTEN_OUT" > /dev/null 2>&1 <<'PY' &
import socket, sys, time
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(8)
open(sys.argv[1], "w").write(str(s.getsockname()[1]))
time.sleep(120)
PY
LISTEN_PID=$!
for _ in $(seq 200); do [ -s "$LISTEN_OUT" ] && break; sleep 0.05; done
BUSY_PORT=$(cat "$LISTEN_OUT" 2>/dev/null || echo "")
if [ -z "$BUSY_PORT" ]; then
    bad "A-16 teeth: could not bind a listener (fixture setup)"
else
    out=$(tools/port-guard.sh check fixture-gate "$BUSY_PORT" MEMRA_FIXTURE_PORT 2>&1); RC=$?
    assert_rc "A-16 teeth: an occupied port is a refusal" 1 "$RC"
    assert_has "A-16 teeth: the refusal names the port" "$out" "port $BUSY_PORT is already LISTENing"
    assert_has "A-16 teeth: the refusal names the override" "$out" "MEMRA_FIXTURE_PORT=<free port>"
    kill "$LISTEN_PID" 2>/dev/null; wait "$LISTEN_PID" 2>/dev/null
    # CONTROL: the same port, now free, passes. Without this the guard could be "always refuse".
    for _ in $(seq 100); do tools/port-guard.sh check f "$BUSY_PORT" >/dev/null 2>&1 && break; sleep 0.05; done
    tools/port-guard.sh check fixture-gate "$BUSY_PORT" MEMRA_FIXTURE_PORT >/dev/null 2>&1
    assert_rc "A-16 control: a free port passes" 0 "$?"
fi

# TEETH: no observability tool at all. An unobservable port must not read as a free one.
out=$(env PATH=/nonexistent "$(command -v bash)" tools/port-guard.sh check fixture-gate 8178 2>&1); RC=$?
assert_rc "A-16 teeth: no ss and no lsof is rc=2, not a pass" 2 "$RC"
assert_has "A-16 teeth: the blind case says it is blind" "$out" "cannot observe listening sockets"

# WIRING: the guard has to be CALLED, and called before the server starts. A shared helper
# nobody sources is the A-18 shape (11 gates with no caller) wearing a new hat.
for f in serve-smoke.sh serve-stress-gate.sh apikeys-gate.sh serve-st-gate.sh \
         step35-b2-geometry-gate.sh serve-gemma4-batch-gate.sh serve-gemma4-spec-gate.sh; do
    guard=$(grep -n 'memra_port_guard' "$SRC/$f" | head -1 | cut -d: -f1)
    boot=$(grep -nE 'MEMRA_ADDR=' "$SRC/$f" | head -1 | cut -d: -f1)
    if [ -n "$guard" ] && [ -n "$boot" ] && [ "$guard" -lt "$boot" ]; then
        ok "A-16 wiring: $f guards (line $guard) before it boots (line $boot)"
    else
        bad "A-16 wiring: $f guard=${guard:-none} boot=${boot:-none}"
    fi
done

echo "=== A-10: canary arms assert the specific failure, not any nonzero exit ==="
# These three gates need two GPUs and a 105GB artifact, so the assertion here is on the CODE
# PATH: the old form was `[ "$RC" -ne 0 ] && echo CANARY OK`, and 75 is what `flock -w` returns
# on timeout. What must be true after the fix: rc=75 is handled by name, and the success branch
# is guarded by evidence rather than by the exit code alone.

# The two gates that take the GPU lock: `flock -w 3600 || exit 75` means 75 must be handled by
# name, before any success branch.
for f in step35-prime-batch-gate.sh step35-b2-geometry-gate.sh; do
    src=$(cat "$SRC/$f")
    if [[ "$src" == *"exit 75"* ]] && [[ "$src" == *'-eq 75'* ]]; then
        ok "A-10: $f distinguishes the flock timeout (75) from a comparator failure"
    else
        bad "A-10: $f still cannot tell rc=75 from a real red"
    fi
    # And the old shape must be gone: a bare `-ne 0` reaching a CANARY OK is the whole defect.
    # Comments are stripped first — these files DOCUMENT the old shape verbatim, and matching the
    # documentation of a bug as the bug is its own kind of blind assertion.
    if printf '%s\n' "$src" | grep -v '^[[:space:]]*#' \
        | grep -A2 -E '"\$?RC" -ne 0|\$rc -ne 0' | grep -q 'CANARY OK\|CANARY.*PASS'; then
        bad "A-10: $f still routes 'any nonzero exit' straight to a canary pass"
    else
        ok "A-10: $f no longer routes any-nonzero-exit to a canary pass"
    fi
done
# prime-split-gate takes no lock, so it has no 75; its equivalent is telling the injected defect
# (PIPE-NOT-LIVE) apart from the probe dying (DOOR-SHUT, no verdict) and from a DIFFERENT defect
# (MISMATCH / SPLIT-NOT-LIVE), which its own header says the canary must not produce.
psrc=$(cat "$SRC/prime-split-gate.sh")
for needle in 'DOOR-SHUT' 'ppsplit verdict: \*\*\* RED' 'MISMATCH|SPLIT-NOT-LIVE' 'PIPE-NOT-LIVE'; do
    if printf '%s\n' "$psrc" | grep -qE "$needle"; then
        ok "A-10: prime-split canary discriminates $needle"
    else
        bad "A-10: prime-split canary does not look for $needle"
    fi
done
if printf '%s\n' "$psrc" | grep -v '^[[:space:]]*#' | grep -A2 -E '\$rc -ne 0' | grep -q 'PASS'; then
    bad "A-10: prime-split still routes 'any nonzero exit' to PASS"
else
    ok "A-10: prime-split no longer routes any-nonzero-exit to PASS"
fi
# And the two lock-taking gates must reach the success branch only through the comparator's words.
assert_has "A-10: prime-batch canary requires the comparator's own verdict" \
    "$(cat "$SRC/step35-prime-batch-gate.sh")" 'NOT-LIVE'
# The needle is the VLOG read, not the message: the pre-fix file already CONTAINS that FAIL
# string (it is the verdict block's own line) — it just never looked at it from the canary. An
# assertion that matches the pre-fix file too is exactly the constant-true shape being fixed.
assert_has "A-10: b2-geometry canary reads that assertion out of the banked verdicts" \
    "$(cat "$SRC/step35-b2-geometry-gate.sh")" 'first B>1'"'"' line" "$VLOG"'

# ---------------------------------------------------------------------------
# Summary, counted from the FILE.
# ---------------------------------------------------------------------------
pass=$(grep -c '^PASS ' "$VERDICTS" || true)
fail=$(grep -c '^FAIL ' "$VERDICTS" || true)
total=$(( ${pass:-0} + ${fail:-0} ))
printf '\ntest_gate_integrity_r2: %d passed, %d failed (%d assertions, expected %d)\n' \
    "${pass:-0}" "${fail:-0}" "$total" "$EXPECTED_ASSERTIONS"
if (( total != EXPECTED_ASSERTIONS )); then
    printf 'test_gate_integrity_r2: BROKEN FIXTURE — recorded %d assertions, expected %d.\n' \
        "$total" "$EXPECTED_ASSERTIONS" >&2
    printf '  A run that asserts less than it claims is not a green run.\n' >&2
    exit 3
fi
(( ${fail:-0} == 0 ))
