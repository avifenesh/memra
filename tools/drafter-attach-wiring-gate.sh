#!/usr/bin/env bash
# drafter-attach-wiring-gate.sh — the drafter-attach assertion must be WIRED, COUPLED,
# SATISFIABLE and EXECUTED. CPU only; no GPU, no server, no artifact.
#
# WHY THIS EXISTS. `tools/assert-drafter-attached.sh` was wired into five gates in 6d2f334a7b by
# the one lane of the v0.95.0 train with no banked battery, and it had NEVER EXECUTED ONCE. Its
# first run (the v0.95.0 release battery) failed immediately — not on an engine defect, but
# because the gemma arm's assertion was UNSATISFIABLE BY CONSTRUCTION: the gate matched on the
# artifact path while the engine's attach line did not name its artifact. Fixed in fda04083f4.
#
# The residue was worse than the defect. Of the five gates carrying the assertion, exactly ONE
# is in the battery, and within that one the qwen arms no-op it (`assert_mtp_drafter()` returns 0
# when MTP_DRAFT is empty, and local-ci.sh never set MEMRA_GATE_MTP_DRAFT). So four of five would
# rot exactly the way the gemma arm rotted, and nothing would notice. A gate that exists but
# never executes is WORSE than no gate, because it reads as coverage.
#
# This gate closes that by asserting four things that a rotting wiring cannot satisfy:
#
#   A. CENSUS      — every call site of assert-drafter-attached.sh under tools/ is REGISTERED
#                    here. Wire the assert into a sixth gate without registering it and this
#                    fails: new coverage must arrive with its proof-of-execution.
#   B. COUPLING    — each call site asserts THE SAME shell variable the gate hands the engine.
#                    A gate that boots with MEMRA_MTP_DRAFT=$X and asserts "$Y" is testing
#                    nothing; that mismatch is the seam trap that made a v0.94.0 cell pass while
#                    never loading a drafter.
#   C. SATISFIABLE — the engine's attach line for each seam INTERPOLATES ITS PATH. This is the
#                    exact fda04083f4 defect class: an assertion that greps for a path against a
#                    log line that structurally cannot contain one.
#   D. EXECUTED    — assert-drafter-attached.sh is RUN, per seam, in BOTH directions, against
#                    fixtures carrying the engine's real line shapes: correct path => 0,
#                    wrong path => 1, no attach line => 1. Teeth, every battery run.
#
# MEMRA_WIRING_TEETH=<subject> forces one subject to fail on purpose (used to prove this gate
# can fail; see docs). Exit 0 = all green.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
TOOL="$HERE/assert-drafter-attached.sh"
TEETH=${MEMRA_WIRING_TEETH:-}
FAILS=0
ok()   { echo "  ok: $*"; }
bad()  { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }

# ---------------------------------------------------------------- the registry
# gate-file | seam | variable the gate asserts | the attach-env assignment it must boot with
REG="
spec-on-cache-hit-gate.sh|qwen |MTP_DRAFT|MEMRA_MTP_DRAFT=\$MTP_DRAFT
spec-on-cache-hit-gate.sh|gemma|DRAFT    |MEMRA_DRAFT=\$DRAFT
spec-cache-gate.sh       |qwen |MTP_DRAFT|MEMRA_MTP_DRAFT=\$MTP_DRAFT
spec-cache-mixed-gate.sh |qwen |MTP_DRAFT|MEMRA_MTP_DRAFT=\$MTP_DRAFT
spec-cache-mixed-gate.sh |gemma|DRAFT    |MEMRA_DRAFT=\$DRAFT
btemp-power.sh           |qwen |DRAFT    |MEMRA_MTP_DRAFT=\$DRAFT
sampled-restore-ab.sh    |qwen |MTP_DRAFT|MEMRA_MTP_DRAFT=\$MTP_DRAFT
"
# NOTE on the spellings: btemp-power.sh names its variable DRAFT while using the QWEN seam
# (MEMRA_MTP_DRAFT), whereas the other qwen gates name theirs MTP_DRAFT. The variable NAME is not
# the contract; the contract is that the asserted variable and the booted env var carry the SAME
# value. Check B reads each gate's own spelling out of the file, so a rename cannot pass silently
# — the first draft of this registry got btemp-power.sh's and sampled-restore-ab.sh's spellings
# crossed, and check B caught it on the first run.

echo "== drafter-attach wiring gate =="
echo "root: $ROOT"
[ -x "$TOOL" ] || { echo "FAIL: $TOOL missing or not executable"; exit 1; }

############################## A — CALL-SITE CENSUS ############################
echo
echo "-- A. CENSUS: every call site under tools/ must be registered --"
# COMMENTS ARE STRIPPED FIRST. A naive grep counts any file that merely NAMES the tool in prose
# as a call site — local-ci.sh documents this gate and was reported as an unregistered caller on
# the first run. Same trap the house already pays for with `pkill`: `sed 's/#.*//' | grep pkill`
# is the check, because a trailing comment fools the plain grep.
#
# And the count is taken with `grep -c`, NOT `grep -q`. Under `set -o pipefail`, `grep -q` exits
# as soon as it matches, SIGPIPEs the `sed` upstream, and the pipeline's status becomes 141 — so
# the test FAILED for exactly the biggest file (the 740-line hit gate, whose match is at line 117)
# while passing for the short ones. A size-dependent false negative in a coverage gate is the
# same disease this whole gate exists to cure.
FOUND=$(for f in "$HERE"/*.sh; do
            b=$(basename "$f")
            case "$b" in assert-drafter-attached.sh|drafter-attach-wiring-gate.sh) continue;; esac
            n=$(sed 's/#.*//' "$f" | grep -c 'assert-drafter-attached\.sh')
            [ "${n:-0}" -gt 0 ] && echo "$b"
        done | sort -u)
REGFILES=$(printf '%s\n' "$REG" | sed '/^$/d' | cut -d'|' -f1 | tr -d ' ' | sort -u)
if [ "$TEETH" = "census" ]; then
    REGFILES=$(printf '%s\n' "$REGFILES" | grep -v '^btemp-power.sh$')
    echo "  (TEETH: btemp-power.sh removed from the registry on purpose)"
fi
UNREG=$(comm -23 <(printf '%s\n' "$FOUND") <(printf '%s\n' "$REGFILES"))
GONE=$(comm -13 <(printf '%s\n' "$FOUND") <(printf '%s\n' "$REGFILES"))
echo "  call sites found: $(printf '%s' "$FOUND" | tr '\n' ' ')"
if [ -n "$UNREG" ]; then
    bad "gate(s) wire the drafter assertion but are NOT registered here: $(echo $UNREG)"
    echo "        A new gate must arrive with its proof-of-execution, or it rots."
else ok "no unregistered call sites"; fi
if [ -n "$GONE" ]; then
    bad "registered gate(s) no longer call the assertion: $(echo $GONE)"
else ok "every registered gate still calls it"; fi

############################## B — COUPLING ####################################
echo
echo "-- B. COUPLING: the asserted variable is the one handed to the engine --"
printf '%s\n' "$REG" | sed '/^$/d' | while IFS='|' read -r gf seam var envassign; do
    gf=$(echo "$gf" | tr -d ' '); seam=$(echo "$seam" | tr -d ' ')
    var=$(echo "$var" | tr -d ' '); envassign=$(echo "$envassign" | tr -d ' ')
    f="$HERE/$gf"
    [ -r "$f" ] || { echo "  FAIL: $gf unreadable"; continue; }
    # the call site must pass "$var" as the expected-path-substring
    if grep -q "assert-drafter-attached\.sh\".*\"\$$var\"" "$f"; then
        echo "  ok: $gf ($seam): asserts \"\$$var\""
    else
        echo "  FAIL: $gf ($seam): no call site passing \"\$$var\" as expected-path-substring"
    fi
    # ...and the gate must boot with the seam's env var set from THAT SAME variable
    if grep -qF "$envassign" "$f"; then
        echo "  ok: $gf ($seam): boots with $envassign"
    else
        echo "  FAIL: $gf ($seam): never boots with $envassign — asserted value is not the"
        echo "        value handed to the engine, so the assertion tests nothing"
    fi
done > /tmp/wiring-B.$$ 2>&1
cat /tmp/wiring-B.$$
BADB=$(grep -c '^  FAIL:' /tmp/wiring-B.$$)
rm -f /tmp/wiring-B.$$
FAILS=$((FAILS + BADB))

############################## C — SATISFIABILITY ##############################
# The engine's attach line must NAME ITS ARTIFACT. Checked in the engine source, because an
# assertion that greps for a path against a line that cannot contain one is unsatisfiable by
# construction — fda04083f4, found by this assertion's first-ever execution.
echo
echo "-- C. SATISFIABLE: each seam's engine log line interpolates its path --"
check_emit() { # $1 label  $2 source file  $3 marker  $4 placeholder
    local label=$1 src="$ROOT/$2" marker=$3 ph=$4
    [ -r "$src" ] || { bad "$label: $2 unreadable"; return; }
    local ln
    ln=$(grep -nF -- "$marker" "$src" | grep -v '^\s*//' | head -1 | cut -d: -f1)
    if [ -z "$ln" ]; then
        bad "$label: marker not found in $2 — the log contract moved and the gate did not"
        return
    fi
    local win
    win=$(sed -n "${ln},$((ln + 4))p" "$src")
    if [ "$TEETH" = "satisfiable-$label" ]; then
        win=$(printf '%s' "$win" | sed "s/$ph//g")
        echo "  (TEETH: placeholder stripped from the $label window on purpose)"
    fi
    if printf '%s' "$win" | grep -qF -- "$ph"; then
        ok "$label: emit site at $2:$ln interpolates $ph"
    else
        bad "$label: emit site at $2:$ln does NOT interpolate $ph — an assertion that"
        echo "        greps for the artifact path can never pass (the fda04083f4 defect)"
    fi
}
check_emit mtp    crates/memra-engine/src/hybrid.rs   '[mtp-draft] loading external MTP draft:' '{path}'
check_emit regime crates/memra-server/src/worker.rs   'regime draft attached ('                 '{dpath}'
check_emit gemma  crates/memra-server/src/worker.rs   'GEMMA SPEC route armed'                  '{dpath}'

############################## D — EXECUTED, WITH TEETH ########################
echo
echo "-- D. EXECUTED: the assertion runs per seam, both directions --"
WORK=$(mktemp -d /tmp/wiring-gate.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
P=/models/real-drafter.gguf
Q=/models/some-other-drafter.gguf
# Fixture lines carry the engine's REAL shapes. Check C above pins them to the source, so a
# change in the engine's wording reddens this gate instead of silently un-testing the seam.
printf '%s\n' "[mtp-draft] loading external MTP draft: $P" > "$WORK/mtp.log"
printf '%s\n' "[worker] q38: regime draft attached ($P)" > "$WORK/regime.log"
printf '%s\n' "[worker] g31: GEMMA SPEC route armed (K=5, assistant drafter attached ($P); greedy/unconstrained/text-only/solo-admission; MEMRA_GEMMA4_SPEC=0 = off)" > "$WORK/gemma.log"
printf '%s\n' 'models config = [("q38", "/models/trunk.gguf", None)]' > "$WORK/none.log"
if [ "$TEETH" = "executed" ]; then
    printf '%s\n' "[mtp-draft] loading external MTP draft: $Q" > "$WORK/mtp.log"
    echo "  (TEETH: the mtp fixture now names the WRONG artifact on purpose)"
fi

run_case() { # $1 label  $2 want-rc  $3.. tool args
    local label=$1 want=$2; shift 2
    local out rc
    out=$("$TOOL" "$@" 2>&1); rc=$?
    if [ "$rc" = "$want" ]; then
        ok "$label: rc=$rc (expected $want)"
    else
        bad "$label: rc=$rc, expected $want"
        printf '%s\n' "$out" | sed 's/^/        /'
    fi
}
# PASS direction — the seam's own line, naming the artifact the gate asked for
run_case "mtp    attach present, path matches" 0 "$WORK/mtp.log"    "$P"
run_case "regime attach present, path matches" 0 "$WORK/regime.log" "$P"
run_case "gemma  attach present, path matches" 0 --gemma "$WORK/gemma.log" "$P"
# FAIL direction 1 — the line is there but names a DIFFERENT artifact (the gemma defect shape)
run_case "mtp    wrong artifact must FAIL"     1 "$WORK/mtp.log"    "$Q"
run_case "gemma  wrong artifact must FAIL"     1 --gemma "$WORK/gemma.log" "$Q"
# FAIL direction 2 — no attach line at all (served on the trunk's own head)
run_case "mtp    no attach line must FAIL"     1 "$WORK/none.log"   "$P"
run_case "gemma  no attach line must FAIL"     1 --gemma "$WORK/none.log"  "$P"
# and an unreadable log is a failure, not a silent pass
run_case "unreadable log must FAIL"            1 "$WORK/does-not-exist.log" "$P"

echo
if [ "$FAILS" = 0 ]; then
    echo "DRAFTER-ATTACH WIRING GATE: ALL GREEN"
    exit 0
fi
echo "DRAFTER-ATTACH WIRING GATE: $FAILS FAILURE(S)"
exit 1
