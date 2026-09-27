#!/usr/bin/env bash
# Teeth for tools/spec-on-cache-hit-gate.sh's two lock arms (C day 28, research/spill-c-20260919/DAY28.md).
# GPU-less: no model, no server, no rig lock. The gate's `--lock-self-test FILE` runs the arm's launch
# wrapper around a probe that asks whether FILE is locked while the wrapper runs:
#   default arm        must read probe=held  (the gate's own `flock -w 300` wraps every boot)
#   --external-lock FD must read probe=free  (under the collector's hold the gate takes no lock of its own;
#                                             the file's inode and mtime are unchanged)
#   --external-lock x  must be REFUSED with exit 2 before anything runs
# Verdicts go to a file and the summary counts them (the gate-integrity fixture's discipline): fewer
# assertions than expected is a broken fixture, not a green run.
set -uo pipefail
cd -- "$(dirname -- "$0")/.." || exit 1
GATE=tools/spec-on-cache-hit-gate.sh
EXPECTED=7
VERDICTS=$(mktemp "${TMPDIR:-/tmp}/hitgate-lock-verdicts-XXXXXX")
LOCK=$(mktemp "${TMPDIR:-/tmp}/hitgate-lock-selftest-XXXXXX")
trap 'rm -f "$VERDICTS" "$LOCK"' EXIT
say() { echo "$1" | tee -a "$VERDICTS"; }
check() { # $1 name  $2 condition-result (0 ok)  $3 detail
    if [ "$2" = 0 ]; then say "ok: $1 ($3)"; else say "FAIL: $1 ($3)"; fi
}

# 1. default arm: the wrapper holds the lock while the boot runs
out=$(bash "$GATE" --lock-self-test "$LOCK" 2>&1); rc=$?
check "default arm exits 0" "$rc" "rc=$rc"
if [[ $out == *"owner=internal-canonical"*"probe=held"* ]]; then c=0; else c=1; fi; check "default arm: the wrapper holds the lock (probe=held)" $c "$out"

# 2. external arm: the gate takes no lock of its own (FD open on the file, held by nobody)
exec 7>>"$LOCK"
before=$(stat -c '%i:%Y' "$LOCK")
out=$(bash "$GATE" --external-lock 7 --lock-self-test "$LOCK" 2>&1); rc=$?
after=$(stat -c '%i:%Y' "$LOCK")
exec 7>&-
check "external arm exits 0" "$rc" "rc=$rc"
if [[ $out == *"owner=collector fd=7"*"wrapper=none"*"probe=free"* ]]; then c=0; else c=1; fi; check "external arm: no wrapper, the lock file is not held by the gate (probe=free)" $c "$out"
if [ "$before" = "$after" ]; then c=0; else c=1; fi; check "external arm: the lock file's inode and mtime are unchanged" $c "before=$before after=$after"

# 3. a non-numeric FD is refused before anything runs
out=$(bash "$GATE" --external-lock x --lock-self-test "$LOCK" 2>&1); rc=$?
if [ "$rc" = 2 ] && [[ $out == REFUSED:* ]]; then c=0; else c=1; fi; check "external arm refuses a non-numeric FD with exit 2" $c "rc=$rc $out"

# 4. the self-test never reaches the positional contract (no EV dir, no port guard): a missing FILE is refused
out=$(bash "$GATE" --lock-self-test 2>&1); rc=$?
if [ "$rc" = 2 ] && [[ $out == REFUSED:* ]]; then c=0; else c=1; fi; check "--lock-self-test without a file is refused with exit 2" $c "rc=$rc $out"

ok=$(grep -c '^ok: ' "$VERDICTS"); fail=$(grep -c '^FAIL: ' "$VERDICTS")
if [ "$((ok + fail))" -ne "$EXPECTED" ]; then
    echo "SPEC-ON-CACHE-HIT-GATE LOCK TEETH: BROKEN FIXTURE ($((ok + fail)) of $EXPECTED assertions recorded)"; exit 1
fi
if [ "$fail" -ne 0 ]; then echo "SPEC-ON-CACHE-HIT-GATE LOCK TEETH: $fail FAILURE(S)"; exit 1; fi
echo "SPEC-ON-CACHE-HIT-GATE LOCK TEETH: ALL GREEN ($ok ok)"
