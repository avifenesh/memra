#!/usr/bin/env bash
# SPEC ACCEPTANCE-RACE lane (2026-09-01) — THE REPRODUCTION PROTOCOL, banked as the
# REGRESSION HARNESS. CI has no GPU, so this script is the standing gate for the defect:
# `glm5-spec-ppn-gate` is nondeterministic ONLY under host load, so an unloaded run of the
# gate proves nothing about it and must never be recorded as coverage.
#
# WHAT IT REPORTS PER REP
#   * `[E forced-rejection sweep K=7]`  — the arm the defect was found on (14/42 PASS vs
#     13/42 FAIL, perfectly bimodal: exactly one silently lost acceptance).
#   * `[P0 prime-determinism]` / `[P1 prime-determinism-post-spec]` — the arms added by
#     this lane. R repeated split primes of one prompt against the door-OFF prime, so one rep
#     samples the race R times instead of once. P1 runs AFTER the spec arms and is the
#     DETECTOR; P0's regime is measurably too quiet (see LANE.md §5).
#
# LOAD SOURCE, and the two rules it obeys:
#   * capped: `nice -n 19` spinners inside systemd user scopes with an explicit CPUQuota
#     (the owner's no-uncapped-local-CPU law);
#   * raised INSIDE the rig lock and torn down before it is released, so a co-tenant lane
#     that holds `/tmp/memra-5090.lock` is never perturbed by this harness. Never use
#     another lane's live work as the load source.
#
# Rig law: exactness only — no timing number is read from any log here.
#
# usage: repro.sh <outdir-tag> <stages> <reps> [env assignments...]
#   e.g. repro.sh fix-n2 2 12
#        repro.sh ctl-n2 2 12 MEMRA_PP_EXIT_PUBLISH=0     # the known-racy control arm
set -u
cd "$(dirname "$0")/../../.."
TAG="${1:?tag}"; STAGES="${2:?stages}"; REPS="${3:?reps}"; shift 3
OUT="research/glm53-flash-bringup-20260827/accrace-20260901/receipts/$TAG"
mkdir -p "$OUT"
BIN=./target/debug/glm5-spec-ppn-gate
[ -x "$BIN" ] || { echo "missing $BIN — build it OUTSIDE the lock first (the sccache-under-flock trap)"; exit 2; }

PRIME_REPS="${ACCRACE_PRIME_REPS:-8}"
# PASS-line count of a fully green gate run. It moved 23 -> 24 when this lane added arm P0;
# assert it, never merely print it (a count of 0 with exit 0 would otherwise read as green).
WANT_PASS="${ACCRACE_WANT_PASS:-25}"
SPINNERS="${ACCRACE_SPINNERS:-6}"
QUOTA="${ACCRACE_CPUQUOTA:-400%}"
GPULOAD="${ACCRACE_GPULOAD:-1}"

# The load window lives in a helper so it is raised and dropped INSIDE the flock.
INNER="$(mktemp /tmp/accrace-inner-XXXX.sh)"
cat >"$INNER" <<'SH'
#!/usr/bin/env bash
# ONE rep's LOAD WINDOW, raised and dropped INSIDE the caller's flock.
#   * HOST load: capped `nice -n 19` spinners in systemd user scopes with a CPUQuota.
#   * GPU load: a SECOND CUDA CONTEXT of our OWN gate binary, looping with its output
#     discarded. This is load, never a sample. It is required: the defect's original
#     reproduction happened while an unrelated process held a CUDA context on the card, and
#     with the rig otherwise idle inside our own lock window a 12x2 interleaved A/B went
#     0/96 on the KNOWN-RACY control arm — i.e. the window tested nothing. Never use another
#     lane's live work as the load source (the standing ban); this context is ours and it
#     runs only while we hold the lock.
set -u
LOG="$1"; N="$2"; Q="$3"; GPU="$4"; BIN="$5"; shift 5
for n in $(seq "$N"); do
  systemd-run --user --quiet --unit="accrace-load$n" --scope -p CPUQuota="$Q" \
    nice -n 19 /usr/bin/timeout 1200 /usr/bin/yes >/dev/null 2>&1 &
done
GPUPID=""
if [ "$GPU" = 1 ]; then
  ( while :; do NVIDIA_TF32_OVERRIDE=0 MEMRA_PP_STAGES=2 nice -n 10 "$BIN" 2 24 20 0 2 >/dev/null 2>&1 || true; sleep 0.2; done ) &
  GPUPID=$!
fi
sleep 1
env NVIDIA_TF32_OVERRIDE=0 "$@" >"$LOG" 2>&1
rc=$?
if [ -n "$GPUPID" ]; then
  kill "$GPUPID" 2>/dev/null
  pkill -P "$GPUPID" 2>/dev/null
fi
for n in $(seq "$N"); do systemctl --user stop "accrace-load$n.scope" >/dev/null 2>&1; done
exit $rc
SH
chmod +x "$INNER"
trap 'rm -f "$INNER"' EXIT

SUM="$OUT/repstudy.txt"
: >"$SUM"
echo "# tag=$TAG stages=$STAGES reps=$REPS prime_reps=$PRIME_REPS want_pass=$WANT_PASS env='$*' spinners=$SPINNERS quota=$QUOTA gpuload=$GPULOAD" >>"$SUM"
fails=0
for i in $(seq "$REPS"); do
  log="$OUT/rep$i.log"
  # CAPTURE-THEN-GATE: no pipe on the failable step; take rc, then judge the file.
  flock /tmp/memra-5090.lock "$INNER" "$log" "$SPINNERS" "$QUOTA" "$GPULOAD" "$BIN" \
    "$@" timeout 900 nice -n 5 "$BIN" "$STAGES" 24 20 0 "$PRIME_REPS"
  rc=$?
  echo "exit=$rc" >>"$log"
  got="$(grep -cE 'gate PASS' "$log")"
  echo "pass_lines=$got want=$WANT_PASS" >>"$log"

  sweep_line="$(grep -E 'gate (PASS|FAIL) \[E forced-rejection' "$log" | head -1)"
  p0_line="$(grep -E 'gate (PASS|FAIL) \[P0 prime-determinism' "$log" | head -1)"
  p1_line="$(grep -E 'gate (PASS|FAIL) \[P1 prime-determinism-post-spec' "$log" | head -1)"
  case "$sweep_line" in
    *"gate PASS"*) sweep=PASS ;; *"gate FAIL"*) sweep=FAIL ;; *) sweep=MISSING ;;
  esac
  case "$p0_line" in
    *"gate PASS"*) p0=PASS ;; *"gate FAIL"*) p0=FAIL ;; *) p0=MISSING ;;
  esac
  case "$p1_line" in
    *"gate PASS"*) p1=PASS ;; *"gate FAIL"*) p1=FAIL ;; *) p1=MISSING ;;
  esac
  acc="$(printf '%s' "$sweep_line" | grep -oE '\([0-9]+/[0-9]+\)' | tail -1)"
  dev="$(printf '%s' "$p0_line" | grep -oE '[0-9]+ deviated' | grep -oE '^[0-9]+')"
  dev1="$(printf '%s' "$p1_line" | grep -oE '[0-9]+ deviated' | grep -oE '^[0-9]+')"

  bad=0
  [ "$rc" -ne 0 ] && bad=1
  [ "$sweep" != PASS ] && bad=1
  [ "$p0" != PASS ] && bad=1
  [ "$p1" != PASS ] && bad=1
  [ "$got" -ne "$WANT_PASS" ] && bad=1
  fails=$(( fails + bad ))
  printf '%s rep%-3s exit=%-3s pass_lines=%-3s sweep=%-7s accepted=%-8s P0=%-7s dev0=%s/%s P1=%-7s dev1=%s/%s\n' \
    "$TAG" "$i" "$rc" "$got" "$sweep" "${acc:-?}" "$p0" "${dev:-?}" "$PRIME_REPS" \
    "$p1" "${dev1:-?}" "$PRIME_REPS" >>"$SUM"
done
echo "$TAG FAILS=$fails / $REPS" >>"$SUM"
cat "$SUM"
[ "$fails" -eq 0 ]
