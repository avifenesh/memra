#!/usr/bin/env bash
# lane/glm5-accrace — INTERLEAVED A/B of the exit-publication fix against the pre-lane
# program, in ONE load window per rep and alternating arms rep by rep
# (interleaved-ab-protocol law: a cross-window comparison is invalid on this rig).
#
# ARM CTL is `MEMRA_PP_EXIT_PUBLISH=0`, which restores the pre-lane behaviour EXACTLY — and
# that is also the PRE-EXISTING-HEAD check, done without rebuilding an old tree: all three
# sites this lane touched (`glm5_verify_rollback` / `glm5_verify_rows_ppn` in glm_spec.rs,
# `prime_cache_hyper_ppn` in hybrid_forward.rs, pp.rs) are byte-identical between the
# pre-doors head 92ea07376 and this lane's base 216ffd114, so the CTL arm IS that head's
# program for this defect.
#
# Per rep it records BOTH signals: the `[E forced-rejection sweep K=7]` arm (1 sample) and
# the `[P0 prime-determinism]` arm's deviation count (R samples — the sensitive instrument).
#
# Load: capped `nice -n 19` spinners in systemd user scopes, raised INSIDE the rig lock and
# dropped before it is released, so a co-tenant lane holding /tmp/memra-5090.lock is never
# perturbed. Rig law: exactness only, no timing number is read here.
#
# usage: ab.sh <tag> <stages> <reps> [shared env assignments...]
set -u
cd "$(dirname "$0")/../../.."
TAG="${1:?tag}"; STAGES="${2:?stages}"; REPS="${3:?reps}"; shift 3
SHARED=("$@")
OUT="research/glm53-flash-bringup-20260827/accrace-20260901/receipts/ab-$TAG"
mkdir -p "$OUT"
BIN=./target/debug/glm5-spec-ppn-gate
[ -x "$BIN" ] || { echo "missing $BIN — build it OUTSIDE the lock first"; exit 2; }
PRIME_REPS="${ACCRACE_PRIME_REPS:-8}"
WANT_PASS="${ACCRACE_WANT_PASS:-25}"
SPINNERS="${ACCRACE_SPINNERS:-6}"
QUOTA="${ACCRACE_CPUQUOTA:-400%}"
GPULOAD="${ACCRACE_GPULOAD:-1}"

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

SUM="$OUT/ab.txt"
: >"$SUM"
echo "# tag=$TAG stages=$STAGES reps=$REPS prime_reps=$PRIME_REPS want_pass=$WANT_PASS shared='${SHARED[*]-}' spinners=$SPINNERS quota=$QUOTA gpuload=$GPULOAD" >>"$SUM"
declare -A SWEEPF P0F P1F DEV DEV1
for a in FIX CTL; do SWEEPF[$a]=0; P0F[$a]=0; P1F[$a]=0; DEV[$a]=0; DEV1[$a]=0; done
for i in $(seq "$REPS"); do
  for a in FIX CTL; do
    if [ "$a" = FIX ]; then EV=MEMRA_PP_EXIT_PUBLISH=1; else EV=MEMRA_PP_EXIT_PUBLISH=0; fi
    log="$OUT/$a-rep$i.log"
    flock /tmp/memra-5090.lock "$INNER" "$log" "$SPINNERS" "$QUOTA" "$GPULOAD" "$BIN" \
      "$EV" ${SHARED[@]+"${SHARED[@]}"} timeout 900 nice -n 5 "$BIN" "$STAGES" 24 20 0 "$PRIME_REPS"
    rc=$?
    echo "exit=$rc" >>"$log"
    got="$(grep -cE 'gate PASS' "$log")"
    echo "pass_lines=$got want=$WANT_PASS" >>"$log"
    sw="$(grep -E 'gate (PASS|FAIL) \[E forced-rejection' "$log" | head -1)"
    p0="$(grep -E 'gate (PASS|FAIL) \[P0 prime-determinism' "$log" | head -1)"
    p1="$(grep -E 'gate (PASS|FAIL) \[P1 prime-determinism-post-spec' "$log" | head -1)"
    case "$sw" in *"gate PASS"*) SV=PASS ;; *"gate FAIL"*) SV=FAIL ;; *) SV=MISSING ;; esac
    case "$p0" in *"gate PASS"*) PV=PASS ;; *"gate FAIL"*) PV=FAIL ;; *) PV=MISSING ;; esac
    case "$p1" in *"gate PASS"*) QV=PASS ;; *"gate FAIL"*) QV=FAIL ;; *) QV=MISSING ;; esac
    acc="$(printf '%s' "$sw" | grep -oE '\([0-9]+/[0-9]+\)' | tail -1)"
    dev="$(printf '%s' "$p0" | grep -oE '[0-9]+ deviated' | grep -oE '^[0-9]+')"
    dev1="$(printf '%s' "$p1" | grep -oE '[0-9]+ deviated' | grep -oE '^[0-9]+')"
    [ "$SV" = PASS ] || SWEEPF[$a]=$(( ${SWEEPF[$a]} + 1 ))
    [ "$PV" = PASS ] || P0F[$a]=$(( ${P0F[$a]} + 1 ))
    [ "$QV" = PASS ] || P1F[$a]=$(( ${P1F[$a]} + 1 ))
    DEV[$a]=$(( ${DEV[$a]} + ${dev:-0} ))
    DEV1[$a]=$(( ${DEV1[$a]} + ${dev1:-0} ))
    printf 'rep%-3s %-3s exit=%-3s pass_lines=%-3s sweep=%-7s accepted=%-8s P0=%-7s dev0=%s/%s P1=%-7s dev1=%s/%s\n' \
      "$i" "$a" "$rc" "$got" "$SV" "${acc:-?}" "$PV" "${dev:-?}" "$PRIME_REPS" \
      "$QV" "${dev1:-?}" "$PRIME_REPS" | tee -a "$SUM"
  done
done
{
  echo "=== $TAG :: interleaved x$REPS, stages=$STAGES ==="
  for a in FIX CTL; do
    echo "$a  sweep FAILS=${SWEEPF[$a]}/$REPS   P0 FAILS=${P0F[$a]}/$REPS (dev ${DEV[$a]}/$(( REPS * PRIME_REPS )))   P1 FAILS=${P1F[$a]}/$REPS (dev ${DEV1[$a]}/$(( REPS * PRIME_REPS )))"
  done
} | tee -a "$SUM"
# The FIX arm must be perfect; the CTL arm must actually reproduce, or the window was too
# quiet to have tested anything and the run is not evidence.
# The FIX arm must be perfect on every signal; the CTL arm must actually reproduce on at
# least one, or the window was too quiet to have tested anything and this run is NOT evidence.
[ "${SWEEPF[FIX]}" -eq 0 ] && [ "${P0F[FIX]}" -eq 0 ] && [ "${P1F[FIX]}" -eq 0 ] \
  && { [ "${DEV1[CTL]}" -gt 0 ] || [ "${DEV[CTL]}" -gt 0 ] || [ "${SWEEPF[CTL]}" -gt 0 ]; }
