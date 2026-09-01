#!/usr/bin/env bash
# glm5-tp-gate runner for the TRANSPORT lane (lane/glm5-tp-transport).
#
# One invocation = the full arm matrix INCLUDING the transport arms X0/X1/X2/X3/XT/XF. The
# binary owns its own env discipline: it PINS MEMRA_GLM5_TP_TRANSPORT=0 for the whole banked
# battery, asserts the movement census is flat on that pin, then runs the peer-pull twins and
# the direct transport-vs-transport byte-identity arm. No knob matrix is needed here, and none
# should be added — a knob set from outside would silently redefine the pinned arm.
#
# Rig law: exactness only (LAW:rig-exactness-only). No timing number is read out of this
# script. The rig is one card, so the peer rank is a second CUDA context on that card
# (MEMRA_GLM5_TP_GATE_SAME_DEV=1): these arms prove the peer-pull program is BIT-PRESERVING
# and its counters non-vacuous, and prove NOTHING about a real PCIe fabric. The fabric is the
# box window's arm — see LANE.md stage 4, and run HEALTH.sh there before anything else.
set -u
BIN=${BIN:-./target/release/glm5-tp-gate}
OUT=${OUT:-research/glm53-flash-bringup-20260827/tp-transport-20260901/gates}
P=${P:-16}
N=${N:-12}
mkdir -p "$OUT"

LOG="$OUT/01-tp-gate-transport-p${P}-n${N}.log"
echo "########## glm5-tp-gate (transport arms) P=$P N=$N ##########"
# WRITE TO A TEMP, THEN MOVE. `... | tee "$LOG"` truncates $LOG the instant the pipeline is
# built, which is BEFORE flock returns — so a queued re-run on a busy rig destroys the
# previously banked log and leaves a zero-byte receipt for however long it waits for the lock.
# Caught exactly that way on 2026-09-01 (the banked 237-line log went to 0 bytes while this
# script sat behind another lane's rig lock). A receipt file must never be shorter than the last
# run that produced it.
TMP=$(mktemp "${TMPDIR:-/tmp}/glm5-tp-gate-XXXXXX.log")
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  timeout 2400 "$BIN" "$P" "$N" 2>&1 | grep -v '^\[loader-law\]' \
  | tee "$TMP"
rc=${PIPESTATUS[0]}
echo "exit=$rc" | tee -a "$TMP"
mv -f "$TMP" "$LOG"

# The receipt lines a reviewer greps, extracted so the verdict is one file.
{
  echo "--- transport announces ---"
  grep -E '^\[glm5-tp-transport\]' "$LOG" || true
  echo "--- seam transport= strings (all four seams must name the LIVE arm) ---"
  # The seam announces WRAP, so match the transport= token on its own line and keep the
  # preceding marker line with it. -A/-B rather than a single -oE pattern: an -oE that assumed
  # one line silently printed NOTHING on the first run of this script, which is the
  # loud-failures-fail-quietly shape.
  grep -nE 'transport=(host-canonical|peer-pull)' \
    "$LOG" | sed 's/  */ /g' || true
  echo "--- transport verdicts (X arms) ---"
  grep -E 'glm5-tp-gate (PASS|FAIL): \[X' "$LOG" || true
  echo "--- movement census lines ---"
  grep -E 'census transport=' "$LOG" || true
} | tee "$OUT/02-transport-receipts.txt"

exit "$rc"
