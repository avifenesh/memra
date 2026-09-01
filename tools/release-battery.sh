#!/usr/bin/env bash
# release-battery.sh — the GPU release gate, run on the rig before tagging.
#
# CI is compile-only. This is the only thing that proves kernels compute right answers,
# and until 2026-08-28 it was three prose commands in docs/RELEASING.md whose model list
# read `<each affected model>`. That phrase is what let v0.118.0 ship without running
# ornith-1.5-35b-a3b: the change touched no kernel, the GGUF was not on the rig, and
# "affected" absorbed both facts. The roster is now a FILE and a missing `own` model is a
# REFUSAL, because a gate you satisfy by not having the file is not a gate.
#
#   tools/release-battery.sh [--roster FILE] [--allow-missing-vendor]
#
# Exit 0 only when every required arm PASSED. Prints a receipt block for the tag message.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BIN=$ROOT/target/release
ROSTER=$HERE/release-roster.tsv
ALLOW_MISSING_VENDOR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --roster) ROSTER=${2:?}; shift 2 ;;
    --allow-missing-vendor) ALLOW_MISSING_VENDOR=1; shift ;;
    -h|--help) sed -n '1,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# argmax-margin-gate.sh answers "SKIP (build target/release/argmax-margin-probe to enable)"
# and exits 0 when its probe is absent — a gate that disables itself on a missing dependency.
# Require the probe HERE so a release can never be gated by a skip. (Observed 2026-08-28: the
# first calibrated run reported SKIP for both models and only failed because this script
# refuses to read a skip as a pass.)
for t in kernel-check run-spec argmax-margin-probe; do
  [ -x "$BIN/$t" ] || { echo "release-battery: $BIN/$t missing — cargo build --release --bins first" >&2; exit 1; }
done
[ -x "$HERE/argmax-margin-gate.sh" ] || { echo "release-battery: tools/argmax-margin-gate.sh missing" >&2; exit 1; }
[ -r "$ROSTER" ] || { echo "release-battery: roster not readable: $ROSTER" >&2; exit 1; }

FAILED=0; MISSING_OWN=0; OWN_ROWS=0; LINES=""
note() { LINES="${LINES}$1"$'\n'; printf '%s\n' "$1"; }

# Every roster loop uses `read ... || [ -n "$class" ]`: plain `read` returns non-zero on a
# final line with NO trailing newline, so the body never runs for it and the row vanishes
# silently — own rows included. That is this lane's shape again (satisfied by not having the
# ROW), reachable from a one-byte edit artifact rather than an intent. (revuto, PR #63.)
#
# STRUCTURE FIRST, before anything loads a 20GB model. Two reasons: a roster defect is
# cheap to detect and expensive to discover after a full battery, and the own-row rule below
# must not be reachable only via a path that already spent the GPU.
#
# THE ROW ITSELF IS NOT SKIPPABLE. Enforcing `own` per row still let a roster with ZERO own
# rows pass: point --roster at a vendor-only file, or comment out the own row, and the run
# exits 0 with a receipt that simply has no own arm. Same shape this gate exists to close --
# satisfied by not having the ROW instead of not having the FILE -- and reachable without
# editing the repo. (revuto, PR #63.)
while IFS=$'\t' read -r class id path _ || [ -n "${class:-}" ]; do
  case "$class" in ''|\#*) continue ;; esac
  case "$class" in
    own) OWN_ROWS=$((OWN_ROWS + 1)) ;;
    vendor) ;;
    *) echo "release-battery: unknown roster class $(printf '%q' "$class") for $id (own|vendor)" >&2; exit 1 ;;
  esac
done < "$ROSTER"
if [ "$OWN_ROWS" -eq 0 ]; then
  echo "REFUSED: $ROSTER contains no \`own\` model — an own row is not skippable." >&2
  exit 1
fi

# kernel-check runs once, on the largest present roster model: it exercises kernels, not
# a model's weights, so one pass covers the arch. It still needs a real GGUF to load.
KC_MODEL=""
while IFS=$'\t' read -r class id path _ || [ -n "${class:-}" ]; do
  case "$class" in ''|\#*) continue ;; esac
  [ -f "$path" ] && { KC_MODEL=$path; break; }
done < "$ROSTER"
if [ -z "$KC_MODEL" ]; then
  note "kernel-check           REFUSED   no roster model present on this rig"
  FAILED=1
else
  if OUT=$("$BIN/kernel-check" "$KC_MODEL" 2>&1) && printf '%s' "$OUT" | grep -q "ALL GREEN"; then
    note "kernel-check           PASS      $(printf '%s' "$OUT" | grep -o 'ALL GREEN.*') [$(basename "$KC_MODEL")]"
  else
    note "kernel-check           FAIL      $(printf '%s' "$OUT" | tail -1)"
    FAILED=1
  fi
fi

while IFS=$'\t' read -r class id path _ || [ -n "${class:-}" ]; do
  case "$class" in ''|\#*) continue ;; esac
  if [ ! -f "$path" ]; then
    if [ "$class" = own ]; then
      note "$id  REFUSED   OWN model absent from this rig: $path"
      MISSING_OWN=1; FAILED=1
    elif [ "$ALLOW_MISSING_VENDOR" = 1 ]; then
      note "$id  SKIPPED   vendor model absent (--allow-missing-vendor)"
    else
      note "$id  REFUSED   vendor model absent: $path (pass --allow-missing-vendor to accept)"
      FAILED=1
    fi
    continue
  fi
  # CALIBRATED, not raw run-gen. run-gen carries a hard prefill-vs-decode argmax assert
  # whose wording ("cache threading bug") is a documented landmine: batched prefill and the
  # tokenwise decode loop are two legitimate arithmetics, so a NEAR-TIE position legitimately
  # flips between them. tools/argmax-margin-gate.sh judges each flip against the prompt's own
  # distribution and passes when the config spread covers the margin.
  # 2026-08-28: the first roster run called raw run-gen and reported ornith-1.5-35b-a3b FAIL
  # on a position whose prefill top-2 margin was 0.0256 against a config spread of 0.5540 —
  # i.e. it was about to file a defect against our own model that the repo already documents
  # as not-a-defect. The calibrated gate returns flips=0 bad=0 PASS on the same model.
  if OUT=$("$HERE/argmax-margin-gate.sh" "$path" 2>&1) && printf '%s' "$OUT" | grep -q "^  PASS:"; then
    note "$id  argmax-margin  PASS  $(printf '%s' "$OUT" | grep -m1 -o 'SUMMARY flips=[0-9]* bad=[0-9]*')"
  else
    if printf '%s' "$OUT" | grep -q "SKIP"; then
      note "$id  argmax-margin  REFUSED   gate skipped itself: $(printf '%s' "$OUT" | grep -m1 -o 'SKIP.*')"
    else
      note "$id  argmax-margin  FAIL  $(printf '%s' "$OUT" | tail -1)"
    fi
    FAILED=1
  fi
  if OUT=$("$BIN/run-spec" "$path" 2>&1) && printf '%s' "$OUT" | grep -q "SELF-CONSISTENCY PASS"; then
    note "$id  run-spec  PASS      K=1..8 self-consistency, identical to plain target"
  else
    note "$id  run-spec  FAIL      $(printf '%s' "$OUT" | tail -1)"
    FAILED=1
  fi
done < "$ROSTER"

echo
if [ "$FAILED" = 0 ]; then
  echo "=== RELEASE BATTERY PASS — paste into the tag message ==="
  echo "  roster: $ROSTER ($OWN_ROWS own)"
  printf '%s' "$LINES" | sed 's/^/  /'
  exit 0
fi
[ "$MISSING_OWN" = 1 ] && echo "REFUSED: an OWN model was not tested (absent file, or no own row in $ROSTER). Neither is skippable." >&2
echo "RELEASE BATTERY FAILED" >&2
exit 1
