#!/usr/bin/env bash
# ONE ENTRYPOINT for the whole glm5 vision-on-ppN battery (arms A, B, C, D).
#
# Vision became a SHIP GATE alongside the cache (owner ruling 2026-09-01), and the slot is on the
# critical path, so this exists to spend zero slot time on orchestration: everything decidable
# before the window is decided before the window, and the run is one command.
#
# It REFUSES rather than guesses. Every input is either present and verified or the script stops
# with the reason named — the fixture pins, the boot commands, the readyz URL, the binary path.
# A battery that improvises on a critical-path slot is how a window produces an uninterpretable
# receipt.
#
# WHAT IT DOES NOT DO: claim a slot, deploy, or touch the live dark stack. It runs against a slot
# the launch agent has handed over, using boot/stop commands the operator supplies.
#
# usage:
#   GLM53_KEY=...                       # a keyring key for the slot
#   VISION_FIXTURES=<dir>               # the 5 sha-pinned request JSONs (launch lane branch)
#   BOOT_CMD_VISION='<cmd>'             # boots the slot with vision ARMED (door default)
#   BOOT_CMD_TEXTONLY='<cmd>'           # same with MEMRA_GLM5_VISION=0
#   BOOT_CMD_CONTROL='<cmd>'            # same as VISION plus MEMRA_VISION_OVERLAY_PUBLISH=0
#   STOP_CMD='<cmd>' READYZ=<url> SERVER_BIN=<path> GLM53_HOST=<hostname>
#   ./run-battery.sh <out-dir> [reps]
set -euo pipefail

OUT=${1:?output dir (e.g. ../receipts/box-$(date -u +%Y%m%dT%H%M%SZ))}
REPS=${2:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${GLM53_KEY:?}" "${BOOT_CMD_VISION:?}" "${BOOT_CMD_TEXTONLY:?}" "${BOOT_CMD_CONTROL:?}"
: "${STOP_CMD:?}" "${READYZ:?}" "${SERVER_BIN:?}" "${VISION_FIXTURES:?}"
# The endpoint is named, never inherited: the probe's default is the EDGE hostname over https, so
# an on-box loopback run that omits this boots the whole model and then fails every request.
: "${BASE:?BASE must name the endpoint, e.g. http://127.0.0.1:18893}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/battery.log") 2>&1
echo "=== glm5 vision-on-ppN battery — $(date -u +%FT%TZ)"
echo "out=$OUT reps=$REPS host=${GLM53_HOST:-glm53-api.tiyuvta.ai}"

# ---- PREFLIGHT, all of it before any box time is spent on a request -------------------------
echo
echo "--- preflight: the instrument, before the endpoint"
python3 "$HERE/verify-probe-refusals.py" "$VISION_FIXTURES" > "$OUT/preflight-refusals.txt" || {
  echo "REFUSE: the probe's own refusal paths did not all fire — the instrument is not trustworthy"
  exit 1; }
tail -1 "$OUT/preflight-refusals.txt"
echo "fixture dir: $VISION_FIXTURES"
sha256sum "$VISION_FIXTURES"/*.json | tee "$OUT/fixture-shas.txt" | sed 's/^/  /'

fail=0
note() { echo "$1" >> "$OUT/VERDICT.txt"; }
: > "$OUT/VERDICT.txt"

boot_and_probe() {  # arm-label boot-cmd probe-arm
  local label=$1 cmd=$2 arm=$3
  local nonce="${label}-$(date -u +%H%M%S)-$$"
  local log="$OUT/boot-$nonce.log"
  echo
  echo "--- $label: stop, clear, boot, probe (nonce=$nonce)"
  $STOP_CMD >>"$OUT/stop.log" 2>&1 || true
  local waited=0
  while pgrep -f "$SERVER_BIN" >/dev/null 2>&1; do
    sleep 2; waited=$((waited+2))
    [ $waited -ge 120 ] && { echo "REFUSE: previous server survived 120s of STOP_CMD"; exit 1; }
  done
  echo "BOOT_NONCE=$nonce arm=$label utc=$(date -u +%FT%TZ)" > "$log"
  ( eval "$cmd" ) >>"$log" 2>&1 &
  waited=0
  until curl -fsS "$READYZ" >/dev/null 2>&1; do
    sleep 3; waited=$((waited+3))
    [ $waited -ge 900 ] && { echo "REFUSE: $label never reached readyz in 900s"; tail -30 "$log"; exit 1; }
  done
  # Boot receipts that decide whether this window can testify at all.
  grep -m1 "^\[server\] build: " "$log" || { echo "REFUSE: no build-identity line"; exit 1; }
  grep -m1 "^\[server\] build: " "$log" | grep -q "id: source-tree" || {
    echo "REFUSE: build identity DEGRADED — its rows cannot back a published claim"; exit 1; }
  if [ "$label" != "text-only" ]; then
    grep -m1 "\[glm5-vision\] overlay intake:" "$log" || {
      echo "REFUSE: no overlay-intake line — binary predates the lane, or no glm5 model loaded"; exit 1; }
    grep -q "cross_context=true" "$log" || {
      echo "VOID: cross_context=false — this is NOT the ppN serving shape; no row here may be cited"
      exit 1; }
  fi
  BOOT_NONCE=$nonce VISION_FIXTURES=$VISION_FIXTURES \
    python3 "$HERE/probe-vision-ppn.py" --base "$BASE" --arm "$arm" --nonce "$nonce" --reps 1 \
      --out "$OUT/probe-$label" && local rc=0 || local rc=$?
  return ${rc:-0}
}

# ---- ARM A: the fix arm — exact codes on greedy AND vendor-default sampled -------------------
if boot_and_probe fix "$BOOT_CMD_VISION" fix; then
  note "ARM A (fix, default door): PASS — exact can't-hallucinate codes on greedy and vendor-default; negatives refused by name"
  grep -c "\[vision\] overlay published to the intake engine" "$OUT/boot-fix-"*.log \
    | sed 's/^/  publication receipts in the boot log: /' | tail -1
else
  note "ARM A (fix): FAIL — see probe-fix/receipts.json"; fail=1
fi

# ---- ARM B: the control — publish=0 must refuse at the waist, never 500 ----------------------
if boot_and_probe control "$BOOT_CMD_CONTROL" control; then
  note "ARM B (control, MEMRA_VISION_OVERLAY_PUBLISH=0): PASS — image requests refuse at the HTTP waist (named 4xx), not a mid-prefill 500"
else
  note "ARM B (control): FAIL — a control that does not reproduce means arm A proved nothing about this lane's code"; fail=1
fi

# ---- ARM C + D: text-only identity and the interleaved no-tax rows ---------------------------
echo
echo "--- arms C+D: boot-interleaved decode rows and the text-only byte-identity twin"
if BOOT_CMD_VISION="$BOOT_CMD_VISION" BOOT_CMD_TEXTONLY="$BOOT_CMD_TEXTONLY" \
   STOP_CMD="$STOP_CMD" READYZ="$READYZ" SERVER_BIN="$SERVER_BIN" \
   VISION_FIXTURES="$VISION_FIXTURES" GLM53_KEY="$GLM53_KEY" BASE="$BASE" \
   "$HERE/interleave.sh" "$REPS" "$OUT/armD"; then
  note "ARM D (no decode tax, boot-interleaved): see armD/ARM-D-VERDICT.txt"
  grep -E "^VERDICT" -A3 "$OUT/armD/ARM-D-VERDICT.txt" | sed 's/^/  /'
else
  note "ARM D: FAIL or refused as non-evidence — see armD/ARM-D-VERDICT.txt"; fail=1
fi

# Arm C: the vision-armed and text-only greedy completions must be byte-identical.
A=$(ls -1 "$OUT/probe-fix/text-greedy.txt" 2>/dev/null | head -1)
B=$(ls -1 "$OUT"/armD/probe-text-only-*/text-greedy.txt 2>/dev/null | head -1)
# NON-EMPTY FIRST, THEN EQUAL. Two EMPTY files compare equal, so a pair of failed requests — the
# slot is key-authed and a keyless call 401s — would render as "byte identity" and pass this arm
# on nothing. Same shape as the e3b0c442… empty-input sha near-miss banked in darklanes#11: a
# comparison over piped/absent bytes asserts the byte count, or it asserts nothing.
if [ -n "$A" ] && [ -n "$B" ] && [ ! -s "$A" ]; then
  note "ARM C: FAIL — the vision-armed greedy completion is EMPTY ($A). Two empty files compare equal, so this is a refusal, not identity (check auth: the slot is key-authed)"
  fail=1
elif [ -n "$A" ] && [ -n "$B" ] && [ ! -s "$B" ]; then
  note "ARM C: FAIL — the text-only greedy completion is EMPTY ($B); see above"
  fail=1
elif [ -n "$A" ] && [ -n "$B" ]; then
  if cmp -s "$A" "$B"; then
    note "ARM C (text-only byte identity with vision armed): PASS — identical NON-EMPTY bytes ($(wc -c <"$A") bytes, sha $(sha256sum "$A" | cut -c1-16))"
  else
    note "ARM C: FAIL — a vision feature moved a text token; diff banked as armC.diff"
    diff "$A" "$B" > "$OUT/armC.diff" || true; fail=1
  fi
else
  note "ARM C: INCONCLUSIVE — one side's greedy text row is missing (A=${A:-none} B=${B:-none})"; fail=1
fi

$STOP_CMD >>"$OUT/stop.log" 2>&1 || true
echo
echo "=== VERDICT"
cat "$OUT/VERDICT.txt"
if [ $fail -eq 0 ]; then
  echo "BATTERY GREEN — report to the coordinator as GREEN with $OUT"
else
  echo "BATTERY NOT GREEN — report the findings list, never a partial green"
fi
exit $fail
