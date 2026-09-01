#!/usr/bin/env bash
# Arm D — NO DECODE TAX, boot-level interleaved (coordinator condition, 2026-09-01:
# "take your decode rows INTERLEAVED against the text-only arm in the SAME boot sequence,
#  since the whole claim is 'no tax' and a cross-boot comparison can't carry that").
#
# WHY BOOT-LEVEL AND NOT REQUEST-LEVEL. The two arms differ by a BOOT env
# (`MEMRA_GLM5_VISION` armed vs `=0`), and two 3-card resident stacks do not fit on the same
# three cards, so the arms cannot be alive at once. The interleave therefore alternates BOOTS
# inside ONE contiguous window — V,T,V,T,V,T — instead of running all of one arm and then all of
# the other. That is the interleaved-A/B law's own remedy for the thing it exists to stop: box
# clock drift, thermal drift and any slow background change land on BOTH arms equally instead of
# on whichever ran second.
#
# ARM IDENTITY, and the honest limit of it (ab-arm-identity-not-liveness: a health 200 proves a
# listener, not WHICH server). memra has no per-boot nonce surface — `system_fingerprint` is the
# BUILD sha, identical in both arms by design, which is itself worth asserting (the arms must
# differ by env only, never by binary). So identity is constructed from what exists:
#   1. pgrep-clear BEFORE each boot: no server process on this slot's binary may survive;
#   2. the wrapper writes BOOT_NONCE as the first line of a per-boot log, and the probe stamps
#      the SAME nonce onto every row it emits;
#   3. the PID is recorded after readyz and asserted to have started after the boot command;
#   4. the boot log must carry this arm's own distinguishing vision lines (armed vs not).
# A row is attributable to an arm because (1)-(4) hold together, not because a port answered.
#
# usage:
#   GLM53_KEY=... VISION_FIXTURES=<dir> \
#   BOOT_CMD_VISION='<command that boots the vision-armed slot, backgrounded>' \
#   BOOT_CMD_TEXTONLY='<same with MEMRA_GLM5_VISION=0>' \
#   STOP_CMD='<command that stops the slot>' READYZ=http://127.0.0.1:PORT/readyz \
#   SERVER_BIN=/data/glm53/bin/memra-server.<sha> \
#   ./interleave.sh <reps> <out-dir>
#
# BOOT_CMD_* / STOP_CMD are REQUIRED and have no defaults: guessing a boot command on a box that
# is serving something else is exactly the class of mistake this lane's receipts cannot absorb.
set -euo pipefail

REPS=${1:?reps (3 is the banked shape)}
OUT=${2:?output dir}
: "${GLM53_KEY:?}" "${BOOT_CMD_VISION:?}" "${BOOT_CMD_TEXTONLY:?}" "${STOP_CMD:?}" "${READYZ:?}" "${SERVER_BIN:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE=$HERE/probe-vision-ppn.py
ROWS=$OUT/decode-rows.jsonl
mkdir -p "$OUT"
: > "$ROWS"

pgrep_clear() {
  local waited=0
  $STOP_CMD >>"$OUT/stop.log" 2>&1 || true
  while pgrep -f "$SERVER_BIN" >/dev/null 2>&1; do
    sleep 2; waited=$((waited+2))
    if [ $waited -ge 120 ]; then
      echo "REFUSE: a server on $SERVER_BIN survived 120s of STOP_CMD — an arm cannot be" \
           "attributed while the previous boot is alive" >&2
      pgrep -af "$SERVER_BIN" >&2 || true
      exit 1
    fi
  done
}

boot() {
  local arm=$1 nonce=$2 log=$3 cmd
  case $arm in
    fix) cmd=$BOOT_CMD_VISION ;;
    text-only) cmd=$BOOT_CMD_TEXTONLY ;;
    *) echo "REFUSE: unknown arm $arm" >&2; exit 1 ;;
  esac
  echo "BOOT_NONCE=$nonce arm=$arm utc=$(date -u +%FT%TZ)" > "$log"
  local t0; t0=$(date +%s)
  ( eval "$cmd" ) >>"$log" 2>&1 &
  local waited=0
  until curl -fsS "$READYZ" >/dev/null 2>&1; do
    sleep 3; waited=$((waited+3))
    if [ $waited -ge 900 ]; then
      echo "REFUSE: $arm boot ($nonce) never reached readyz in 900s" >&2
      tail -40 "$log" >&2; exit 1
    fi
  done
  local pid; pid=$(pgrep -f "$SERVER_BIN" | head -1)
  [ -n "$pid" ] || { echo "REFUSE: readyz answered but no $SERVER_BIN process exists — the" \
                          "listener is not the server this battery thinks it is" >&2; exit 1; }
  # Started AFTER the boot command: a stale process that happens to answer readyz is the exact
  # failure this asserts against.
  #
  # `ps -o etimes=` (elapsed SECONDS), not `ps -o lsstart=` + `date -d`. Measured on the rig
  # 2026-09-01 before this ever ran on a box: this procps build has no `lsstart` specifier, and
  # `date -d ""` on the empty result returns TODAY AT MIDNIGHT, which made the comparison PASS
  # spuriously — a false-green identity assertion, the very thing this line exists to prevent.
  # etimes is refused if it is not a number, so a missing primitive fails LOUD instead of green.
  local age; age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  case $age in
    ''|*[!0-9]*)
      echo "REFUSE: cannot read pid $pid elapsed time ('ps -o etimes=' gave ${age:-<empty>});" \
           "arm identity would be unverified, and an unverified identity is not evidence" >&2
      exit 1 ;;
  esac
  local boot_elapsed=$(( $(date +%s) - t0 ))
  if [ "$age" -gt $((boot_elapsed + 5)) ]; then
    echo "REFUSE: pid $pid is ${age}s old but this boot command started only ${boot_elapsed}s" \
         "ago — readyz is being answered by a PREVIOUS server and rows would be attributed to" \
         "the wrong arm" >&2; exit 1
  fi
  # BUILD IDENTITY, from the log rather than inferred (main's lane/real-system-fingerprint,
  # 2026-09-01): every boot prints `[server] build: memra-<ver>-<id> (id: source-tree, git: ..)`.
  # Two things are asserted with it, because an A/B's arms must differ by ENV ONLY:
  #   - the line exists and the id is NOT degraded. A degraded id is version-only and does not
  #     identify the source tree it was compiled from, so its rows cannot back a published claim
  #     (darklanes check-claim-builds --live) — and these rows are meant to move the modality fact.
  #   - every boot in the window carries the IDENTICAL line. A rebuild mid-window would otherwise
  #     silently turn this into a build-vs-build comparison wearing a flag's name.
  local build_line; build_line=$(grep -m1 "^\[server\] build: " "$log" || true)
  [ -n "$build_line" ] || { echo "REFUSE: boot ($nonce) printed no build-identity line — the" \
      "binary predates lane/real-system-fingerprint and its rows cannot be attributed to a" \
      "source tree" >&2; exit 1; }
  case $build_line in
    *"id: source-tree"*) : ;;
    *) echo "REFUSE: boot ($nonce) build identity is DEGRADED ($build_line): a version-only id" \
            "does not identify the source this binary came from, so no row from it may back a" \
            "published claim. Rebuild where the workspace source tree is readable." >&2
       exit 1 ;;
  esac
  if [ -s "$OUT/build-identity.txt" ]; then
    if [ "$build_line" != "$(cat "$OUT/build-identity.txt")" ]; then
      echo "REFUSE: build identity CHANGED mid-window." >&2
      echo "  first: $(cat "$OUT/build-identity.txt")" >&2
      echo "  now:   $build_line" >&2
      echo "  An A/B whose arms differ by BUILD cannot testify about a flag." >&2
      exit 1
    fi
  else
    printf '%s' "$build_line" > "$OUT/build-identity.txt"
  fi

  # Arm-distinguishing boot evidence. The fix arm must show a servable cross-context intake;
  # the text-only arm must show the tower was never loaded.
  case $arm in
    fix)
      grep -q "\[glm5-vision\] overlay intake:" "$log" || {
        echo "REFUSE: fix boot ($nonce) printed no overlay-intake line — either the binary" \
             "predates lane/glm53-vision-ppn or no glm5 model loaded" >&2; exit 1; }
      grep -q "servable=true" "$log" || {
        echo "REFUSE: fix boot ($nonce) reports servable=false; check the door" >&2
        grep "\[glm5-vision\]" "$log" >&2; exit 1; }
      grep -q "cross_context=true" "$log" || {
        echo "VOID: fix boot ($nonce) reports cross_context=false — this is NOT the shape the" \
             "battery claims to test (tower and intake share a context), so no row from this" \
             "window may be cited for the ppN serving shape" >&2; exit 1; }
      ;;
    text-only)
      grep -q "MEMRA_GLM5_VISION=0" "$log" || {
        echo "REFUSE: text-only boot ($nonce) shows no vision-off line" >&2; exit 1; }
      ;;
  esac
  echo "$pid"
}

echo "arm D: boot-level interleave, $REPS reps per arm, one contiguous window"
for rep in $(seq 1 "$REPS"); do
  for arm in fix text-only; do
    nonce="${arm}-r${rep}-$(date -u +%H%M%S)-$$"
    log=$OUT/boot-$nonce.log
    pgrep_clear
    pid=$(boot "$arm" "$nonce" "$log")
    echo "[boot] arm=$arm rep=$rep nonce=$nonce pid=$pid"
    # The image arms run on the FIRST fix boot only: they are correctness, not perf, and
    # re-running them per rep would spend window time without adding evidence.
    probe_arm=$arm
    if [ "$arm" = "fix" ] && [ "$rep" != "1" ]; then probe_arm=text-only; fi
    # ^ rep>1 fix boots take the text-only probe PATH (no image requests) while remaining the
    #   vision-ARMED server: the decode row is what this rep is for. The row's own `arm` field
    #   is forced below so it is never mislabelled.
    VISION_FIXTURES=${VISION_FIXTURES:-} BOOT_NONCE=$nonce \
      python3 "$PROBE" --arm "$probe_arm" --nonce "$nonce" --reps 1 \
        --rows-jsonl "$ROWS.raw" --out "$OUT/probe-$nonce" || {
          echo "REFUSE: probe failed on arm=$arm rep=$rep (see $OUT/probe-$nonce)" >&2; exit 1; }
    python3 - "$ROWS.raw" "$ROWS" "$arm" <<'PY'
import json, sys
src, dst, arm = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(src)]
with open(dst, "a") as f:
    for r in rows:
        r["arm"] = arm          # the SERVER's arm, not the probe's request path
        f.write(json.dumps(r) + "\n")
open(src, "w").close()
PY
  done
done
pgrep_clear

python3 - "$ROWS" "$OUT/ARM-D-VERDICT.txt" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
out = open(sys.argv[2], "w")
def w(s=""): print(s); out.write(s + "\n")
w("arm D — no decode tax, boot-level interleaved (vendor-default sampled, tok/s from the")
w("server's own usage.elapsed_s). Rows in decode-rows.jsonl carry arm + rep + boot nonce.")
w()
fps = {r.get("fingerprint") for r in rows}
w(f"system_fingerprint across all rows: {fps}")
if len(fps) != 1:
    w("REFUSED as evidence: the arms did not run the same binary — an A/B whose arms differ by")
    w("build cannot testify about a flag.")
    sys.exit(1)
by = {}
for r in rows:
    by.setdefault(r["arm"], []).append(r)
for arm in sorted(by):
    tps = [r["tok_s"] for r in by[arm] if r["status"] == 200 and r["completion_tokens"] > 0]
    spec = all((r.get("spec") or {}).get("rounds", 0) > 0 for r in by[arm] if r["status"] == 200)
    w(f"{arm:10s} n={len(tps)} rows={tps} median={statistics.median(tps):.2f} "
      f"min={min(tps):.2f} max={max(tps):.2f} spec_engaged={spec}")
if len(by) != 2:
    w("REFUSED as evidence: only one arm produced rows.")
    sys.exit(1)
v = [r["tok_s"] for r in by["fix"] if r["status"] == 200]
t = [r["tok_s"] for r in by["text-only"] if r["status"] == 200]
mv, mt = statistics.median(v), statistics.median(t)
delta = (mv - mt) / mt * 100
span_t = (max(t) - min(t)) / mt * 100
span_v = (max(v) - min(v)) / mv * 100
w()
w(f"vision-armed median {mv:.2f} vs text-only median {mt:.2f} -> {delta:+.2f}%")
w(f"within-arm spread: vision {span_v:.2f}%, text-only {span_t:.2f}%")
w()
if abs(delta) <= max(span_v, span_t):
    w("VERDICT: NO TAX — the between-arm gap sits inside the arms' own spread. Read this as")
    w("'no measurable tax at this rep count', never as a performance claim: there is also no")
    w("MECHANISM for one (publication is a single host round trip per SESSION, in prefill; a")
    w("text request never builds an overlay and decode never touches one).")
else:
    w("VERDICT: GAP EXCEEDS SPREAD — do not ship. Either a real tax exists or the window was")
    w("perturbed; the confounder to rule out FIRST is the tower's own ~2.1 GiB on the primary")
    w("card changing the MoE resident-experts decision. Compare the '[moe] resident-experts")
    w("decision' line across the two arms' boot logs before reading the number as a tax.")
PY
echo
echo "banked: $OUT (boot logs, per-boot probe receipts, decode-rows.jsonl, ARM-D-VERDICT.txt)"
