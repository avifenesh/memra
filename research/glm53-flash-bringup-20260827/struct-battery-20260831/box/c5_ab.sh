#!/usr/bin/env bash
# STRUCT-BATTERY CELL 5 — THE PLACEMENT A/B (first of fleet; ep-place LANE §4 steps 3+4,
# amended): naive even split vs the measured coactivation map on the TP-2 pair.
# The SERVED path refuses MEMRA_GLM5_TP by design (v1), so this is the ENGINE-TWIN A/B —
# the tp2-battery harness, whose banked instrument trap (engine twins under-read served;
# single-root walks ~0.8x class) survives because BOTH arms ride the SAME instrument and
# the deliverable is the RELATIVE delta only. Doors T/X/K/W ride at their defaults in
# both arms identically (engine twin, plain decode walk).
# Phase A (untimed): map-vs-even identity spot on the real artifact — even tape reference,
#   map arm TEACHER-FORCED on it. Rig arm M proved decode BYTE-IDENTICAL under a skewed
#   map (correctness is placement-independent by construction); any real-artifact decode
#   divergence is measured, compared against the banked tp2 bars (green 5.2e-2 class /
#   red 1.4e2 class) and STOPS the timed phase unless orders below red.
# Phase B (TIMED, caller holds /root/TIMING-IN-FLIGHT): interleaved x3 fresh boots per
#   arm, decode pool, BOXP_SAMPLED=1 (one vendor-default sampled row per run) +
#   BOXP_MAX_NEW=256. x5 on anomaly per the escalation rules in agg output.
# Usage: c5_ab.sh identity | timed <round-idx...> | report
set -uo pipefail
OUT=/root/out-struct
cd /root/out-struct

case "${1:?identity|timed|report}" in
identity)
  bash "$OUT/probe_arm.sh" even tape "$OUT/prompts-c1" abref || { echo "C5_IDENTITY=REF_FAIL"; exit 1; }
  BOXP_FORCE_DIR="$OUT/even-tape-abref" bash "$OUT/probe_arm.sh" map tape "$OUT/prompts-c1" abmap \
    || { echo "C5_IDENTITY=MAP_FAIL"; exit 1; }
  echo "=== C5 IDENTITY COMPARE (bar: decode steps byte-identical, per rig arm M) ==="
  python3 "$OUT/compare.py" "$OUT/even-tape-abref" "$OUT/map-tape-abmap"
  rc=$?
  echo "C5_IDENTITY_RC=$rc (0 = byte-identical/in-band; nonzero = MEASURE against the tp2 bars before any timed run)"
  exit $rc
  ;;
timed)
  shift
  rc=0
  # TRAP (found in this window, receipted): probe_arm.sh SCRUBS inherited BOXP_* env,
  # so BOXP_SAMPLED/BOXP_MAX_NEW must ride as TRAILING ARGS (probe_arm.sh "$@" extras),
  # never as env-prefix assignments — the x3 rounds of 2026-08-31 ran greedy-only at
  # max_new=200 both arms identically (relative delta unaffected); t4v was the repair.
  # The tp2-battery RUNBOOK's env-prefix spelling carries the same trap.
  for i in "$@"; do
    bash "$OUT/probe_arm.sh" even timed "$OUT/prompts-decode" "t$i" BOXP_SAMPLED=1 BOXP_MAX_NEW=256 || rc=1
    bash "$OUT/probe_arm.sh" map  timed "$OUT/prompts-decode" "t$i" BOXP_SAMPLED=1 BOXP_MAX_NEW=256 || rc=1
  done
  echo "C5_TIMED_ROUNDS_DONE: $* rc=$rc"
  ;;
report)
  echo "=== C5 AGGREGATE (medians, 128-token floor named) ==="
  python3 "$OUT/agg.py" "$OUT"/even-timed-* "$OUT"/map-timed-*
  echo "=== C5 LOOP-LAW SCREEN ==="
  python3 "$OUT/looplaw_screen.py" "$OUT"/even-timed-* "$OUT"/map-timed-*
  echo "=== C5 RELATIVE DELTA + COUNTER RECONCILIATION ==="
  python3 - "$OUT" <<'PY'
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]
def runs(arm):
    meds, peers = [], []
    for d in sorted(glob.glob(os.path.join(out, f"{arm}-timed-t*"))):
        rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
        ok = [r for r in rows if r["arm"] == "greedy" and r["out_tokens"] >= 128]
        if ok:
            meds.append(st.median(r["decode_tok_s"] for r in ok))
        log = os.path.join(out, "logs", f"probe-{os.path.basename(d)}.log")
        m = re.findall(r"ep-peer-slot-dispatches=(\d+)", open(log).read())
        if m:
            peers.append(int(m[-1]))
    return meds, peers
em, ep = runs("even")
mm, mp = runs("map")
if em and mm:
    me, mmed = st.median(em), st.median(mm)
    spread_e = (max(em) - min(em)) / me if len(em) > 1 else 0.0
    spread_m = (max(mm) - min(mm)) / mmed if len(mm) > 1 else 0.0
    pooled = max(max(em) - min(em) if len(em) > 1 else 0, max(mm) - min(mm) if len(mm) > 1 else 0)
    gap = mmed - me
    print(f"even  boot medians {[round(x,3) for x in em]} median {me:.3f} spread {100*spread_e:.3f}%")
    print(f"map   boot medians {[round(x,3) for x in mm]} median {mmed:.3f} spread {100*spread_m:.3f}%")
    print(f"RELATIVE DELTA map/even = {mmed/me:.4f} (gap {gap:+.3f} tok/s, pooled spread {pooled:.3f})")
    if max(spread_e, spread_m) > 0.005:
        print("ESCALATE_TO_X5: RULE(a) within-arm spread > 0.5%")
    if len(em) > 1 and len(mm) > 1 and abs(gap) <= 2 * pooled:
        print("ESCALATE_TO_X5: RULE(b) |gap| <= 2x pooled spread (too close)")
if ep and mp:
    pe, pm = st.median(ep), st.median(mp)
    print(f"peer-slot dispatches: even {ep} median {pe:.0f} | map {mp} median {pm:.0f} "
          f"| map/even = {pm/pe:.4f}")
    print("reconciliation: the mint's per-layer peer_touch predicts the peer-slot share; "
          "compare map/even dispatch ratio against (map peer_touch / even peer_touch) "
          "from maps/mint-stats-summary.txt")
PY
  ;;
esac
