#!/usr/bin/env bash
# FLIP RE-BATTERY CELL 5 — PMIN tau ladder (count-based): DFlash2 K=3 with
# MEMRA_SPEC_PMIN in {0.3, 0.5, 0.7}. The loop-port port-2 gate is deliberately
# default-OFF because tau is a per-model measurement (q27 -1.9% at 0.3; step37 ships
# 0.5); this cell prices glm5's. Per tau: acceptance counts (stage-2 shape, greedy 128,
# both pools) + drafted/round + a timed row set for tok/s deltas (the caller holds the
# marker if it wants the timed rows to count; the c3 K=3 no-PMIN arm is the control).
# Greedy tapes across taus must be pairwise identical (truncation moves DRAFTS, never
# output) — compared at the end.
set -uo pipefail
OUT=/root/out-flip2/c5
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

rc=0
for TAU in 0.3 0.5 0.7; do
  name="tau$TAU"
  echo "######## C5 BOOT $name ########"
  /root/out-flip2/serve.sh start "c5-$name" "${DFL[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN="$TAU" \
    || { echo "C5_${name}_EXIT=BOOTFAIL"; rc=1; continue; }
  log=/root/out-flip2/logs/boot-c5-$name.log
  echo "confidence-gate boot receipt:"
  grep -m2 -iE 'confidence gate|PMIN' "$log" || echo "  (no confidence-gate boot line)"
  python3 /root/out-flip2/run_pool.py sample --out "$OUT/$name" || { echo "C5_${name}_EXIT=SAMPLEFAIL"; rc=1; continue; }
  python3 /root/out-flip2/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy --max-tokens 128 --k 3
  python3 /root/out-flip2/run_pool.py timed --out "$OUT/$name-timed" --max-tokens 256
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "C5_${name}_EXIT=0"
done

/root/out-flip2/serve.sh stop
echo "=== TAU CROSS-IDENTITY (greedy tapes must not move) ==="
python3 /root/out-flip2/run_pool.py compare --a "$OUT/tau0.3" --b "$OUT/tau0.5" || rc=1
python3 /root/out-flip2/run_pool.py compare --a "$OUT/tau0.5" --b "$OUT/tau0.7" || rc=1
echo "=== ACCEPTANCE TABLE ==="
python3 /root/out-flip2/run_pool.py agg --dirs "$OUT"/tau0.3 "$OUT"/tau0.5 "$OUT"/tau0.7
echo "=== DRAFTED/ROUND ==="
python3 - <<'PY'
import json, glob, os
for d in sorted(glob.glob("/root/out-flip2/c5/tau*")):
    if d.endswith("-timed"):
        continue
    mf = os.path.join(d, "meta-greedy.json")
    if not os.path.exists(mf):
        continue
    meta = json.load(open(mf))
    specs = [r["spec"] for r in meta["rows"] if r.get("spec") and not r.get("err")]
    drf = sum(s["drafted"] for s in specs); rnd = sum(s["rounds"] for s in specs)
    acc = sum(s["accepted"] for s in specs)
    print(f"{os.path.basename(d)}: rounds={rnd} drafted={drf} accepted={acc} "
          f"drafted/round={drf/rnd:.3f} acc/round={acc/rnd:.3f} accrate={acc/drf:.3f}")
PY
echo "=== LOOP-LAW SCREEN (c5 tapes) ==="
python3 /root/out-flip2/looplaw_screen.py "$OUT"/*/
echo "C5_ALL_DONE rc=$rc"
exit "$rc"
