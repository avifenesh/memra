#!/usr/bin/env bash
# STRUCT-BATTERY CELL 1 — D+H PRICING (the decision number, TIMED): ship config
# (DFlash2 + auto-K nopin + PMIN0.7, doors T/X/K/W at their DEFAULT ON, VERIFY_BATCH +
# HYPER_BATCH default ON) with doors D+H ON vs OFF, interleaved fresh boots x3 (x5 on
# anomaly — round indices as args). OFF is PINNED =0 on both flags (owner law: unset is
# not an OFF arm — here D and H are default OFF so unset==0, but the pin is explicit and
# printed so the receipt cannot be misread). Baseline: 70.458 tok/s (mv-battery c4
# winner, same env, same pools, doors T/X/K/W ON). Caller holds /root/TIMING-IN-FLIGHT.
# Greedy pool + one vendor-default sampled row per boot (run_pool timed), loop-law screen
# + cross-arm tape identity at the end (doors D and H carry rig bit gates: ANY tape
# divergence between arms is a defect finding and STOPS the window).
set -uo pipefail
OUT=/root/out-struct/c1
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
DH_ON=(MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1)
DH_OFF=(MEMRA_MOE_VROWS_DEV_TABLES=0 MEMRA_GLM5_HTOD_DIET=0)
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C1 BOOT $name ########"
  /root/out-struct/serve.sh start "c1-$name" "$@" || { echo "C1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-struct/run_pool.py sample --out "$OUT/$name" || { echo "C1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-struct/serve.sh engage "c1-$name" "$@" || { echo "C1_${name}_EXIT=ENGAGEFAIL"; return 1; }
  python3 /root/out-struct/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-struct/logs/boot-c1-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") vrows=$(grep -c '\[glm5-vrows\]' "$log") devtables=$(grep -c '\[moe-vrows-dev-tables\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  echo "C1_${name}_EXIT=0"
}

rc=0
for i in "$@"; do
  boot_and_time "dhoff-$i" "${DFL[@]}" "${DH_OFF[@]}" || rc=1
  boot_and_time "dhon-$i"  "${DFL[@]}" "${DH_ON[@]}"  || rc=1
done

/root/out-struct/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c1 tapes) ==="
python3 /root/out-struct/looplaw_screen.py "$OUT"/*/
echo "=== CROSS-ARM TAPE IDENTITY (D+H are bit gates on the rig; divergence = STOP) ==="
idrc=0
for i in "$@"; do
  python3 /root/out-struct/run_pool.py compare --a "$OUT/dhoff-1" --b "$OUT/dhon-$i" || idrc=1
done
echo "C1_IDENTITY_RC=$idrc"
echo "=== C1 D+H PRICE TABLE (baseline = ship config, D+H pinned =0) ==="
python3 /root/out-struct/mv_check.py --base "$OUT" --baseline dhoff --arms dhon
echo "C1_ROUNDS_DONE: $* rc=$rc idrc=$idrc"
