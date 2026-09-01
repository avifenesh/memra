#!/usr/bin/env bash
# MV-DOORS CELL 1 — byte-identity spot, THE STOP BAR (LANE.md §7): composed-doors greedy
# tapes vs no-doors on the ship spec shape (DFlash2, pinned K=3, PMIN0.7), 4 prompts
# spanning both pools incl. the rejection-heavy d02 row. All five doors carry rig bit
# gates, so ANY divergence here is a DEFECT, not a numeric class — it STOPS the window.
set -uo pipefail
OUT=/root/out-mv/c1
TAGS=d00-code,d02-code,d06-prose,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 MEMRA_SPEC_K=3)
DOORS=(MEMRA_BF16_TCOLS_WIDE=1 MEMRA_BF16_TCOLS_X1=1 MEMRA_MOE_VROWS_PACK=1 MEMRA_TOPK_SHARDS=1 MEMRA_GLM5_VERIFY_WS=1)
mkdir -p "$OUT"

run_arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C1 BOOT $name ########"
  /root/out-mv/serve.sh start "c1-$name" "$@" || { echo "C1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py sample --out "$OUT/$name" || { echo "C1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-mv/serve.sh doors "c1-$name" "$@" || { echo "C1_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$TAGS" --k 3
  echo "C1_${name}_EXIT=0"
}

run_arm nodoors "${DFL[@]}" || exit 1
run_arm doors "${DFL[@]}" "${DOORS[@]}" || exit 1
echo "=== C1 IDENTITY doors vs nodoors (STOP bar) ==="
rc=0
python3 /root/out-mv/run_pool.py compare --a "$OUT/nodoors" --b "$OUT/doors" || rc=1
for n in nodoors doors; do
  log=/root/out-mv/logs/boot-c1-$n.log
  echo "engagement[$n]: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
done
/root/out-mv/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c1 tapes) ==="
python3 /root/out-mv/looplaw_screen.py "$OUT"/*/
if [ "$rc" -ne 0 ]; then echo "C1_STOP: IDENTITY DIVERGENCE — the window STOPS here (defect, not numeric class)"; fi
echo "C1_DONE rc=$rc"
exit "$rc"
