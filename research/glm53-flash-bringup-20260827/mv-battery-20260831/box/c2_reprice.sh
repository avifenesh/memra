#!/usr/bin/env bash
# MV-DOORS CELL 2 — THE COMPOSED RE-PRICE (the decision number, TIMED): ship config
# (DFlash2 + auto-K nopin + PMIN0.7, VERIFY_BATCH default ON) doors-OFF vs ALL FIVE
# doors ON, interleaved fresh boots x3 (x5 on anomaly — round indices as args).
# Baseline: 62.43 tok/s (diet-battery V3 SHIP row on the vrest head, same env, same
# pools). Caller holds /root/TIMING-IN-FLIGHT. Greedy pool + one vendor-default sampled
# row per boot (run_pool timed), loop-law screen at the end.
set -uo pipefail
OUT=/root/out-mv/c2
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
DOORS=(MEMRA_BF16_TCOLS_WIDE=1 MEMRA_BF16_TCOLS_X1=1 MEMRA_MOE_VROWS_PACK=1 MEMRA_TOPK_SHARDS=1 MEMRA_GLM5_VERIFY_WS=1)
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C2 BOOT $name ########"
  /root/out-mv/serve.sh start "c2-$name" "$@" || { echo "C2_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py sample --out "$OUT/$name" || { echo "C2_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-mv/serve.sh doors "c2-$name" "$@" || { echo "C2_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-mv/logs/boot-c2-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log") vrows=$(grep -c '\[glm5-vrows\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -m1 -E '\[glm5-spec\] route=spec' "$log" || true
  echo "C2_${name}_EXIT=0"
}

rc=0
for i in "$@"; do
  boot_and_time "off-$i" "${DFL[@]}" || rc=1
  boot_and_time "don-$i" "${DFL[@]}" "${DOORS[@]}" || rc=1
done

/root/out-mv/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c2 tapes) ==="
python3 /root/out-mv/looplaw_screen.py "$OUT"/*/
echo "=== C2 COMPOSED RE-PRICE TABLE (baseline = ship config doors-OFF) ==="
python3 /root/out-mv/mv_check.py --base "$OUT" --baseline off --arms don
echo "C2_ROUNDS_DONE: $* rc=$rc"
