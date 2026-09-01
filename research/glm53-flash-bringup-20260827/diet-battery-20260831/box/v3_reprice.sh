#!/usr/bin/env bash
# VREST PHASE CELL V3 — THE RE-PRICE ON THE VREST HEAD (the 100-bar number, TIMED):
# plain vs the flip-reprice DEPLOYABLE CONFIG (DFlash2 + auto-K nopin + PMIN=0.7,
# baseline 45.65 on the pre-vrest head; predicted ~55-57 here) vs deployable + the
# measured-good doors (kda6 + mlasplit, -1.05 ms/tok plain). Interleaved fresh boots
# x3 (x5 on anomaly — round indices as args). Caller holds /root/TIMING-IN-FLIGHT.
set -uo pipefail
OUT=/root/out-diet/v3
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 DIET_PHASE=vrest)
D34=(MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## V3 BOOT $name ########"
  /root/out-diet/serve.sh start "v3-$name" "$@" || { echo "V3_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "V3_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "v3-$name" "$@" || { echo "V3_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-diet/logs/boot-v3-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log") vrows=$(grep -c '\[glm5-vrows\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -m1 -E '\[glm5-spec\] route=spec' "$log" || true
  echo "V3_${name}_EXIT=0"
}

rc=0
for i in "$@"; do
  boot_and_time "plain-$i" || rc=1
  boot_and_time "ship-$i" "${DFL[@]}" || rc=1
  boot_and_time "shipd34-$i" "${D34[@]}" "${DFL[@]}" || rc=1
done

/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all v3 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "=== V3 RE-PRICE TABLE (baseline = plain on the vrest head) ==="
python3 /root/out-diet/diet_check.py --base "$OUT" --baseline plain --arms ship,shipd34
echo "V3_ROUNDS_DONE: $* rc=$rc"
