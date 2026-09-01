#!/usr/bin/env bash
# DECODE-DIET CELL 7 — THE COMPOSED SPEC RE-PRICE (the 100-bar number): composed diet
# (all four doors ON) plain vs composed + DFlash2 spec K in {1,2,3} (batched verify walk
# default ON at this head). K=2 is a NAMED DEVIATION from the pre-registered {1,3}:
# flip-reprice cell 3 (2026-08-31T00:05Z box clock) measured K2 as the PEAK K on the
# batched walk (44.245 tok/s vs K1 41.221 / K3 43.420), and this cell's deliverable is
# the best single-stream number the composed shape produces — omitting the known peak
# would undermine the claim. Interleaved fresh boots x3 (x5 on anomaly — round indices
# as args). TIMED: the caller raises /root/TIMING-IN-FLIGHT and holds it for the cell.
# The chain to the doors-OFF plain baseline is cell 1 (off vs don interleaved); this
# cell prices spec ON TOP of the diet shape. Report the best single-stream number
# against the owner's 100 tok/s bar explicitly.
set -uo pipefail
OUT=/root/out-diet/c7
DOORS=(MEMRA_HC_FUSED_PRE=1 MEMRA_HC_DECODE_WS=1 MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C7 BOOT $name ########"
  /root/out-diet/serve.sh start "c7-$name" "$@" || { echo "C7_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "C7_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "c7-$name" "$@" || { echo "C7_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-diet/logs/boot-c7-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -m1 -E '\[glm5-spec\] route=spec' "$log" || true
  echo "C7_${name}_EXIT=0"
}

rc=0
for i in "$@"; do
  boot_and_time "don-$i"  "${DOORS[@]}" || rc=1
  boot_and_time "dfk1-$i" "${DOORS[@]}" "${DFL[@]}" MEMRA_SPEC_K=1 || rc=1
  boot_and_time "dfk2-$i" "${DOORS[@]}" "${DFL[@]}" MEMRA_SPEC_K=2 || rc=1
  boot_and_time "dfk3-$i" "${DOORS[@]}" "${DFL[@]}" MEMRA_SPEC_K=3 || rc=1
done

/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c7 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "=== C7 RE-PRICE TABLE (baseline = composed plain) ==="
python3 /root/out-diet/diet_check.py --base "$OUT" --baseline don --arms dfk1,dfk2,dfk3
echo "C7_ROUNDS_DONE: $* rc=$rc"
