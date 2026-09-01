#!/usr/bin/env bash
# DECODE-DIET CELL 7x — NAMED EXTENSION (evidence-driven, receipted): the deployable-
# config twin on the BEST-DOOR shape. Cells 2-5 measured doors 1-2 as a net drag
# (-hcpre +0.318 ms, hcws nil) and doors 3+4 as the whole win (d34 36.777 = 1.0386x,
# additive to +/-0.01 ms), so "the best single-stream number this shape can produce
# today" is d34 + the flip-reprice deployable spec config (nopin auto-K + PMIN 0.7,
# 45.654 on the no-doors shape). Arms interleaved: d34 plain (rounds 2,3 — round 1 is
# the cells-2-5 boot, symlinked) vs d34sp = doors 3+4 + DFlash2 nopin + PMIN0.7 x3.
# TIMED: caller holds /root/TIMING-IN-FLIGHT.
set -uo pipefail
OUT=/root/out-diet/c7x
D34=(MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
mkdir -p "$OUT"
ln -sfn /root/out-diet/c2to5/d34-1 "$OUT/d34-1"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C7X BOOT $name ########"
  /root/out-diet/serve.sh start "c7x-$name" "$@" || { echo "C7X_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "C7X_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "c7x-$name" "$@" || { echo "C7X_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-diet/logs/boot-c7x-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log") pmin=$(grep -ci 'pmin' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -im1 'pmin' "$log" || true
  echo "C7X_${name}_EXIT=0"
}

rc=0
boot_and_time "d34sp-1" "${D34[@]}" "${DFL[@]}" || rc=1
boot_and_time "d34-2"   "${D34[@]}" || rc=1
boot_and_time "d34sp-2" "${D34[@]}" "${DFL[@]}" || rc=1
boot_and_time "d34-3"   "${D34[@]}" || rc=1
boot_and_time "d34sp-3" "${D34[@]}" "${DFL[@]}" || rc=1

/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c7x tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "=== C7X DEPLOYABLE TABLE (baseline = d34 plain) ==="
python3 /root/out-diet/diet_check.py --base "$OUT" --baseline d34 --arms d34sp
echo "C7X_DONE rc=$rc"
