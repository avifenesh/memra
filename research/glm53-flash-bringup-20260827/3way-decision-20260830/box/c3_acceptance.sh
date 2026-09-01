#!/usr/bin/env bash
# CELL 3 — DFlash2 real-artifact acceptance sanity + tap-shift RED arm.
# Shape is deliberately IDENTICAL to spec-battery-20260830 stage 2 (both pools = 14 real
# prompts, max_tokens 128, greedy + vendor-default) so the rows are directly comparable to the
# banked native-MTP numbers (notrim K3 greedy 1.443 acc/cyc, K5 1.473) and to the probe band
# (3.06 tokens/cycle all, 4.66 tool-wire, acc@1 0.73).
# RED arm: MEMRA_GLM5_DFLASH_GATE_RED=tap-shift on the REAL artifact — acceptance must COLLAPSE
# while the tape stays byte-identical (a draft source may only move acceptance, never output).
set -uo pipefail
OUT=/root/out-3way/c3
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C3 BOOT $name ########"
  /root/out-3way/serve.sh start "c3-$name" "$@" || { echo "C3_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name-greedy" || { echo "C3_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py cell --out "$OUT/$name-greedy" --pool both --mode greedy --max-tokens 128
  python3 /root/out-3way/run_pool.py cell --out "$OUT/$name-vendor" --pool both --mode vendor --max-tokens 128
  local log=/root/out-3way/logs/boot-c3-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 -E '\[glm5-spec\] (serve route ARMED|RED-ARM)' "$log" || true
  grep -m1 'RED-ARM tap-shift' "$log" || true
  echo "C3_${name}_EXIT=0"
}

# plain reference at the 128 shape (the tape reference for both spec arms and the red arm)
arm plain || exit 1
arm k3 "${DFL[@]}" MEMRA_SPEC_K=3 || exit 1
arm k5 "${DFL[@]}" MEMRA_SPEC_K=5 || exit 1
# RED: same K as the k3 arm, wrong tap layers (+1)
arm k3-red "${DFL[@]}" MEMRA_SPEC_K=3 MEMRA_GLM5_DFLASH_GATE_RED=tap-shift || exit 1

/root/out-3way/serve.sh stop

echo "=== TAPE IDENTITY (greedy, 128) vs plain: every spec arm INCLUDING the red arm ==="
rc=0
for a in k3 k5 k3-red; do
  echo "-- $a"
  python3 /root/out-3way/run_pool.py compare --a "$OUT/plain-greedy" --b "$OUT/$a-greedy" || rc=1
done

echo "=== ACCEPTANCE TABLE (greedy + vendor-default; compare to native K3 1.443 / K5 1.473) ==="
python3 /root/out-3way/run_pool.py agg --dirs "$OUT"/k3-greedy "$OUT"/k3-vendor \
  "$OUT"/k5-greedy "$OUT"/k5-vendor "$OUT"/k3-red-greedy "$OUT"/k3-red-vendor

echo "=== LOOP-LAW SCREEN ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "C3_ALL_DONE rc=$rc"
exit "$rc"
