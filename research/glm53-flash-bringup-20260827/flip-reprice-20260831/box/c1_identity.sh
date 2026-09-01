#!/usr/bin/env bash
# FLIP RE-PRICE CELL 1 — boot + byte-identity gate on the batched verify walk
# (lane/glm5-verify-batch @ c62677352, MEMRA_GLM5_VERIFY_BATCH default ON).
# The rig gates proved bit identity (glm5_verify_batch_gpu 3/3, tparallel 9/9); the real
# artifact on the served path is the final word. spec-vs-plain greedy, K in {1,3} x 6
# prompts spanning both pools (incl. the rejection-heavy d02/d04 rows).
# ANY divergence STOPS the window. K is a BOOT PIN (MEMRA_SPEC_K).
set -uo pipefail
OUT=/root/out-flip3/c1
TAGS=d00-code,d02-code,d04-code,d06-prose,l3-WARM,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C1 BOOT $name ########"
  /root/out-flip3/serve.sh start "c1-$name" "$@" || { echo "C1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/$name" || { echo "C1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$TAGS"
  echo "C1_${name}_EXIT=0"
}

run_arm plain || exit 1
rc_all=0
for K in 1 3; do
  run_arm "dfl-k$K" "${DFL[@]}" MEMRA_SPEC_K="$K" || { rc_all=1; continue; }
  # engagement receipt for THE SEAM UNDER TEST: the batched walk announced itself
  /root/out-flip3/serve.sh walk "c1-dfl-k$K" batched || rc_all=1
  echo "=== IDENTITY dfl-k$K vs plain ==="
  python3 /root/out-flip3/run_pool.py compare --a "$OUT/plain" --b "$OUT/dfl-k$K" || rc_all=1
  log=/root/out-flip3/logs/boot-c1-dfl-k$K.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "K receipt:"; grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
  echo "acc tail:"; grep '\[glm5-acc\]' "$log" | tail -2 || true
done

/root/out-flip3/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c1 tapes) ==="
python3 /root/out-flip3/looplaw_screen.py "$OUT"/*/
echo "C1_ALL_DONE rc=$rc_all"
exit "$rc_all"
