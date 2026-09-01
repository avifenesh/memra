#!/usr/bin/env bash
# FLIP RE-BATTERY CELL 1 — boot + byte-identity re-gate on the loop-ported head (bb8d9e3cc).
# The loop-port lane claims byte identity held through all four ports (rig gates GREEN);
# this re-gates it on the SERVED path on the serving shape. spec-vs-plain greedy,
# K in {1,3} x 6 prompts spanning both pools (incl. the rejection-heavy d02/d04 rows).
# ANY divergence STOPS the window: that is a port bug, the highest-value catch possible.
# K is a BOOT PIN (MEMRA_SPEC_K) — no request-level spec_k exists on this server.
set -uo pipefail
OUT=/root/out-flip2/c1
TAGS=d00-code,d02-code,d04-code,d06-prose,l3-WARM,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C1 BOOT $name ########"
  /root/out-flip2/serve.sh start "c1-$name" "$@" || { echo "C1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py sample --out "$OUT/$name" || { echo "C1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$TAGS"
  echo "C1_${name}_EXIT=0"
}

run_arm plain || exit 1
rc_all=0
for K in 1 3; do
  run_arm "dfl-k$K" "${DFL[@]}" MEMRA_SPEC_K="$K" || { rc_all=1; continue; }
  echo "=== IDENTITY dfl-k$K vs plain ==="
  python3 /root/out-flip2/run_pool.py compare --a "$OUT/plain" --b "$OUT/dfl-k$K" || rc_all=1
  log=/root/out-flip2/logs/boot-c1-dfl-k$K.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "K receipt:"; grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
  echo "acc tail:"; grep '\[glm5-acc\]' "$log" | tail -2 || true
done

/root/out-flip2/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c1 tapes) ==="
python3 /root/out-flip2/looplaw_screen.py "$OUT"/*/
echo "C1_ALL_DONE rc=$rc_all"
exit "$rc_all"
