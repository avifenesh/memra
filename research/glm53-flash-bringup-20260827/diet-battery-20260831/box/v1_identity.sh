#!/usr/bin/env bash
# VREST PHASE CELL V1 — byte-identity re-gate FIRST on the vrest head a3fc59aaf
# (carry list: flip-battery cell-1 shape, spec-vs-plain greedy, K in {1,3} x 6 prompts
# incl the rejection-heavy d02/d04 rows; ANY divergence STOPS the window). Then the
# composed-doors spot: d34-plain vs d34+DFlash2 K3 x 4 prompts (the doors must compose
# with the vrest walk on the real artifact).
set -uo pipefail
OUT=/root/out-diet/v1
TAGS=d00-code,d02-code,d04-code,d06-prose,l3-WARM,l3-A4630
TAGS4=d00-code,d02-code,d06-prose,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 DIET_PHASE=vrest)
D34=(MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
mkdir -p "$OUT"

run_arm() {  # name, tags, extras...
  local name="$1" tags="$2"; shift 2
  echo "######## V1 BOOT $name ########"
  /root/out-diet/serve.sh start "v1-$name" "$@" || { echo "V1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "V1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "v1-$name" "$@" || { echo "V1_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$tags"
  echo "V1_${name}_EXIT=0"
}

rc=0
run_arm plain "$TAGS" || exit 1
for K in 1 3; do
  run_arm "dfl-k$K" "$TAGS" "${DFL[@]}" MEMRA_SPEC_K="$K" || { rc=1; continue; }
  echo "=== V1 IDENTITY dfl-k$K vs plain (ANY divergence STOPS) ==="
  python3 /root/out-diet/run_pool.py compare --a "$OUT/plain" --b "$OUT/dfl-k$K" || rc=2
done
run_arm d34-plain "$TAGS4" "${D34[@]}" || rc=1
run_arm d34-k3 "$TAGS4" "${D34[@]}" "${DFL[@]}" MEMRA_SPEC_K=3 || rc=1
echo "=== V1 IDENTITY d34-k3 vs d34-plain (composed spot) ==="
python3 /root/out-diet/run_pool.py compare --a "$OUT/d34-plain" --b "$OUT/d34-k3" || rc=2

/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all v1 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "V1_DONE rc=$rc"
exit "$rc"
