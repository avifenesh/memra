#!/usr/bin/env bash
# DECODE-DIET CELL 6 — byte-identity spot on the composed arm: spec-vs-plain greedy,
# DFlash2 K=3 x 4 prompts (both pools incl. the rejection-heavy d02 row), all four doors
# ON in BOTH boots. The doors are bit-gated on the rig; the real artifact confirms here.
# ANY divergence blocks cell 7 (the spec re-price is invalid on a broken identity).
set -uo pipefail
OUT=/root/out-diet/c6
TAGS=d00-code,d02-code,d06-prose,l3-A4630
DOORS=(MEMRA_HC_FUSED_PRE=1 MEMRA_HC_DECODE_WS=1 MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C6 BOOT $name ########"
  /root/out-diet/serve.sh start "c6-$name" "$@" || { echo "C6_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "C6_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "c6-$name" "$@" || { echo "C6_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$TAGS"
  echo "C6_${name}_EXIT=0"
}

run_arm don-plain "${DOORS[@]}" || exit 1
rc=0
run_arm don-k3 "${DOORS[@]}" "${DFL[@]}" MEMRA_SPEC_K=3 || rc=1
echo "=== C6 IDENTITY don-k3 vs don-plain ==="
python3 /root/out-diet/run_pool.py compare --a "$OUT/don-plain" --b "$OUT/don-k3" || rc=1
log=/root/out-diet/logs/boot-c6-don-k3.log
echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c6 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "C6_DONE rc=$rc"
exit "$rc"
