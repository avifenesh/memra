#!/usr/bin/env bash
# CELL 2 — DFlash2 served byte-identity spot battery.
# spec-vs-plain greedy, K in {1,3,5} x 8 prompts spanning BOTH pools (code + prose + deep,
# incl. the rejection-heavy d02/d04 rows the spec-battery lane identified). ANY divergence
# STOPS the window: a draft source may only move acceptance, never output.
# K is a BOOT PIN (MEMRA_SPEC_K) — there is no request-level spec_k on this server, so each
# K arm is its own fresh boot.
set -uo pipefail
OUT=/root/out-3way/c2
TAGS=d00-code,d02-code,d04-code,d05-code,d06-prose,d09-prose,l3-WARM,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_arm() {  # name, extras...
  local name="$1"; shift
  echo "######## C2 BOOT $name ########"
  /root/out-3way/serve.sh start "c2-$name" "$@" || { echo "C2_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name" || { echo "C2_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 256 --tags "$TAGS"
  echo "C2_${name}_EXIT=0"
}

run_arm plain || exit 1
rc_all=0
for K in 1 3 5; do
  run_arm "dfl-k$K" "${DFL[@]}" MEMRA_SPEC_K="$K" || { rc_all=1; continue; }
  echo "=== IDENTITY dfl-k$K vs plain ==="
  python3 /root/out-3way/run_pool.py compare --a "$OUT/plain" --b "$OUT/dfl-k$K" || rc_all=1
  log=/root/out-3way/logs/boot-c2-dfl-k$K.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "K receipt:"; grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
  echo "acc tail:"; grep '\[glm5-acc\]' "$log" | tail -2 || true
done

/root/out-3way/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c2 tapes) ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "C2_ALL_DONE rc=$rc_all"
exit "$rc_all"
