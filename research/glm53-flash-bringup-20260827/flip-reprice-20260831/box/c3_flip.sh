#!/usr/bin/env bash
# FLIP RE-PRICE CELL 3 — THE FLIP TABLE (timed): plain vs DFlash2 K=1/K=2/K=3 with the
# BATCHED verify walk ON (MEMRA_GLM5_VERIFY_BATCH default, the seam under test).
# Interleaved fresh boots plain->k1->k2->k3 per round. OWNER PROTOCOL (amended
# 2026-08-30): x3 rounds by default; escalate to x5 ONLY on anomaly — (a) within-arm
# relative spread of decode-tok/s boot-medians > 0.5%, or (b) verdict too close at any K
# (spec median within 2x pooled spread of plain). `c3_flip.sh 1 2 3` runs rounds 1-3;
# `c3_flip.sh 4 5` extends. TIMED: caller raises /root/TIMING-IN-FLIGHT and holds it.
# Per boot: fresh-boot sample gate, streamed greedy decode pool 256 (TTFT + tok/s), l3
# pool tok/s, deep TTFT @~0.4k and ~3.7k cold, ONE vendor-default sampled row
# (never-serve-greedy law; 128-token floor guard), engagement + walk receipts, loop-law.
# ARMS=zctl runs the ONE =0 control boot (old per-row walk, MEMRA_GLM5_VERIFY_BATCH=0,
# K=3): the A/B seam receipt at the wall level — it must reproduce the flip-battery
# round wall (~91.08 ms at K=3), and it never enters the flip table.
set -uo pipefail
OUT=/root/out-flip3/c3
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
ARMS_ARG="${ARMS:-plain k1 k2 k3}"
mkdir -p "$OUT"

boot_and_time() {  # name, walk-expect(batched|perrow|none), extras...
  local name="$1" expect="$2"; shift 2
  echo "######## C3 BOOT $name ########"
  /root/out-flip3/serve.sh start "c3-$name" "$@" || { echo "C3_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/$name" || { echo "C3_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-flip3/logs/boot-c3-$name.log
  [ "$expect" != none ] && { /root/out-flip3/serve.sh walk "c3-$name" "$expect" || echo "C3_${name}_WALK=RED"; }
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -m1 -E '\[glm5-spec\] route=spec' "$log" || true
  echo "C3_${name}_EXIT=0"
}

for i in "$@"; do
  for arm in $ARMS_ARG; do
    case "$arm" in
      plain) boot_and_time "plain$i" none ;;
      k1)    boot_and_time "k1-$i" batched "${DFL[@]}" MEMRA_SPEC_K=1 ;;
      k2)    boot_and_time "k2-$i" batched "${DFL[@]}" MEMRA_SPEC_K=2 ;;
      k3)    boot_and_time "k3-$i" batched "${DFL[@]}" MEMRA_SPEC_K=3 ;;
      zctl)  boot_and_time "zctl-$i" perrow "${DFL[@]}" MEMRA_SPEC_K=3 MEMRA_GLM5_VERIFY_BATCH=0 ;;
    esac
  done
done

/root/out-flip3/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c3 tapes) ==="
python3 /root/out-flip3/looplaw_screen.py "$OUT"/*/
echo "C3_ROUNDS_DONE: $*"
