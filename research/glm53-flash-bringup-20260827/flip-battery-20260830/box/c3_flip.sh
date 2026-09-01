#!/usr/bin/env bash
# FLIP RE-BATTERY CELL 3 — THE FLIP TABLE (timed): plain vs DFlash2 at K=1, K=2, K=3
# on the loop-ported head. Interleaved fresh boots per arm, plain->k1->k2->k3 per round.
# OWNER PROTOCOL (2026-08-30, this window): x3 rounds by default; escalate to x5 ONLY on
# anomaly — (a) within-arm relative spread of the decode-tok/s boot-medians > 0.5%, or
# (b) verdict too close at any K (spec median within 2x pooled spread of plain median).
# Escalation extends BOTH the affected spec arm and plain, still interleaved (this script
# takes round indices as args, so `c3_flip.sh 4 5` runs the extension rounds).
# TIMED: the caller raises /root/TIMING-IN-FLIGHT before and holds it for the window.
# Per boot: fresh-boot sample gate, streamed greedy decode pool 256 (TTFT + tok/s),
# l3 pool tok/s, deep TTFT at ~0.4k and ~3.7k cold, ONE vendor-default sampled row
# (never-serve-greedy law), engagement receipts, loop-law screen at the end.
set -uo pipefail
OUT=/root/out-flip2/c3
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
ARMS_ARG="${ARMS:-plain k1 k2 k3}"
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C3 BOOT $name ########"
  /root/out-flip2/serve.sh start "c3-$name" "$@" || { echo "C3_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py sample --out "$OUT/$name" || { echo "C3_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-flip2/logs/boot-c3-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  grep -m1 -E '\[glm5-spec\] route=spec' "$log" || true
  echo "C3_${name}_EXIT=0"
}

for i in "$@"; do
  for arm in $ARMS_ARG; do
    case "$arm" in
      plain) boot_and_time "plain$i" ;;
      k1)    boot_and_time "k1-$i" "${DFL[@]}" MEMRA_SPEC_K=1 ;;
      k2)    boot_and_time "k2-$i" "${DFL[@]}" MEMRA_SPEC_K=2 ;;
      k3)    boot_and_time "k3-$i" "${DFL[@]}" MEMRA_SPEC_K=3 ;;
    esac
  done
done

/root/out-flip2/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c3 tapes) ==="
python3 /root/out-flip2/looplaw_screen.py "$OUT"/*/
echo "C3_ROUNDS_DONE: $*"
