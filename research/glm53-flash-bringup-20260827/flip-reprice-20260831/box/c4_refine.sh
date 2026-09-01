#!/usr/bin/env bash
# FLIP RE-PRICE CELL 4 — fires ONLY if a spec arm beat plain in the cell-3 flip table.
# On the WINNER K: (a) K sweep refinement around the winner (neighbors + K5 if the curve
# is still rising), interleaved with plain x3 per the owner protocol; (b) c=4 concurrency
# row (plain vs winner-K pinned vs nopin auto-policy, the 3way cell-5 shape); (c) PMIN
# overlay MEMRA_SPEC_PMIN in {0.5, 0.7} on the winner (tau receipts exist at old K=3:
# 0.5/0.7 = 33.47/33.79 — this prices tau ON TOP of the batched walk).
# TIMED: caller holds /root/TIMING-IN-FLIGHT. Usage: c4_refine.sh <winnerK> <sweepKs...>
set -uo pipefail
OUT=/root/out-flip3/c4
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
WINNER="${1:?winner K}"; shift
SWEEP=("$@")
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C4 BOOT $name ########"
  /root/out-flip3/serve.sh start "c4-$name" "$@" || { echo "C4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/$name" || { echo "C4_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  echo "C4_${name}_EXIT=0"
}

# (a) K sweep refinement, interleaved with plain, x3 rounds
for i in 1 2 3; do
  boot_and_time "plain$i"
  for K in "${SWEEP[@]}"; do
    boot_and_time "k$K-$i" "${DFL[@]}" MEMRA_SPEC_K="$K"
    /root/out-flip3/serve.sh walk "c4-k$K-$i" batched || echo "C4_k$K-${i}_WALK=RED"
  done
done

# (b) c=4 concurrency on the winner: plain / nopin (auto policy) / winner-K pinned
boot_and_time conc-plain
python3 /root/out-flip3/run_pool.py conc --out "$OUT/conc-plain" --n 4 --mode greedy
boot_and_time conc-nopin "${DFL[@]}"
python3 /root/out-flip3/run_pool.py conc --out "$OUT/conc-nopin" --n 4 --mode greedy
grep -m2 '\[spec-gate\]' /root/out-flip3/logs/boot-c4-conc-nopin.log || true
boot_and_time "conc-k$WINNER" "${DFL[@]}" MEMRA_SPEC_K="$WINNER"
python3 /root/out-flip3/run_pool.py conc --out "$OUT/conc-k$WINNER" --n 4 --mode greedy

# (c) PMIN overlay on the winner
for TAU in 0.5 0.7; do
  boot_and_time "k$WINNER-tau$TAU" "${DFL[@]}" MEMRA_SPEC_K="$WINNER" MEMRA_SPEC_PMIN="$TAU"
  grep -m2 -iE 'confidence gate|PMIN' "/root/out-flip3/logs/boot-c4-k$WINNER-tau$TAU.log" || true
done
echo "=== TAU CROSS-IDENTITY (greedy tapes must not move) ==="
python3 /root/out-flip3/run_pool.py compare --a "$OUT/k$WINNER-tau0.5" --b "$OUT/k$WINNER-tau0.7" || true

/root/out-flip3/serve.sh stop
echo "=== LOOP-LAW SCREEN (c4 tapes) ==="
python3 /root/out-flip3/looplaw_screen.py "$OUT"/*/
echo "C4_ALL_DONE"
