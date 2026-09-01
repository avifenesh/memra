#!/usr/bin/env bash
# c4 ADDENDUM (named deviation, receipted): PMIN 0.5/0.7 overlay ALSO on K=3 — the flip
# table vendor rows favor K3 (46.21 vs K2 43.22) and the tau arithmetic favors the
# larger-K arm at the halved 11.2 ms/K marginal; the task overlay names "the winner",
# which K2 (greedy) and K3 (sampled) straddle. TIMED: caller holds the marker.
set -uo pipefail
OUT=/root/out-flip3/c4
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
for TAU in 0.5 0.7; do
  name="k3-tau$TAU"
  echo "######## C4B BOOT $name ########"
  /root/out-flip3/serve.sh start "c4-$name" "${DFL[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN="$TAU" \
    || { echo "C4B_${name}_EXIT=BOOTFAIL"; continue; }
  grep -m1 -iE "confidence gate" /root/out-flip3/logs/boot-c4-$name.log || true
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/$name" || { echo "C4B_${name}_EXIT=SAMPLEFAIL"; continue; }
  python3 /root/out-flip3/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  /root/out-flip3/serve.sh walk "c4-$name" batched || echo "C4B_${name}_WALK=RED"
  echo "C4B_${name}_EXIT=0"
done
/root/out-flip3/serve.sh stop
echo "=== TAU K3 CROSS-IDENTITY ==="
python3 /root/out-flip3/run_pool.py compare --a "$OUT/k3-tau0.5" --b "$OUT/k3-tau0.7" || true
python3 /root/out-flip3/looplaw_screen.py "$OUT"/k3-tau0.5 "$OUT"/k3-tau0.7
echo "C4B_ALL_DONE"
