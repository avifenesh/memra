#!/usr/bin/env bash
# DECODE-DIET CELL 1 — COMPOSED-FIRST (the decision number): all four diet doors ON vs
# all OFF, interleaved fresh boots (x3 per the amended law, x5 on anomaly — this script
# takes round indices as args so `c1_composed.sh 4 5` runs extension rounds).
# TIMED: the caller raises /root/TIMING-IN-FLIGHT before and holds it for the cell.
# Per boot: fresh-boot sample gate, DOOR ENGAGEMENT gate (announces demanded per ON arm,
# zero in OFF — checked after the sample so first-engagement announces exist), streamed
# greedy decode pool 256 (TTFT + tok/s), l3 pool tok/s, deep TTFT 0.4k/3.7k, ONE
# vendor-default sampled row (never-serve-greedy law).
# EXTRA IDENTITY BAR: the doors claim BIT identity (rig-gated); the ON-vs-OFF greedy
# tapes of every round are compared and ANY divergence STOPS THE WINDOW (a door bug on
# the real artifact is the highest-value catch possible).
set -uo pipefail
OUT=/root/out-diet/c1
DOORS=(MEMRA_HC_FUSED_PRE=1 MEMRA_HC_DECODE_WS=1 MEMRA_KDA_FUSED_PROJ=1 MEMRA_MLA_DECODE_SPLIT=1)
mkdir -p "$OUT"

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## C1 BOOT $name ########"
  /root/out-diet/serve.sh start "c1-$name" "$@" || { echo "C1_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name" || { echo "C1_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "c1-$name" "$@" || { echo "C1_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  echo "C1_${name}_EXIT=0"
}

rc_all=0
for i in "$@"; do
  boot_and_time "off-$i" || rc_all=1
  boot_and_time "don-$i" "${DOORS[@]}" || rc_all=1
  echo "=== C1 IDENTITY don-$i vs off-$i (greedy tapes; ANY divergence STOPS) ==="
  python3 /root/out-diet/run_pool.py compare --a "$OUT/off-$i" --b "$OUT/don-$i" \
    || { echo "C1_IDENTITY_DIVERGENCE round=$i — STOP THE WINDOW"; rc_all=2; }
done

/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c1 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "C1_ROUNDS_DONE: $* rc=$rc_all"
exit "$rc_all"
