#!/usr/bin/env bash
# FLIP RE-BATTERY CELL 4 — native-MTP TTFT spot receipt (one reduced round only).
# The loop-port fold-in A (batched MTP-plane warm fill) claims the native arm's O(prompt)
# sequential plane warm (~400 tok/s, the 3way's 12.30s TTFT at ~3.7k) is closed. One boot,
# the cell-4 timed shape, and the 3.7k cold deep-TTFT row is the receipt (spec-battery
# flip condition 1). TIMED: caller holds /root/TIMING-IN-FLIGHT.
set -uo pipefail
OUT=/root/out-flip2/c4
NAT=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

echo "######## C4 BOOT native ########"
/root/out-flip2/serve.sh start "c4-native" "${NAT[@]}" MEMRA_SPEC_K=3 || { echo "C4_EXIT=BOOTFAIL"; exit 1; }
python3 /root/out-flip2/run_pool.py sample --out "$OUT/native" || { echo "C4_EXIT=SAMPLEFAIL"; exit 1; }
python3 /root/out-flip2/run_pool.py timed --out "$OUT/native" --max-tokens 256
log=/root/out-flip2/logs/boot-c4-native.log
echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
grep -m2 -iE 'plane|warm|fill' "$log" || true
/root/out-flip2/serve.sh stop
echo "=== LOOP-LAW SCREEN (c4 tapes) ==="
python3 /root/out-flip2/looplaw_screen.py "$OUT"/*/
echo "C4_ALL_DONE"
