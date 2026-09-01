#!/usr/bin/env bash
# FLIP RE-BATTERY CELL 2 — PHASE TIMER receipt (MEMRA_GLM5_SPEC_TRACE=1, the loop-port
# port-0 instrument): the first real-artifact per-phase attribution ever taken on the
# serving shape. LAW (loop-port LANE.md): phase boundaries synchronize the stream, so
# these numbers are attribution SHARES, never round walls and never perf rows — the
# flip table (cell 3) is the only pricing instrument in this window.
# Boots: DFlash2 K=3, DFlash2 K=1 (named extra: two K points let the SHARES be read
# against the 3way fixed-vs-marginal split 31.6 + 20.1*K ms), native K=3.
# 4 prompts per boot, greedy 128.
set -uo pipefail
OUT=/root/out-flip2/c2
TAGS=d00-code,d02-code,d06-prose,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
NAT=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_trace() {  # name, extras...
  local name="$1"; shift
  echo "######## C2 TRACE BOOT $name ########"
  /root/out-flip2/serve.sh start "c2-$name" "$@" MEMRA_GLM5_SPEC_TRACE=1 \
    || { echo "C2_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py sample --out "$OUT/$name" || { echo "C2_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip2/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 128 --tags "$TAGS"
  local log=/root/out-flip2/logs/boot-c2-$name.log
  grep '\[glm5-phase\]' "$log" > "$OUT/$name/glm5-phase-lines.txt" || true
  echo "phase lines banked: $(wc -l < "$OUT/$name/glm5-phase-lines.txt")"
  echo "K receipt:"; grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
  echo "C2_${name}_EXIT=0"
}

rc=0
run_trace dfl-k3-trace "${DFL[@]}" MEMRA_SPEC_K=3 || rc=1
run_trace dfl-k1-trace "${DFL[@]}" MEMRA_SPEC_K=1 || rc=1
run_trace nat-k3-trace "${NAT[@]}" MEMRA_SPEC_K=3 || rc=1
/root/out-flip2/serve.sh stop
echo "=== LOOP-LAW SCREEN (c2 tapes) ==="
python3 /root/out-flip2/looplaw_screen.py "$OUT"/*/
echo "C2_ALL_DONE rc=$rc"
exit "$rc"
