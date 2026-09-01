#!/usr/bin/env bash
# FLIP RE-PRICE CELL 2 — PHASE RECEIPT at trace level 2 (MEMRA_GLM5_SPEC_TRACE=2, the
# verify-batch instrument): [glm5-phase] per-phase lines (level-1-comparable with the
# flip-battery cell-2 receipts) PLUS the [glm5-phase-v] verify SUB-SPLIT — vkda (with the
# in-kernel scan share), vmla, vrest = glue+FFN+head. LAW: phase (and level-2 per-layer)
# boundaries synchronize the stream, so these are attribution SHARES, never round walls,
# never perf rows — the flip table (cell 3) is the only pricing instrument.
# Boots: DFlash2 K=3 batched, K=1 batched (two K points = fixed-vs-marginal split), and
# DFlash2 K=3 with MEMRA_GLM5_VERIFY_BATCH=0 — the A/B seam receipt at the instrument
# level: the old per-row walk must reproduce the flip-battery verify shares
# (K=3 verify ~96.5 ms/round). Prediction to read against (LANE.md section 4, verbatim):
# "verify K=3 96.5 -> ~40-55 ms ... round ~50-64 ms"; "K=1: verify 51.6 -> ~28-33 ms".
# 4 prompts per boot, greedy 128.
set -uo pipefail
OUT=/root/out-flip3/c2
TAGS=d00-code,d02-code,d06-prose,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
mkdir -p "$OUT"

run_trace() {  # name, walk-expect, extras...
  local name="$1" expect="$2"; shift 2
  echo "######## C2 TRACE BOOT $name ########"
  /root/out-flip3/serve.sh start "c2-$name" "$@" MEMRA_GLM5_SPEC_TRACE=2 \
    || { echo "C2_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/$name" || { echo "C2_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-flip3/run_pool.py cell --out "$OUT/$name" --pool both --mode greedy \
    --max-tokens 128 --tags "$TAGS"
  local rc=0
  /root/out-flip3/serve.sh walk "c2-$name" "$expect" || rc=1
  local log=/root/out-flip3/logs/boot-c2-$name.log
  grep '\[glm5-phase\]' "$log" > "$OUT/$name/glm5-phase-lines.txt" || true
  grep '\[glm5-phase-v\]' "$log" > "$OUT/$name/glm5-phase-v-lines.txt" || true
  echo "phase lines banked: $(wc -l < "$OUT/$name/glm5-phase-lines.txt") + v-split $(wc -l < "$OUT/$name/glm5-phase-v-lines.txt")"
  echo "K receipt:"; grep -m2 -E '\[glm5-spec\] route=spec|clamped' "$log" || true
  echo "C2_${name}_EXIT=$rc"
  return "$rc"
}

rc=0
run_trace dfl-k3-vb  batched "${DFL[@]}" MEMRA_SPEC_K=3 || rc=1
run_trace dfl-k1-vb  batched "${DFL[@]}" MEMRA_SPEC_K=1 || rc=1
run_trace dfl-k3-vb0 perrow  "${DFL[@]}" MEMRA_SPEC_K=3 MEMRA_GLM5_VERIFY_BATCH=0 || rc=1
/root/out-flip3/serve.sh stop
echo "=== PHASE AGGREGATION (shares, never walls) ==="
python3 /root/out-flip3/phase_agg.py "$OUT"/dfl-k3-vb "$OUT"/dfl-k1-vb "$OUT"/dfl-k3-vb0
echo "=== IDENTITY vb0 vs vb (K=3, same 4 prompts — the seam moves TIME, never bytes) ==="
python3 /root/out-flip3/run_pool.py compare --a "$OUT/dfl-k3-vb" --b "$OUT/dfl-k3-vb0" || rc=1
echo "=== LOOP-LAW SCREEN (c2 tapes) ==="
python3 /root/out-flip3/looplaw_screen.py "$OUT"/*/
echo "C2_ALL_DONE rc=$rc"
exit "$rc"
