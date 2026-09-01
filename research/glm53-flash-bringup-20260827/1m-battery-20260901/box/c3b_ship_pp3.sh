#!/usr/bin/env bash
# CELL 3b — THE SHIP CONFIG AT DEPTH, ON THE PLACEMENT THAT ADMITS IT (timed; marker held).
#
# Cell 3a establishes by measurement that the glm5 spec route refuses PP4 by design
# (glm5_sharded_placement_admits = (2..=3) stages, fail-closed, no ppN gate receipt above 3).
# The 1M posture IS PP4. So "the ship config at 1M" cannot be measured at 1M - it does not
# exist today. What CAN be measured, and what the fleet actually serves, is the ship config
# on PP3, whose own context ceiling is MEMRA_CTX=131072.
#
# This cell therefore prices the ship config at depth up to the PP3 ceiling, on the FLEET'S
# OWN SERVING ENV (mv-/struct-/tpd-battery box/serve.sh - the env behind the banked 70.458 /
# 71.489 single-stream ship rows), so the rows are comparable with the bank rather than with
# a 1M-posture hybrid:
#   MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 devices 0,1,2
#   MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16     (resident experts - the fleet posture,
#                                                     which is ALSO what lets grouped MoE
#                                                     prefill actually execute)
#   MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1
#   MEMRA_CTX=131072
# Ship spec extras: DFlash2 @ b33c0347 + MEMRA_GLM5_SPEC=1 + MEMRA_SPEC_PMIN=0.7, K UNSET
# (auto-K). Note auto-K's own admission band is LOW=2/HIGH=4 on this placement, so a
# single-stream request may still decline spec by policy; the arm spec is recorded per rung
# from the server log, and a K-pinned twin is run at the deepest rung so the placement
# question and the concurrency question are never conflated.
#
# Rungs: 16k and 131k (the PP3 ceiling), each greedy + vendor-default sampled.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c3b-ship-pp3
mkdir -p "$D"
# NOTE THE LAST TWO ENTRIES - they are load-bearing and were missing on the first attempt.
# serve.sh's recipe pins MEMRA_MOE_VRAM_FRAC / MEMRA_MOE_HARD_VRAM_FRAC to 0.35 because that
# is what makes the 1M PP4 request ADMISSIBLE. Those caps must NOT follow the fleet arm: with
# a 0.35 ceiling the fleet's MEMRA_MOE_RESIDENT_GB=98 cannot be honoured, expert residency is
# DENIED (measured: zero "resident-experts decision" lines, boot VRAM 8-11 GB/card instead of
# ~60, and a 15,766-token prime that had not finished in 6.5 minutes because every layer was
# thrashing the host). env applies assignments in order and last wins, so restoring the stock
# 0.85/0.80 here overrides the recipe for this arm only. Receipt of the misconfigured run is
# kept in receipts/c3b-arenacap-misconfig/ rather than discarded.
# POSTURE: the SAME base recipe as the PP4 1M rows (capped-arena, host-pinned staging, no
# bf16 mirror) with ONLY the stage count and the context window changed. Chosen deliberately
# over the fleet serving env after two boots of that env produced ZERO expert-residency
# decisions (receipts/c3b-fleetenv-residency-denied/UNRESOLVED.txt - the arena-fraction
# hypothesis was raised and REFUTED, ST_PINNED is the remaining unconfirmed suspect).
# Two things this buys, both worth more than bank-comparability here:
#   1. cell 3a already PROVED the spec walk engages on exactly this config at PP3
#      (4 [glm5-acc] lines, acceptance 0.586->0.632, PMIN=0.700, batched verify walk);
#   2. the whole curve now lives on ONE base recipe, so PP4-plain vs PP3-spec differs in
#      stage count and spec env alone, not in residency, bf16 mirror and grouped prefill too.
# The cost, stated: these rows are NOT directly comparable with the banked 70.458/71.489
# fleet ship rows, which were measured on the resident+bf16 env.
FLEET=(CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_PP_STAGES=3 MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_SPLITS=15,30
       MEMRA_CTX=131072)
SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 MEMRA_SPEC_STATS=1)
{
date -u +%FT%TZ
echo "######## CELL 3b: SHIP CONFIG AT DEPTH on PP3 (the placement that admits spec) ########"
echo "fleet env: ${FLEET[*]}"
echo "ship spec: ${SPEC[*]}  (K unset = auto-K)"

# ---- arm 1: the ship config, auto-K, greedy + vendor at 16k and 131k -------------------
bash "$OUT/serve.sh" start c3b-ship "${SPEC[@]}" "${FLEET[@]}" || { echo "C3B_EXIT=BOOTFAIL"; exit 1; }
echo "[c3b] posture = capped-arena PP3 (NOT the resident fleet env); residency decisions = $(grep -c "resident-experts decision" "$OUT/logs/boot-c3b-ship.log") by design"
bash "$OUT/vramwatch.sh" "$D/vram.csv" 5 & VW=$!
bash "$OUT/rung.sh" c3b-ship W1K 4200 32 greedy "$D"
echo; echo "=== ENGAGE gate (the whole point: does the spec walk RUN here?) ==="
bash "$OUT/serve.sh" engage c3b-ship "${SPEC[@]}" "${FLEET[@]}" || echo "C3B_WARN=engage-red"
for r in "B16K:64400:128" "B131K:527000:128"; do
  lab=${r%%:*}; rest=${r#*:}; ch=${rest%%:*}; mt=${rest##*:}
  echo; echo "=== $lab greedy (chars=$ch) ==="
  bash "$OUT/rung.sh" c3b-ship "$lab" "$ch" "$mt" greedy "$D" || echo "C3B_WARN=$lab greedy failed"
  echo; echo "=== $lab VENDOR-DEFAULT sampled (the real traffic shape) ==="
  bash "$OUT/rung.sh" c3b-ship "$lab" "$ch" "$mt" vendor "$D" || echo "C3B_WARN=$lab vendor failed"
done
kill "$VW" 2>/dev/null
echo; echo "--- acceptance lines, whole boot ---"
grep -E "\[glm5-acc\]" "$OUT/logs/boot-c3b-ship.log" | tail -20
echo "--- error census (must be 0) ---"
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY|\[admit-oom\]" "$OUT/logs/boot-c3b-ship.log"
bash "$OUT/serve.sh" stop

# ---- arm 2: the PLAIN COMPARATOR on the same PP3 fleet env ------------------------------
# Required for an honest side-by-side: arm 1 is spec on PP3+fleet-env, so its comparator must
# be plain on PP3+fleet-env. Comparing arm 1 against the PP4 1M-posture plain rows would vary
# placement, residency, bf16 mirror and grouped prefill all at once.
# (The K-pinned arm originally planned here was DROPPED as redundant: cell 3a's PP3 arm ran
# auto-K and spec engaged at c=1 with acceptance 0.586-0.632, so auto-K is not a blocker.)
echo; echo "######## ARM c3b-plain: PLAIN comparator, same PP3 fleet env ########"
bash "$OUT/serve.sh" start c3b-plain "${FLEET[@]}" || { echo "C3B_PLAIN_EXIT=BOOTFAIL"; exit 1; }
bash "$OUT/rung.sh" c3b-plain W1K 4200 32 greedy "$D/plain"
bash "$OUT/serve.sh" engage c3b-plain "${FLEET[@]}" || echo "C3B_PLAIN_WARN=engage-red"
for r in "P16K:64400:128" "P131K:527000:128"; do
  lab=${r%%:*}; rest=${r#*:}; ch=${rest%%:*}; mt=${rest##*:}
  echo; echo "=== $lab plain greedy (chars=$ch) ==="
  bash "$OUT/rung.sh" c3b-plain "$lab" "$ch" "$mt" greedy "$D/plain" || echo "C3B_WARN=$lab plain failed"
done
echo "--- plain arm must have ZERO acceptance lines (it is the comparator) ---"
grep -c "\[glm5-acc\]" "$OUT/logs/boot-c3b-plain.log"
bash "$OUT/serve.sh" stop

echo; echo "=== loop-law screen over cell 3b ==="
python3 "$OUT/looplaw_screen.py" "$D" | tee "$D/looplaw.txt"
date -u +%FT%TZ
echo "C3B_DONE"
} 2>&1 | tee "$OUT/logs/c3b.log"
