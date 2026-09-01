#!/usr/bin/env bash
# CELL 4 — PHASE RECEIPT AT TWO DEPTHS (UNTIMED, marker DOWN; MEMRA_GLM5_SPEC_TRACE=2).
#
# READ THESE AS SHARES, NEVER AS WALLS. Level 2 synchronizes the stream at every phase
# boundary AND adds per-layer stream drains, so it serializes what an untraced round overlaps.
# The numbers attribute; they never price. That is why this cell is UNTIMED, holds no marker,
# and its rows never enter a perf table.
#
# DEPTH PAIR - DELIBERATE DEVIATION FROM THE BRIEF, stated: the brief asked for 16k vs 525k.
# The phase split [glm5-phase-v] only exists on a VERIFY WALK, the verify walk only exists on a
# spec boot, and cell 3a measured that the glm5 spec route REFUSES PP4 (admits 2..=3 stages).
# So a phase receipt has to run on PP3, whose fleet serving ceiling is MEMRA_CTX=131072 - which
# makes 131k the deepest verify-walk phase receipt obtainable today, and 525k unreachable for
# this measurement. (525k IS reachable on PP3 with the capped-arena posture instead of the
# resident one - roughly 26 GB of planes on three cards - but it costs a ~45 min prime that did
# not fit this window. Named as a follow-up, not silently skipped.)
#
# THE PREDICTION UNDER TEST: vkda flat with depth (KDA is linear attention - its per-token work
# does not grow with the plane) and vmla growing (MLA plus the DSA k-pool indexer scan is the
# depth term). Note the trace has NO separate indexer bucket: the indexer cost sits INSIDE
# vmla, so vmla growth is the indexer+MLA term jointly - that is the attribution the
# indexer-diet lane needs, and the honest limit of what this instrument can separate.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c4
mkdir -p "$D"
# POSTURE: the capped-arena PP3 config cell 3b used, NOT the resident fleet env. Two boots of
# the fleet env produced ZERO expert-residency decisions on this window's base recipe
# (receipts/c3b-fleetenv-residency-denied/UNRESOLVED.txt), and an unresolved posture must not
# carry a phase attribution. This posture is the one whose spec engagement is PROVEN (cell 3a)
# and whose ship rows are the ones this attribution is meant to explain.
FLEET=(CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_PP_STAGES=3 MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_SPLITS=15,30
       MEMRA_CTX=131072)
SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7
      MEMRA_SPEC_STATS=1 MEMRA_GLM5_SPEC_TRACE=2)
{
date -u +%FT%TZ
echo "######## CELL 4: PHASE RECEIPT, trace=2, 16k vs 131k on PP3 (UNTIMED - shares, not walls) ########"
bash "$OUT/serve.sh" start c4-trace "${SPEC[@]}" "${FLEET[@]}" || { echo "C4_EXIT=BOOTFAIL"; exit 1; }
echo "[c4] posture = capped-arena PP3 (matches cell 3b, so the shares explain those rows)"
bash "$OUT/rung.sh" c4-trace W1K 4200 32 greedy "$D"
bash "$OUT/serve.sh" engage c4-trace "${SPEC[@]}" "${FLEET[@]}" || echo "C4_WARN=engage-red"
echo; echo "=== SHALLOW: 16k (chars 64,400 -> 15,766 tok) ==="
bash "$OUT/rung.sh" c4-trace T16K 64400 128 greedy "$D" || echo "C4_WARN=16k failed"
echo; echo "=== DEEP: 131k (chars 527,000 -> 128,566 tok) ==="
bash "$OUT/rung.sh" c4-trace T131K 527000 128 greedy "$D" || echo "C4_WARN=131k failed"
echo; echo "=== ALL PHASE LINES (the attribution receipt) ==="
grep -E "\[glm5-phase" "$OUT/logs/boot-c4-trace.log" | tee "$D/phase-lines.txt"
echo; echo "=== PER-RUNG PHASE SPLIT (16k vs 131k) ==="
for lab in T16K T131K; do
  echo "--- $lab ---"
  grep -E "\[glm5-phase" "$D/$lab-greedy.serverlog" 2>/dev/null | tail -6
done
echo; echo "=== acceptance per rung ==="
for lab in T16K T131K; do
  echo -n "  $lab: "; grep -E "\[glm5-acc\]" "$D/$lab-greedy.serverlog" 2>/dev/null | tail -1 || echo "none"
done
bash "$OUT/serve.sh" stop
date -u +%FT%TZ
echo "C4_DONE"
} 2>&1 | tee "$OUT/logs/c4.log"
