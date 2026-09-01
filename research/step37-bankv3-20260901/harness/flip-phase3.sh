#!/usr/bin/env bash
# PHASE 3: the 8-TURN LARGER-PROMPT CACHE TWIN (LAW:multiturn-cache-twin, owner 2026-08-21).
#
# Two real 8-turn agentic conversations from the owner-blessed mt-agentic packs, re-minted on the
# rig with research/multiturn-cache-20260821/mint-mt-packs.py and verified 16/16 against the
# committed packs-MANIFEST.json sha16 list, so the corpus is byte-pinned WITHOUT committing private
# session text. Depth by turn 8: ~38.9k and ~24.7k estimated prompt tokens.
#
# THREE SHAPING DECISIONS, each traceable to a banked trap rather than to taste:
#
#  * max_tokens 4096, NOT 1024. QUIRK:step37:think-budget-wall-1024 -- on agentic-complexity
#    prompts under this model the think phase ALONE exceeds 1024 tokens at every depth, so a 1024
#    budget makes a contentless 200 (finish=length, content="") the DEFAULT outcome and the replayed
#    history fills with nothing. Budget, not depth, is the lever.
#
#  * FIXED-HISTORY REPLAY, not self-feeding. A self-fed conversation forks history after any
#    divergent turn, so a cross-arm per-turn comparison of two self-fed runs is not an A/B. The OFF
#    arm banks its per-turn assistant CONTENT (--save-transcript) and the ON arm replays it
#    (--replay-from), so every turn's rendered prompt is byte-identical across arms. That makes the
#    per-turn content sha a real exactness gate AT DEPTH, and makes the TTFT/accept comparison fair.
#    mt_bench replays CONTENT only and never reasoning: TRAP:reasoning-replayed-as-history-content
#    (replaying truncated think prose is what faked the 2026-08-28/29 "deep-context degeneration").
#
#  * A CACHE-BUST ARM. cached_tokens is a counter, and a counter is not evidence until something
#    makes it move for a known reason. --bust-turn 5 gives turns 5-8 a fresh cache_salt, so the TTFT
#    jump and the cached_tokens collapse are the can't-hallucinate proof that the counter tracks
#    real KV reuse. Without it, "cache engagement proven" is a number read aloud.
#
# Spec stays ON (the serving policy). The cache x spec zone is exactly where the gemma-lc ~12x TTFT
# mispricing lived, and every single-shot cell in this lane is structurally blind to it.
set -u
D=/home/ubuntu/bankv3/lane
FLIP=$D/bin/memra-server-afc681b1a
PACK=$D/harness/bv3-flip-twin.jsonl
MT=4096

echo "########## ARM MTOFF (rollback seam) -- banks the transcript ##########"
"$D/harness/boot.sh" MTOFF "$FLIP" flip-off || { echo "BOOT_FAIL MTOFF"; "$D/harness/stop.sh"; exit 2; }
python3 "$D/harness/mt_bench.py" "$PACK" --url http://127.0.0.1:18640 \
  --model stepfun/step-3.7-flash --arm MTOFF --rep 1 --max-tokens $MT \
  --save-transcript "$D/receipts/mt-transcript-MTOFF.jsonl" \
  --out "$D/receipts/rows-mt.jsonl" 2>&1 | tail -30
echo "--- cache-bust probe, same arm, same boot: turns 5-8 get a fresh salt ---"
python3 "$D/harness/mt_bench.py" "$PACK" --url http://127.0.0.1:18640 \
  --model stepfun/step-3.7-flash --arm MTOFF-BUST --rep 1 --max-tokens $MT --bust-turn 5 \
  --out "$D/receipts/rows-mt.jsonl" 2>&1 | tail -30
"$D/harness/assert-engagement.sh" MTOFF flip-off || echo "ENGAGEMENT_REFUSED MTOFF"
"$D/harness/stop.sh"

echo "########## ARM MTON (the DEFAULT) -- replays MTOFF history ##########"
"$D/harness/boot.sh" MTON "$FLIP" flip-on || { echo "BOOT_FAIL MTON"; "$D/harness/stop.sh"; exit 2; }
python3 "$D/harness/mt_bench.py" "$PACK" --url http://127.0.0.1:18640 \
  --model stepfun/step-3.7-flash --arm MTON --rep 1 --max-tokens $MT \
  --replay-from "$D/receipts/mt-transcript-MTOFF.jsonl" \
  --out "$D/receipts/rows-mt.jsonl" 2>&1 | tail -30
"$D/harness/assert-engagement.sh" MTON flip-on || echo "ENGAGEMENT_REFUSED MTON"
"$D/harness/stop.sh"
echo "===== PHASE 3 DONE ====="
