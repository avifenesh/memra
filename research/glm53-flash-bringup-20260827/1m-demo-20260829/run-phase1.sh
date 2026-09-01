#!/bin/bash
# Phase 1 of the 1M demonstration: prove the cell binary serves, prove the deadline
# override works, and prove the PP4 placement is byte-identical to the receipted
# door-off config on a real prompt BEFORE committing hours to it.
#
# Arm A (DOOROFF-8K): the ring lane's exact receipted config (2 cards visible, SLRU
#   experts, MEMRA_CTX=8192), this cell's binary. Greedy 64 on a real-corpus prompt.
# Arm B (PP4-8K): MEMRA_PP_STAGES=4 across all four cards, same prompt, same binary.
# Verdict: the two "output" fields must be BYTE-IDENTICAL (PP is a placement; the
#   hyper ppN gate holds split-vs-unsplit at bit identity, this is the artifact-scale echo).
set -u
R=$HOME/lane-1mdemo-vast-20260829
BIN=$HOME/wt-1mdemo/target/release/memra-server
C=$R/corpus-1m.txt
{
  date -u +%FT%TZ
  echo "=== ARM A: DOOROFF-8K (ring-lane receipted config, cell binary) ==="
  bash $R/serve-1m.sh DOOROFF-8K "$BIN" MEMRA_CTX=8192 CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 || exit 1
  python3 $R/primeprobe.py A-smoke-1k "$C" 4200 64 greedy $R/phase1-A-1k.json
  python3 $R/primeprobe.py A-smoke-6k "$C" 26000 64 greedy $R/phase1-A-6k.json
  echo
  echo "=== ARM B: PP4-8K (all four cards, pp door open, residency default) ==="
  bash $R/serve-1m.sh PP4-8K "$BIN" MEMRA_CTX=8192 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1
  python3 $R/primeprobe.py B-smoke-1k "$C" 4200 64 greedy $R/phase1-B-1k.json
  python3 $R/primeprobe.py B-smoke-6k "$C" 26000 64 greedy $R/phase1-B-6k.json
  echo
  echo "=== IDENTITY VERDICT ==="
  python3 - <<'EOF'
import json
for rung in ("1k", "6k"):
    a = json.load(open(f"/root/lane-1mdemo-vast-20260829/phase1-A-{rung}.json"))
    b = json.load(open(f"/root/lane-1mdemo-vast-20260829/phase1-B-{rung}.json"))
    same = a["output"] == b["output"]
    print(f"rung {rung}: doorOFF vs PP4 byte-identical = {same}"
          f"  (A {len(a['output'])} chars, B {len(b['output'])} chars,"
          f" prompt_tokens A={a['usage'] and a['usage'].get('prompt_tokens')}"
          f" B={b['usage'] and b['usage'].get('prompt_tokens')})")
    if not same:
        print("  A:", repr(a["output"][:200]))
        print("  B:", repr(b["output"][:200]))
EOF
  date -u +%FT%TZ
  echo PHASE1DONE
} 2>&1 | tee $R/03-phase1.txt
