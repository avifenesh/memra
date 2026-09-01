#!/usr/bin/env bash
# CELL 1 — BOOT GATE + THE 1M PRIME, PLAIN (TIMED, marker held by the caller).
#
# The question: what does a 1,035,357-token prime cost on the CURRENT head? The banked
# baseline is the 1m-demo's PRE-MLA-TC, PRE-grouped-prefill arm: 6,419.8 s (107.0 min) at
# 161.28 tok/s. This cell is the gate for the whole window - if the prime has not come
# down hard, the two-ladder plan does not fit the wall and cell 3 stops at 525k.
#
# Same corpus (sha-verified == demo), same char slice (4,282,700 -> 1,035,357 tokens on the
# demo's ratio), same probe method, same PP4 + capped-SLRU posture. Ascending order matters:
# the W1K warm rung populates the SLRU arena and gives a session-1 VRAM reading BEFORE the
# 1M request's upfront allocations, exactly as the demo's phase 7 did.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c1
mkdir -p "$D"
CHARS_1M=4282700   # demo phase7: -> 1,035,357 prompt tokens inside the 1,048,576 window
{
date -u +%FT%TZ
echo "######## CELL 1: BOOT + 1M PLAIN PRIME (timed) ########"
bash "$OUT/serve.sh" start c1-plain || { echo "C1_EXIT=BOOTFAIL"; exit 1; }
bash "$OUT/vramwatch.sh" "$D/vram.csv" 5 & VW=$!
echo "vramwatch pid $VW"

echo; echo "=== W1K warm rung (greedy, 32 tok): arena populate + session-1 VRAM ==="
bash "$OUT/rung.sh" c1-plain W1K 4200 32 greedy "$D"

echo; echo "=== ENGAGE gate (announces print at first engagement, so AFTER the first request) ==="
bash "$OUT/serve.sh" engage c1-plain || echo "C1_WARN=engage-red (see logs/boot-c1-plain.engage)"

echo; echo "=== R1M PLAIN greedy: THE PRIME (demo baseline 6,419.8 s @ 161.28 tok/s) ==="
bash "$OUT/rung.sh" c1-plain R1M "$CHARS_1M" 256 greedy "$D"

echo; echo "=== R1M PLAIN vendor-default sampled twin (serving law: never greedy-only) ==="
if python3 -c "import json,sys; j=json.load(open('$D/R1M-greedy.json')); sys.exit(0 if j['status']==200 and not j['error'] else 1)"; then
  bash "$OUT/rung.sh" c1-plain R1M 4282700 256 vendor "$D"
else
  echo "R1M greedy FAILED - skipping the sampled twin, the failure is the receipt"
fi

kill "$VW" 2>/dev/null && echo "vramwatch $VW stopped"
echo; echo "--- PER-CARD VRAM PEAK over cell 1 (cards are 97,887 MiB) ---"
python3 - "$D/vram.csv" <<'PY'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0] != "ts"]
for g in range(4):
    pk = max(int(r[g+1]) for r in rows)
    print(f"  gpu{g}: peak {pk} MiB ({100*pk/97887:.1f}% of the card)")
print("  demo phase7 peaks for reference: 81945 / 80121 / 80089 / 94905 MiB")
PY
echo "--- boot-wide error census (must be 0) ---"
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY" "$OUT/logs/boot-c1-plain.log"
bash "$OUT/serve.sh" stop
date -u +%FT%TZ
echo "C1_DONE"
} 2>&1 | tee "$OUT/logs/c1.log"
