#!/usr/bin/env bash
# CELL 3 — admission interplay at high c (count-based, untimed): workspace-priced admission
# (MEMRA_ADMIT_PREFILL_WORKSPACE default ON) + session bars under the ON arm.
# GREEN bar: any refusal is a clean [admit-oom] 429 BEFORE prime; the engine NEVER OOMs
# mid-stream (class=Overloaded / panic = RED, the 262k-2card failure surface); the server
# answers a fresh sample after the stress. Reference: gpf-workspace-20260830 receipts.
set -uo pipefail
OUT=/root/out-hbatch/c3
RP="python3 /root/out-hbatch/run_pool.py"
mkdir -p "$OUT"

/root/out-hbatch/serve.sh start c3-on MEMRA_HYPER_BATCH=1 || exit 1

# vramwatch (1 Hz) for the whole cell
( while [ -f /root/out-hbatch/c3/.vramwatch ]; do
    echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' )" >> "$OUT/vramwatch.csv"
    sleep 1
  done ) & VW=$!
touch /root/out-hbatch/c3/.vramwatch

echo "--- burst A: c=12 ALL-DEEP (l3 WARM/A4630/B5550/C6470 x3 — concurrent prefill workspace stress) ---"
$RP conc --n 12 --picks 10,11,12,13,10,11,12,13,10,11,12,13 --out "$OUT/deep12" || echo "NOTE: burst A rows errored (check codes: 429 clean vs engine fail)"

echo "--- burst B: c=20 mixed (over MEMRA_MAX_SESSIONS=16 — session bar / queue behavior) ---"
$RP conc --n 20 --picks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,0,1,2,3,4,5 --out "$OUT/mixed20" || echo "NOTE: burst B rows errored (check codes)"

echo "--- post-stress health: server must still answer ---"
$RP sample --out "$OUT/post" && echo "POST_STRESS_SAMPLE=OK" || echo "POST_STRESS_SAMPLE=FAIL"

rm -f /root/out-hbatch/c3/.vramwatch; wait $VW 2>/dev/null || true

LOG=/root/out-hbatch/logs/boot-c3-on.log
{
  echo "admit_oom_lines=$(grep -c '\[admit-oom\]' "$LOG")"
  echo "overloaded_lines=$(grep -c 'Overloaded' "$LOG")"
  echo "shed_queue_lines=$(grep -c 'shed_queue' "$LOG")"
  echo "panic_lines=$(grep -cE 'panicked|FATAL' "$LOG")"
  echo "--- admit/shed line samples ---"
  grep -E '\[admit-oom\]|shed_queue' "$LOG" | head -10 || true
  echo "--- http codes burst A ---"
  python3 -c "import json;d=json.load(open('$OUT/deep12/conc-12-greedy.json'));print(sorted((r or {}).get('http_code') for r in d['rows']))" 2>/dev/null || true
  echo "--- http codes burst B ---"
  python3 -c "import json;d=json.load(open('$OUT/mixed20/conc-20-greedy.json'));print(sorted((r or {}).get('http_code') for r in d['rows']))" 2>/dev/null || true
  echo "--- peak vram per card (MiB) ---"
  python3 -c "
import csv
rows=[r for r in csv.reader(open('$OUT/vramwatch.csv')) if len(r)>3]
for i in range(1,5):
    vals=[int(r[i]) for r in rows if len(r)>i and r[i].strip().isdigit()]
    print(f'card{i-1} peak={max(vals) if vals else 0}')"
} > "$OUT/receipts.txt"
cat "$OUT/receipts.txt"

/root/out-hbatch/serve.sh stop
# GREEN = no engine-side failure: zero panic; any Overloaded line is RED
pan=$(grep -c 'panic_lines=0' "$OUT/receipts.txt" || true)
ovl=$(grep -c 'overloaded_lines=0' "$OUT/receipts.txt" || true)
if [ "$pan" -ge 1 ] && [ "$ovl" -ge 1 ]; then echo "C3_VERDICT=GREEN"; else echo "C3_VERDICT=CHECK (see receipts.txt)"; fi
