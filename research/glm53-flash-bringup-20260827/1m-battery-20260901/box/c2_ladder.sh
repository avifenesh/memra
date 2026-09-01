#!/usr/bin/env bash
# CELL 2 — THE DEPTH LADDER, PLAIN ARM (timed, marker held by the caller).
#
# Greedy steady-state decode tok/s vs depth on the CURRENT head, at the demo's own rungs
# and from the demo's own char slices, so each row is directly comparable:
#   16k  64,400 chars   -> 15,766 tok   demo decode 24.47
#   131k 527,000        -> 128,566      demo 22.82
#   262k 1,054,000      -> 257,775      demo 21.13
#   525k 2,161,700      -> 525,616      demo 18.92
#   1M   4,282,700      -> 1,035,357    demo 16.04 (span) / 15.67 (steady p50) - CELL 1 owns it
# Greedy here is the INSTRUMENT (byte-deterministic), not the product; the ship-config arm
# in cell 3 carries the vendor-default sampled rows. Rungs run ASCENDING on one boot: depth
# is the variable, not the arm, so single-boot rows are correct (the interleave law binds
# where ARMS are compared - cell 3's spec-vs-plain reading, not a rung sequence).
# Each rung caps max_tokens (loop-damage bound) and uses the REAL sha-banked corpus.
# usage: c2_ladder.sh [rung...]   default: R16K R131K R262K R525K
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c2
mkdir -p "$D"
RUNGS=("$@"); [ ${#RUNGS[@]} -eq 0 ] && RUNGS=(R16K R131K R262K R525K)
declare -A CH=( [R16K]=64400 [R131K]=527000 [R262K]=1054000 [R525K]=2161700 [R1M]=4282700 )
declare -A MT=( [R16K]=128   [R131K]=128    [R262K]=128     [R525K]=128     [R1M]=256 )
{
date -u +%FT%TZ
echo "######## CELL 2: PLAIN DEPTH LADDER (timed) rungs=${RUNGS[*]} ########"
bash "$OUT/serve.sh" start c2-plain || { echo "C2_EXIT=BOOTFAIL"; exit 1; }
bash "$OUT/vramwatch.sh" "$D/vram.csv" 5 & VW=$!
echo "vramwatch pid $VW"
echo; echo "=== warm rung (arena populate before the ladder) ==="
bash "$OUT/rung.sh" c2-plain W1K 4200 32 greedy "$D"
bash "$OUT/serve.sh" engage c2-plain || echo "C2_WARN=engage-red"
for r in "${RUNGS[@]}"; do
  echo; echo "=== $r greedy (chars=${CH[$r]}) ==="
  bash "$OUT/rung.sh" c2-plain "$r" "${CH[$r]}" "${MT[$r]}" greedy "$D" \
    || echo "C2_WARN=$r failed, the failure is the receipt"
done
kill "$VW" 2>/dev/null && echo "vramwatch $VW stopped"
echo; echo "--- per-card VRAM peak over cell 2 ---"
python3 - "$D/vram.csv" <<'PY'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0] != "ts"]
for g in range(4):
    pk = max(int(r[g+1]) for r in rows)
    print(f"  gpu{g}: peak {pk} MiB ({100*pk/97887:.1f}%)")
PY
echo "--- boot-wide error census (must be 0) ---"
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY" "$OUT/logs/boot-c2-plain.log"
bash "$OUT/serve.sh" stop
date -u +%FT%TZ
echo "C2_DONE"
} 2>&1 | tee "$OUT/logs/c2.log"
