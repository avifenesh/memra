#!/usr/bin/env bash
# Pooled rows draft exactness cells (decode-batch-bench --pooled-draft) on one model: the serial
# draft chain against the pooled rows draft, per arm (graph greedy, graph sampled pure-temperature
# and filtered, eager sampled with and without penalties). Pass = every cell prints ROWS-STEP and
# ROUND-DRAFT "BITWISE EQUAL" and no MISMATCH.
# usage: probe_cells.sh BENCH MODEL OUTDIR   (the caller holds the GPU lease)
set -u
BENCH=$1
MODEL=$2
OUT=$3
mkdir -p "$OUT"
sha256sum "$BENCH" > "$OUT/bin.sha256"
: > "$OUT/cells.txt"
cell() {
    local name=$1
    shift
    echo "== $name: ${*/$HOME/\~}" | tee -a "$OUT/cells.txt"
    "$@" > "$OUT/$name.log" 2>&1
    echo "$name rc=$?" | tee -a "$OUT/cells.txt"
    grep -E "ROWS-STEP|ROUND-DRAFT|DRAFT-TIME|member|session" "$OUT/$name.log" | tee -a "$OUT/cells.txt"
}
cell greedy-ragged "$BENCH" "$MODEL" --pooled-draft 3,3,2,4 --ctx 512 --reps 5
cell mixed-ragged "$BENCH" "$MODEL" --pooled-draft 3,2,4,3 --ctx 512 --reps 5 --temp 0.8
cell sampled-pmin env MEMRA_SPEC_PMIN=0.3 "$BENCH" "$MODEL" --pooled-draft 4,4,3,2 --ctx 512 --reps 5 --temp 0.8 --all-sampled
cell filtered "$BENCH" "$MODEL" --pooled-draft 3,3,3,3 --ctx 512 --reps 5 --temp 0.8 --top-p 0.9 --all-sampled
cell penalized "$BENCH" "$MODEL" --pooled-draft 3,3,3,3 --ctx 512 --reps 5 --temp 0.8 --pen --all-sampled
cell eager-sampled env MEMRA_SPEC_NOGRAPH=1 "$BENCH" "$MODEL" --pooled-draft 3,2,4,3 --ctx 512 --reps 5 --temp 0.8 --all-sampled
cell eager-filtered env MEMRA_SPEC_NOGRAPH=1 "$BENCH" "$MODEL" --pooled-draft 3,3,3,3 --ctx 512 --reps 5 --temp 0.8 --top-p 0.9 --all-sampled
n=$(grep -c "rc=0" "$OUT/cells.txt")
bad=$(grep -c "MISMATCH" "$OUT/cells.txt")
eq=$(grep -c "ROUND-DRAFT.*BITWISE EQUAL" "$OUT/cells.txt")
echo "cells rc=0: $n of 7, ROUND-DRAFT equal: $eq, MISMATCH lines: $bad" | tee -a "$OUT/cells.txt"
[ "$n" = 7 ] && [ "$eq" = 7 ] && [ "$bad" = 0 ] && echo "POOLED-DRAFT PROBE: PASS" | tee -a "$OUT/cells.txt" && exit 0
echo "POOLED-DRAFT PROBE: FAIL" | tee -a "$OUT/cells.txt"
exit 1
