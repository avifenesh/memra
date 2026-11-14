#!/usr/bin/env bash
# Runs on the Spark under a sparklock lease. Review of memra #1312, phase C: repeat counts.
# 20 more passes of mutations 1, 3 and 6 (the timing-dependent ones), 10 more interleaved
# head/base pairs for the run time, then one tar of the logs. No build.
# Done-file out/c.txt, marker C-DONE.
set -u
R=$HOME/.cache/review-1312
D=$HOME/.cache/dsv41-speed/data
O=$R/out
: > "$O/c.txt"
: > "$O/c-fails.txt"
export MEMRA_FULL_PREC=1
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> "$O/c.txt" 2>&1
for pass in $(seq 1 20); do
  for m in 1 3 6; do
    REVIEW_MUT=$m timeout 5m "$R/bins/mut/dsv41-gate" kernels "$D/fixtures-gpu/rand" --out "$O/c-tmp.json" > "$O/c-tmp.log" 2>&1
    rc=$?
    echo "mut $m pass $pass rc $rc $(grep -ao '[0-9]* checks, [0-9]* failed' "$O/c-tmp.log" | tail -1)" >> "$O/c.txt"
    grep -a "^FAIL" "$O/c-tmp.log" | cut -d: -f1 | sed "s|^FAIL schedule/|mut $m pass $pass |; s|^FAIL |mut $m pass $pass |" >> "$O/c-fails.txt"
  done
done
for i in $(seq 1 10); do
  for tag in head base; do
    t0=$(date +%s.%N)
    timeout 5m "$R/bins/$tag/dsv41-gate" kernels "$D/fixtures-gpu/rand" --out "$O/c-tmp.json" > "$O/c-tmp.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    echo "time $tag $i rc $rc wall $(echo "$t1 - $t0" | bc) s $(grep -ao '[0-9]* checks, [0-9]* failed' "$O/c-tmp.log" | tail -1)" >> "$O/c.txt"
  done
done
rm -f "$O/c-tmp.json" "$O/c-tmp.log"
echo "C-DONE $(date +%T)" >> "$O/c.txt"
cd "$R" && tar -czf "$R/review-1312-spark-out.tgz" out
