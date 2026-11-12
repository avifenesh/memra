#!/usr/bin/env bash
# GB10: the cost of the fix, the grouped expert GEMMs' swapped forms (--form 0, by rows) against the normal form
# (--form 1), on the unchanged build's bench (c2), A B B A. Done-file out/perf-form/runs.txt, marker PERF-FORM-DONE.
set -u
R=$HOME/.cache/dsv41-cutlass
O=$R/out/perf-form; mkdir -p $O; : > $O/runs.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/runs.txt 2>&1
i=0
for f in 0 1 1 0; do
  timeout 8m $R/bins/c2/dsv41-bench gemm --form $f --reps 30 --out $O/form$f-$i.json > $O/form$f-$i.log 2>&1
  echo "form $f run $i rc $? $(date +%T)" >> $O/runs.txt
  i=$((i+1))
done
echo "PERF-FORM-DONE $(date +%T)" >> $O/runs.txt
