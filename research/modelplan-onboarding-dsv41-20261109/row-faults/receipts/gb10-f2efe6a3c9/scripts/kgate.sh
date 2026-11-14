#!/usr/bin/env bash
# GB10 kernel gate of each TAG's dsv41-gate on the random fixtures, MEMRA_FULL_PREC=1.
# usage: kgate.sh TAG ...   Done-file out/kgate/k.txt, marker KGATE-DONE. Logs out/kgate/kernels-TAG.{log,json}.
set -u
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
export MEMRA_FULL_PREC=1
O=$R/out/kgate; mkdir -p $O; : > $O/k.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/k.txt 2>&1
for TAG in "$@"; do
  B=$R/bins/$TAG
  sha256sum $B/dsv41-gate >> $O/k.txt
  timeout 20m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels-$TAG.json > $O/kernels-$TAG.log 2>&1
  rc=$?
  echo "kernels $TAG rc $rc $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels-$TAG.log | tail -1) $(grep -ac '^FAIL' $O/kernels-$TAG.log) FAIL lines" >> $O/k.txt
  grep -a "^FAIL" $O/kernels-$TAG.log | head -20 >> $O/k.txt
  grep -a "gemm_ffma_ring\|gemm_ffma2_ring\|gemm_ffma2_partial\|moe_plan/out_of_range" $O/kernels-$TAG.log >> $O/k.txt
done
echo "KGATE-DONE $(date +%T)" >> $O/k.txt
