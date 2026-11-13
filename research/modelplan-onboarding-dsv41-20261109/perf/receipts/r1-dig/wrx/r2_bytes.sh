#!/bin/bash
# wrx, inside a lease of one card (device 0 of the lease): r2's bytes against kpc's on the PRO 6000.
#   1. the layer gate's saved tensors (layers 1,2,14,20,21,24 of the needle capture, prefill + 2 decode steps) of kpc
#      and of r2, compared file by file (want 111 of 111 equal);
#   2. dsv41-bench gemm of both (the router's fp32 call at 1/4/5/8 rows: output hashes equal; the call times).
# usage: r2_bytes.sh   -> receipts-perf/r2-bytes/
set -u
D=/home/access/dsv41
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
R=$D/receipts-perf/r2-bytes
mkdir -p $R
export MEMRA_FULL_PREC=1
: > $R/bytes.txt
sha256sum $D/perf-bins/kpc/dsv41-gate $D/perf-bins/r2/dsv41-gate >> $R/bytes.txt
rm -rf /home/access/dsv41/tmp/save-kpc /home/access/dsv41/tmp/save-r2
for b in kpc r2; do
  mkdir -p $D/tmp/save-$b
  $D/perf-bins/$b/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
    --floor $D/captures-v2/needle-floor --save $D/tmp/save-$b --out $R/layer-$b.json > $R/layer-$b.log 2>&1
  echo "layer $b rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $R/layer-$b.log | tail -1)" >> $R/bytes.txt
done
n=0; bad=0
for f in $(cd $D/tmp/save-kpc && find . -name '*.bin' | sort); do
  n=$((n+1)); cmp -s $D/tmp/save-kpc/$f $D/tmp/save-r2/$f || { bad=$((bad+1)); echo "DIFF $f" >> $R/bytes.txt; }
done
echo "layer saved tensors r2 against kpc: $n compared, $bad differ" >> $R/bytes.txt
rm -rf /home/access/dsv41/tmp/save-kpc /home/access/dsv41/tmp/save-r2
i=0
for b in kpc r2 r2 kpc; do
  i=$((i+1))
  $D/perf-bins/$b/dsv41-bench gemm --reps 50 --out $R/gemm-$i-$b.json > $R/gemm-$i-$b.log 2>&1
  echo "gemm run $i $b rc $?" >> $R/bytes.txt
  grep -a "gemm ffma" $R/gemm-$i-$b.log >> $R/bytes.txt
done
echo "R2-BYTES-DONE $(date +%T)" >> $R/bytes.txt
