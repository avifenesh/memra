#!/usr/bin/env bash
# GB10 gates lease A (cutlass branch): the kernel gate, then the layer gate saving the new reference set (counted
# against k3's and bs1's sets). Done-file out/gates-$TAG/a.txt, marker GATES-A-DONE.
set -u
TAG=$(cat $HOME/.cache/dsv41-cutlass/build-tag)
R=$HOME/.cache/dsv41-cutlass
S=$HOME/.cache/dsv41-speed
D=$S/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/a.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/a.txt; }
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/a.txt 2>&1
timeout 12m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
say "kernels" $? $O/kernels.log
grep -a "^FAIL" $O/kernels.log | head -30 >> $O/a.txt
rm -rf $HOME/.cache/dsv41-cutlass/save-c
mkdir -p $R/save-c
timeout 14m $B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-c --out $O/layer.json > $O/layer.log 2>&1
say "layer" $? $O/layer.log
grep -a "^FAIL" $O/layer.log | head -10 >> $O/a.txt
for ref in $S/save-k3 $HOME/.cache/dsv41-bsmma/save-bs1; do
  n=0; bad=0
  for f in $(cd $ref && find . -name '*.bin' | sort); do
    n=$((n+1)); cmp -s $ref/$f $R/save-c/$f || bad=$((bad+1))
  done
  echo "layer saved tensors against $(basename $ref): $n compared, $bad differ" >> $O/a.txt
done
(cd $R/save-c && find . -name '*.bin' | sort | xargs sha256sum) > $O/save-c.sha256
sha256sum $O/save-c.sha256 >> $O/a.txt
echo "GATES-A-DONE $(date +%T)" >> $O/a.txt
