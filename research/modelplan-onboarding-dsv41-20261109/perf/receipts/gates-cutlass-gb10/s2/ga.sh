#!/usr/bin/env bash
# GB10 gates A: the kernel gate, then the layer gate saving its tensors, counted against the CUTLASS build's set
# (fa825eee..., out/gates-f1|f4/save-c.sha256 of ~/.cache/dsv41-cutlass). Done-file out/gates-$TAG/a.txt, GATES-A-DONE.
set -u
TAG=$1; FORMS=${2:-d}; [ "$FORMS" = none ] && FORMS=""
R=$HOME/.cache/dsv41-cfix
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/a.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/a.txt; }
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/a.txt 2>&1
timeout 20m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
say "kernels" $? $O/kernels.log
grep -a "^FAIL" $O/kernels.log | head -30 >> $O/a.txt
echo "kernels: $(grep -ac '^PASS schedule/dense_small' $O/kernels.log) dense_small PASS, $(grep -ac '^PASS schedule/moe_small_swiglu\|^PASS schedule/moe_small_own_layout\|^PASS schedule/act_quant_sf_split' $O/kernels.log) split-path PASS, $(grep -ac '^PASS schedule/fused' $O/kernels.log) fused PASS" >> $O/a.txt
for F in $FORMS; do
  XF=""; [ $F = d ] || XF="--dense-form $F"
  rm -rf $R/save-$TAG-$F; mkdir -p $R/save-$TAG-$F
  timeout 16m $B/dsv41-gate $XF layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
    --floor $D/captures-v2/needle-floor --save $R/save-$TAG-$F --out $O/layer-$F.json > $O/layer-$F.log 2>&1
  say "layer form $F" $? $O/layer-$F.log
  grep -a "^FAIL" $O/layer-$F.log | head -10 >> $O/a.txt
  (cd $R/save-$TAG-$F && find . -name '*.bin' | sort | xargs sha256sum) > $O/save-$F.sha256
  sha256sum $O/save-$F.sha256 >> $O/a.txt
  for ref in $HOME/.cache/dsv41-cutlass/out/gates-f4/save-c.sha256 $HOME/.cache/dsv41-cutlass/out/gates-f1/save-c.sha256; do
    [ -f $ref ] || continue
    echo "layer form $F saved tensors against $ref: $(cmp -s <(awk '{print $1, $2}' $O/save-$F.sha256) <(awk '{print $1, $2}' $ref) && echo IDENTICAL || echo DIFFER) ($(wc -l < $O/save-$F.sha256) files; ref sha $(sha256sum $ref | cut -c1-16))" >> $O/a.txt
  done
done
echo "GATES-A-DONE $(date +%T)" >> $O/a.txt
