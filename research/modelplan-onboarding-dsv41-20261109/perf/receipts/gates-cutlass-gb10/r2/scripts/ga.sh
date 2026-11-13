#!/usr/bin/env bash
# GB10 gates A at #1286's restacked head: the kernel gate, then the layer gate on the default dense form saving its
# tensors, counted against the CUTLASS build's set (ref/save-c.sha256 = receipts gates-cutlass-gb10/f4/save-c.sha256,
# fa825eee...). Every line names its dense form. usage: ga.sh TAG   Done-file out/gates-TAG/a.txt, GATES-A-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-cfix2
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/gates-$TAG; mkdir -p $O; : > $O/a.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/a.txt; }
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/a.txt 2>&1
sha256sum $B/dsv41-gate >> $O/a.txt
timeout 20m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
say "kernels (every form, by construction)" $? $O/kernels.log
grep -a "^FAIL" $O/kernels.log | head -30 >> $O/a.txt
echo "kernels: $(grep -ac '^PASS schedule/dense_small' $O/kernels.log) dense_small PASS, $(grep -ac '^PASS schedule/moe_small_swiglu\|^PASS schedule/moe_small_own_layout\|^PASS schedule/act_quant_sf_split' $O/kernels.log) split-path PASS, $(grep -ac '^PASS schedule/fused' $O/kernels.log) fused PASS" >> $O/a.txt
rm -rf $R/save-$TAG; mkdir -p $R/save-$TAG
timeout 16m $B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-$TAG --out $O/layer.json > $O/layer.log 2>&1
say "layer, dense form default" $? $O/layer.log
grep -a "^FAIL" $O/layer.log | head -10 >> $O/a.txt
(cd $R/save-$TAG && find . -name '*.bin' | sort | xargs sha256sum) > $O/save.sha256
sha256sum $O/save.sha256 >> $O/a.txt
echo "layer, dense form default, saved tensors against the CUTLASS set: $(cmp -s <(awk '{print $1, $2}' $O/save.sha256) <(awk '{print $1, $2}' $R/ref/save-c.sha256) && echo IDENTICAL || echo DIFFER) ($(wc -l < $O/save.sha256) files; ref sha $(sha256sum $R/ref/save-c.sha256 | cut -c1-16))" >> $O/a.txt
echo "GATES-A-DONE $(date +%T)" >> $O/a.txt
