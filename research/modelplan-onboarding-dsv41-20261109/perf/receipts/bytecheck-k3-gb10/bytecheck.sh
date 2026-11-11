#!/usr/bin/env bash
# GB10 (gb10, under a sparklock lease): does a 121a build write the bytes a 120a build writes?
# 1. o2 (a449d3183c) built for 121a: the layer gate's saved tensors against the reference set the 120a o2 wrote
#    (the 120a set: 111 of 111 equal to the 120a o2 build's)
# 2. only if all 111 are equal: k3's layer gate with --save = the new reference set (save-k3), then k3's one-card
#    gates (kernels, tp on 2 and 4 ranks of the card, decode on three captures)
set -u
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
O=$R/out/bytecheck
mkdir -p $O
: > $O/gates.txt
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1)" >> $O/gates.txt; }
cmpset() {  # REF NEW -> "n compared, m differ"
  local n=0 bad=0 f
  for f in $(cd $1 && find . -name '*.bin' | sort); do
    n=$((n+1)); cmp -s $1/$f $2/$f || { bad=$((bad+1)); echo "DIFF $f" >> $O/diff-$3.txt; }
  done
  echo "$3: $n compared, $bad differ" >> $O/gates.txt
  [ $bad = 0 ]
}
rm -rf $R/save-o2; mkdir -p $R/save-o2
$R/bins/o2/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-o2 --out $O/layer-o2.json > $O/layer-o2.log 2>&1
say "layer o2 (121a)" $? $O/layer-o2.log
if cmpset $D/scratch/perf-save-multi-f34g/A $R/save-o2 "o2 121a against o2 120a"; then
  rm -rf $R/save-k3; mkdir -p $R/save-k3
  $R/bins/k3/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
    --floor $D/captures-v2/needle-floor --save $R/save-k3 --out $O/layer-k3.json > $O/layer-k3.log 2>&1
  say "layer k3 (121a, the new reference set)" $? $O/layer-k3.log
  cmpset $D/scratch/perf-save-multi-f34g/A $R/save-k3 "k3 against the old set (57 expected to move)" || true
  (cd $R/save-k3 && find . -name '*.bin' | sort | xargs sha256sum) > $O/save-k3.sha256
  sha256sum $O/save-k3.sha256 >> $O/gates.txt
  $R/bins/k3/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels-k3.json > $O/kernels-k3.log 2>&1
  say "kernels k3" $? $O/kernels-k3.log
  for dev in 0,0 0,0,0,0; do
    MEMRA_TP_TRANSPORT=peer-pull $R/bins/k3/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 \
      --devices $dev --decode 2 --token-map $TM --out $O/tp-$dev-peer-pull.json > $O/tp-$dev-peer-pull.log 2>&1
    say "tp $dev peer-pull k3" $? $O/tp-$dev-peer-pull.log
  done
  $R/bins/k3/dsv41-gate decode $C $D/dec/captures/straddle126 --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
    --out $O/decode-straddle126.json > $O/decode-straddle126.log 2>&1
  say "decode straddle126 k3" $? $O/decode-straddle126.log
  $R/bins/k3/dsv41-gate decode $C $D/dec/captures/gsm --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
    --out $O/decode-gsm.json > $O/decode-gsm.log 2>&1
  say "decode gsm k3" $? $O/decode-gsm.log
  $R/bins/k3/dsv41-gate decode $C $D/captures-v2/capital --layers 1,2,3,14,20,21,24 --token-map $TM \
    --out $O/decode-capital.json > $O/decode-capital.log 2>&1
  say "decode capital k3" $? $O/decode-capital.log
fi
echo "BYTECHECK-DONE $(date +%T)" >> $O/gates.txt
