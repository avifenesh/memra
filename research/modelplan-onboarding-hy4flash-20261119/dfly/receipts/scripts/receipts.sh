#!/usr/bin/env bash
# GB10 box, step 10: gather the drafter's receipts (no logits arrays) into $D/receipts
# usage: receipts.sh PIN10 (the 10-hex commit the pin chain ran on)
set -u
pin=$1
D=$HOME/hy4flash/pr10
R=$D/receipts
P=$HOME/hy4flash/pr7/venv/bin/python
C=$D/lane/dfly/compare_dfly.py
cd $D
rm -rf $D/receipts
mkdir -p $R/tapes $R/compare $R/target $R/logs $R/gen $R/serve
for d in dtapes/* rtapes/* gtapes/*; do
  n=$(echo $d | tr / -)
  gzip -9 -n -c $d/tape.json > $R/tapes/$n.json.gz
  sha256sum $d/logits.f32 | sed "s#  .*#  $d/logits.f32#" >> $R/tapes/logits.sha256
done
for t in target/*; do [ -f $t/target.json ] && cp $t/target.json $R/target/$(basename $t).json; done
for q in chat long; do
  $P $C dtapes/$q-f64 rtapes/$q --atol 1e-5 --json $R/compare/ref-$q.json > /dev/null
  $P $C dtapes/$q-f64 gtapes/$q --json $R/compare/gpu-drafter-only-$q.json > /dev/null
  $P $C dtapes/$q-f64 dtapes/$q-bf16x2af --json $R/compare/band-drafter-only-$q.json > /dev/null
  $P $C dtapes/$q-f64 dtapes/$q-x2-x2 --json $R/compare/band-composite-$q.json > /dev/null
  $P $C dtapes/$q-f64 gtapes/$q-route --atol 0.0398 --json $R/compare/gpu-composite-$q.json > /dev/null
  $P $C dtapes/$q-f64 gtapes/$q-route-pin --atol 0.0398 --json $R/compare/gpu-composite-$q-$pin.json > /dev/null
  echo "composite $q $pin rc=$?"
  $P $C dtapes/$q-f64 gtapes/$q-pin --json $R/compare/gpu-drafter-only-$q-$pin.json > /dev/null
  for r in layers width no-correction topk; do
    $P $C dtapes/$q-f64 rtapes/$q-red-$r --atol 1e-5 --json $R/compare/ref-red-$r-$q.json > /dev/null
    $P $C dtapes/$q-f64 gtapes/$q-red-$r --atol 0.0398 --json $R/compare/gpu-red-$r-$q.json > /dev/null
    $P $C dtapes/$q-f64 gtapes/$q-red-$r-pin --atol 0.0398 --json $R/compare/gpu-red-$r-$q-$pin.json > /dev/null
    echo "red $r $q $pin rc=$? (1 = the arm fails the gate)"
  done
done
cp gen/*.json $R/gen/ 2>/dev/null
for f in gen-pin/*.json; do cp $f $R/gen/$(basename $f .json)-$pin.json; done
for f in $HOME/hy4flash/logs/pr10-*.log; do
  case $(basename $f) in pr10-build*) continue;; esac
  grep -v "UserWarning\|torch.from_numpy" $f > $R/logs/$(basename $f)
done
for s in serve/*; do
  t=$(basename $s)
  [ -d $s ] || { cp $s $R/serve/$(basename $s | sed "s/-pin/-$pin/"); continue; }
  t=$(echo $t | sed "s/-pin/-$pin/")
  mkdir -p $R/serve/$t
  for f in run.txt chat-gate.json chat-gate.log bench.jsonl bench.log turns.json turns.log; do
    [ -f $s/$f ] && cp $s/$f $R/serve/$t/
  done
  [ -f $s/server.log ] && grep -E "hy4f-serve|admission|admit-mem|hy4f-draft" $s/server.log | gzip -9 -n > $R/serve/$t/server-events.log.gz
done
# the verify bundle of the pin, as step 9 keeps it
V=$R/verify-$pin
mkdir -p $V
for f in artifact.lock bins.sha256 checkpoint-parity.tsv execution-rewrites.tsv gates.txt normalized-config.txt \
         oracle-request.tsv serve-gate.tsv serve-response.json tokenizer-contract.tsv; do
  [ -f verify/pin/$f ] && cp verify/pin/$f $V/
done
for f in model-plan.txt tensor-census.tsv serve.log; do
  [ -f verify/pin/$f ] && gzip -9 -n -c verify/pin/$f > $V/$f.gz
done
(cd verify/pin && sha256sum model-plan.txt tensor-census.tsv serve.log) > $V/raw.sha256
cp $HOME/hy4flash/logs/pr10-verify-pin.log $V/run.log
if [ -d verify/tiny ]; then
  T=$R/verify-$pin-tiny
  mkdir -p $T
  for f in verify/tiny/*; do
    [ -f $f ] || continue
    case $f in *.tsv) case $(basename $f) in reference-oracle*) gzip -9 -n -c $f > $T/$(basename $f).gz; continue;; esac;; esac
    cp $f $T/
  done
fi
cp bins.sha256 $R/ 2>/dev/null
cp bins-pin.sha256 $R/bins-$pin.sha256
mkdir -p $R/scripts; cp *.sh $R/scripts/
du -sh $R
find $R -type f -size +300k -exec ls -la {} \;
echo RECEIPTS-DONE
