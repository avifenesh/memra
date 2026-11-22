#!/usr/bin/env bash
# GB10 box, step 11: gather the receipts into $D/receipts (no weights, no logits arrays)
set -u
D=$HOME/hy4flash/s11
R=$D/receipts
cd $D
rm -rf $R
mkdir -p $R/serve $R/replay $R/arms $R/logs $R/scripts
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
# the chain 1 arms against the final drafts-off run: the same target, the same texts
for t in main h-fixed h-gate; do
  python3 $I/bench_identity.py serve/f-P/bench.jsonl serve/$t/bench.jsonl serve/bench-identity-$t.json \
    > serve/bench-identity-$t.log 2>&1; echo "bench-identity $t rc=$?"
  python3 $I/turns_identity.py serve/f-P/turns.json serve/$t/turns.json --out serve/turns-identity-$t.json \
    > serve/turns-identity-$t.log 2>&1; echo "turns-identity $t rc=$?"
done
for s in serve/*; do
  t=$(basename $s)
  [ -d $s ] || { cp $s $R/serve/; continue; }
  [ "$t" = plain ] && continue  # chain 1's drafts-off run was stopped during its boot
  mkdir -p $R/serve/$t
  for f in run.txt chat-gate.json chat-gate.log bench.jsonl bench.log turns.json turns.log; do
    [ -f $s/$f ] && cp $s/$f $R/serve/$t/
  done
  [ -f $s/telemetry.csv ] && gzip -9 -n -c $s/telemetry.csv > $R/serve/$t/telemetry.csv.gz
  [ -f $s/server.log ] && grep -E "hy4f-serve|admission|admit-mem|hy4f-draft|hy4f-step" $s/server.log | gzip -9 -n > $R/serve/$t/server-events.log.gz
done
for f in replay/*.json replay/*.jsonl; do gzip -9 -n -c $f > $R/replay/$(basename $f).gz; done
for f in replay/*.log; do grep -v "layer .* resident" $f > $R/replay/$(basename $f); done
cp replay-prompts.json turns-prompts.json $R/replay/
for a in arm-bf16 arm-bf16-mlp; do
  mkdir -p $R/arms/$a
  cp $a/config.json $a/hf_quant_config.json $a/quant_config.json $R/arms/$a/
  gzip -9 -n -c $a/model.safetensors.index.json > $R/arms/$a/model.safetensors.index.json.gz
  sha256sum $a/model-nonexpert-09.safetensors > $R/arms/$a/model-nonexpert-09.sha256
done
sha256sum src/model-00148-of-00148.safetensors > $R/arms/source-shard-148.sha256
for f in $HOME/hy4flash/logs/s11-*.log $HOME/hy4flash/logs/perf-check3-s11h4.log; do
  case $(basename $f) in s11-build*) grep -E "BUILD_EXIT|^error|Finished" $f > $R/logs/$(basename $f);; *) cp $f $R/logs/;; esac
done
V=$R/verify-d43aacce37
mkdir -p $V
for f in artifact.lock bins.sha256 checkpoint-parity.tsv execution-rewrites.tsv gates.txt normalized-config.txt \
         oracle-request.tsv serve-gate.tsv serve-response.json tokenizer-contract.tsv; do
  [ -f verify/head/$f ] && cp verify/head/$f $V/
done
for f in model-plan.txt tensor-census.tsv serve.log; do
  [ -f verify/head/$f ] && gzip -9 -n -c verify/head/$f > $V/$f.gz
done
cp $HOME/hy4flash/logs/s11-verify.log $V/run.log
cp bins*.sha256 $R/
cp *.sh *.py $R/scripts/ 2>/dev/null
du -sh $R
find $R -type f -size +300k -exec ls -la {} \;
echo RECEIPTS-DONE
