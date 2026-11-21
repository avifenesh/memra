#!/usr/bin/env bash
# GB10 box, step 10: the readings, sampled at the artifact's generation_config.json (temperature
# 0.9, no top-k, no top-p, seed 20261121), with the pin's memra-server: drafts on, off, on
# again, each the long bench and the six chat turns three times; then the checks that both arms
# generated the same text. Each server alone.
set -u
D=$HOME/hy4flash/pr10
cd $D
echo "SAMPLED-CHAIN START $(date '+%F %T') $(cat src/HEAD.txt)"
sha256sum target/release/memra-server
G=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts/serve_gate.sh
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
export TURNS_RUNS=3
D=$D bash $G sampled-spec5 bench,turns; echo "sampled-spec5 rc=$?"
D=$D bash $G sampled-plain bench,turns MEMRA_HY4F_DRAFT=0; echo "sampled-plain rc=$?"
D=$D bash $G sampled-spec5-b bench,turns; echo "sampled-spec5-b rc=$?"
for t in sampled-spec5 sampled-spec5-b; do
  python3 $I/bench_identity.py serve/sampled-plain/bench.jsonl serve/$t/bench.jsonl serve/bench-identity-$t.json \
    > serve/bench-identity-$t.log 2>&1; echo "bench-identity $t rc=$?"
  python3 $I/turns_identity.py serve/sampled-plain/turns.json serve/$t/turns.json --out serve/turns-identity-$t.json \
    > serve/turns-identity-$t.log 2>&1; echo "turns-identity $t rc=$?"
done
echo "SAMPLED-CHAIN-DONE $(date '+%F %T')"
