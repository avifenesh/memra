#!/usr/bin/env bash
# GB10 box, step 11 chain 1: build the head, then one server at a time, in a balanced order
# (A B C D E F, then D C B A): the head with the draft gate (h-gate), the head drafting the full
# five every round (h-fixed, MEMRA_HY4F_DRAFT_GATE=0), main 63f806baf9 (greedy drafts), drafts
# off (plain), and the head drafting five with the BF16 drafter arms (bf16: every drafter tensor
# from tencent/Hy4-flash@d88c9dfd; bf16mlp: only the MLPs and the hidden correction). Each runs
# the long bench and the six chat turns three times, sampled (temperature 0.9, seed 20261121).
# Then the identity checks against plain, the greedy served-answer gates with drafts on and off,
# and check3 on the head's hy4f-gpu. Log: logs/s11-chain1.log.
set -u
D=$HOME/hy4flash/s11
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN1 START $(date '+%F %T') $(cat src/HEAD.txt)"
./build-gpu.sh > $L/s11-build-h1.log 2>&1
grep -q BUILD_EXIT=0 $L/s11-build-h1.log || { echo BUILD-FAILED; exit 1; }
mkdir -p bins-h1 && cp target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu bins-h1/
sha256sum bins-h1/* bins-main/* | tee bins.sha256
H=$D/bins-h1/memra-server
M=$D/bins-main/memra-server
NV=$HOME/hy4flash/hf-stage
G=$D/s11-serve.sh
run() { bash $G "$@"; echo "$1 rc=$? $(date '+%T')"; }
export TURNS_RUNS=3
run h-gate $H $NV bench,turns
run h-fixed $H $NV bench,turns MEMRA_HY4F_DRAFT_GATE=0
run main $M $NV bench,turns
run plain $H $NV bench,turns MEMRA_HY4F_DRAFT=0
run bf16 $H $D/arm-bf16 bench,turns MEMRA_HY4F_DRAFT_GATE=0
run bf16mlp $H $D/arm-bf16-mlp bench,turns MEMRA_HY4F_DRAFT_GATE=0
run plain-b $H $NV bench,turns MEMRA_HY4F_DRAFT=0
run main-b $M $NV bench,turns
run h-fixed-b $H $NV bench,turns MEMRA_HY4F_DRAFT_GATE=0
run h-gate-b $H $NV bench,turns
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
for t in h-gate h-fixed main bf16 bf16mlp main-b h-fixed-b h-gate-b plain-b; do
  python3 $I/bench_identity.py serve/plain/bench.jsonl serve/$t/bench.jsonl serve/bench-identity-$t.json \
    > serve/bench-identity-$t.log 2>&1; echo "bench-identity $t rc=$?"
  python3 $I/turns_identity.py serve/plain/turns.json serve/$t/turns.json --out serve/turns-identity-$t.json \
    > serve/turns-identity-$t.log 2>&1; echo "turns-identity $t rc=$?"
done
# greedy: the served-answer gates with the gate on, and drafts off, answer by answer
run gate-h $H $NV gate
run gate-plain $H $NV gate MEMRA_HY4F_DRAFT=0
python3 $I/draft_identity.py serve/gate-h/chat-gate.json serve/gate-plain/chat-gate.json serve/draft-identity-h.json \
  > serve/draft-identity-h.log 2>&1; echo "draft-identity rc=$?"
bash $HOME/hy4flash/perf/check3.sh s11h $D/bins-h1/hy4f-gpu; tail -n 1 $L/perf-check3-s11h.log
echo "CHAIN1-DONE $(date '+%F %T')"
