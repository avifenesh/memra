#!/usr/bin/env bash
# GB10 box, step 11 chain 4: the final head. Build, unit tests of the serve route, then one server
# at a time in a balanced order: the draft gate on (G), the gate off (F: five drafts every round,
# main's behavior), drafts off (P), the gate on over the BF16 drafter arm (B), then F and G again.
# Each runs the long bench and the six chat turns three times, sampled (temperature 0.9, seed
# 20261121); G1 and P also run the served-answer gates (greedy). Then the greedy turns drafts off,
# with the gate, and under the two draft-fault doors; the identity checks; check3 on hy4f-gpu.
# Log: logs/s11-chain4.log.
set -u
D=$HOME/hy4flash/s11
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN4 START $(date '+%F %T') $(cat src/HEAD.txt)"
./build-gpu.sh > $L/s11-build-h4.log 2>&1
grep -q BUILD_EXIT=0 $L/s11-build-h4.log || { echo BUILD-FAILED; exit 1; }
mkdir -p bins-h4 && cp target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu bins-h4/
sha256sum bins-h4/* | tee bins-h4.sha256
bash ./test-server.sh hy4f_serve > $L/s11-test-h4.log 2>&1; grep -E "test result|TEST_EXIT" $L/s11-test-h4.log
H=$D/bins-h4/memra-server
NV=$HOME/hy4flash/hf-stage
G=$D/s11-serve.sh
run() { bash $G "$@"; echo "$1 rc=$? $(date '+%T')"; }
export TURNS_RUNS=3
run f-G1 $H $NV gate,bench,turns
run f-F1 $H $NV bench,turns MEMRA_HY4F_DRAFT_GATE=0
run f-P $H $NV gate,bench,turns MEMRA_HY4F_DRAFT=0
run f-B $H $D/arm-bf16 bench,turns
run f-F2 $H $NV bench,turns MEMRA_HY4F_DRAFT_GATE=0
run f-G2 $H $NV bench,turns
export TURNS_GREEDY=1 TURNS_RUNS=1
run f-Pg $H $NV turns MEMRA_HY4F_DRAFT=0
run f-Gg $H $NV turns
run f-fault-draft $H $NV turns MEMRA_HY4F_DRAFT_FAULT=draft:3
run f-fault-backup $H $NV turns MEMRA_HY4F_DRAFT_FAULT=backup:3
unset TURNS_GREEDY
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
for t in f-G1 f-F1 f-B f-F2 f-G2; do
  python3 $I/bench_identity.py serve/f-P/bench.jsonl serve/$t/bench.jsonl serve/bench-identity-$t.json \
    > serve/bench-identity-$t.log 2>&1; echo "bench-identity $t rc=$?"
  python3 $I/turns_identity.py serve/f-P/turns.json serve/$t/turns.json --out serve/turns-identity-$t.json \
    > serve/turns-identity-$t.log 2>&1; echo "turns-identity $t rc=$?"
done
python3 $I/turns_identity.py serve/f-Pg/turns.json serve/f-Gg/turns.json --out serve/turns-identity-f-Gg.json \
  > serve/turns-identity-f-Gg.log 2>&1; echo "turns-identity greedy rc=$?"
for f in draft backup; do
  python3 $I/turns_identity.py serve/f-Pg/turns.json serve/f-fault-$f/turns.json --rounds 3 --log serve/f-fault-$f/server.log \
    --out serve/turns-identity-f-fault-$f.json > serve/turns-identity-f-fault-$f.log 2>&1; echo "turns-identity fault-$f rc=$?"
done
python3 $I/draft_identity.py serve/f-G1/chat-gate.json serve/f-P/chat-gate.json serve/draft-identity-f.json \
  > serve/draft-identity-f.log 2>&1; echo "draft-identity rc=$?"
bash $HOME/hy4flash/perf/check3.sh s11h4 $D/bins-h4/hy4f-gpu; tail -n 1 $L/perf-check3-s11h4.log
echo "CHAIN4-DONE $(date '+%F %T')"
