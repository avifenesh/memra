#!/usr/bin/env bash
# GB10 box, step 10: the record's chain at one commit (the pin): build, the route identity, the drafter
# gates (composite on the route, drafter-only and its red arms), verify checkpoint + serve, then
# the serve gates, readings and chat turns with drafts on, off and on again, the chat turns with a
# refused draft round and a refused ring backup, and the answer comparisons. Each GPU job alone.
# One log: logs/pr10-pin-chain.log. Stop it by killing this script's process group.
set -u
D=$HOME/hy4flash/pr10
L=$HOME/hy4flash/logs
cd $D
echo "PIN-CHAIN START $(date '+%F %T') $(cat src/HEAD.txt)"
SERVER=1 JOBS=16 ./build-gpu.sh > $L/pr10-build-pin.log 2>&1
grep -q BUILD_EXIT=0 $L/pr10-build-pin.log || { echo BUILD-FAILED; exit 1; }
sha256sum target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu target/release/memra > $D/bins-pin.sha256
S=$D/target/release/hy4f-spec
R=$HOME/hy4flash/pr7/requests
mkdir -p gen-pin
$S gen $HOME/hy4flash/hf-stage $R/chat.json 128 --bench 3 > gen-pin/chat-k5.json 2> gen-pin/chat-k5.log; echo "gen chat rc=$?"
$S gen $HOME/hy4flash/hf-stage $R/long.json 64 --prefill-chunk 2048 --bench 3 > gen-pin/long-k5.json 2> gen-pin/long-k5.log; echo "gen long rc=$?"
for req in chat long; do
  $S gate-route $HOME/hy4flash/hf-stage $R/$req.json $HOME/hy4flash/pr7/tapes/$req-f64 dtapes/$req-f64 gtapes/$req-route-pin \
    > $L/pr10-gate-route-$req-pin.log 2>&1; echo "gate-route $req rc=$?"
  timeout 900 $S gate $HOME/hy4flash/hf-stage $R/$req.json target/$req dtapes/$req-f64 gtapes/$req-pin \
    > $L/pr10-gate-$req-pin.log 2>&1; echo "gate $req rc=$?"
  for r in layers width no-correction topk; do
    timeout 900 $S gate $HOME/hy4flash/hf-stage $R/$req.json target/$req dtapes/$req-f64 gtapes/$req-red-$r-pin --red $r \
      > $L/pr10-gate-$req-red-$r-pin.log 2>&1; echo "gate $req red $r rc=$?"
  done
done
bash ./verify.sh pin
tail -n 9 $L/pr10-verify-pin.log
G=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts/serve_gate.sh
D=$D bash $G spec5-pin gate,bench,turns; echo "spec5-pin rc=$?"
D=$D bash $G plain-pin gate,bench,turns MEMRA_HY4F_DRAFT=0; echo "plain-pin rc=$?"
D=$D bash $G spec5-pin-b bench; echo "spec5-pin-b rc=$?"
D=$D bash $G fault-draft-pin turns MEMRA_HY4F_DRAFT_FAULT=draft:3; echo "fault-draft-pin rc=$?"
D=$D bash $G fault-backup-pin turns MEMRA_HY4F_DRAFT_FAULT=backup:3; echo "fault-backup-pin rc=$?"
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
python3 $I/draft_identity.py serve/spec5-pin/chat-gate.json serve/plain-pin/chat-gate.json serve/draft-identity-pin.json \
  > serve/draft-identity-pin.log 2>&1; echo "draft-identity rc=$?"
python3 $I/turns_identity.py serve/plain-pin/turns.json serve/spec5-pin/turns.json serve/turns-identity-pin.json \
  > serve/turns-identity-pin.log 2>&1; echo "turns-identity rc=$?"
for f in draft backup; do
  python3 $I/turns_identity.py serve/plain-pin/turns.json serve/fault-$f-pin/turns.json serve/turns-identity-fault-$f-pin.json --rounds 3 \
    > serve/turns-identity-fault-$f-pin.log 2>&1; echo "turns-identity fault-$f rc=$?"
  echo "fault-$f refusals logged: $(grep -c 'serving it without drafts' serve/fault-$f-pin/server.log)"
done
echo "PIN-CHAIN-DONE $(date '+%F %T')"
