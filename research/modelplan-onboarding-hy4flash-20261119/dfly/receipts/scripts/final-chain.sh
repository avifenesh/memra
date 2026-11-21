#!/usr/bin/env bash
# GB10 box, step 10: the final tree: build, the route identity and composite gate again, verify checkpoint
# + serve, then the serve gates and the chat turns with drafts on, each GPU job alone
set -u
D=$HOME/hy4flash/pr10
cd $D
true
cat src/HEAD.txt
SERVER=1 JOBS=16 ./build-gpu.sh > $HOME/hy4flash/logs/pr10-build-final.log 2>&1
grep -q BUILD_EXIT=0 $HOME/hy4flash/logs/pr10-build-final.log || { echo BUILD-FAILED; exit 1; }
sha256sum target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu target/release/memra > $D/bins-final.sha256
S=$D/target/release/hy4f-spec
R=$HOME/hy4flash/pr7/requests
mkdir -p gen-final
$S gen $HOME/hy4flash/hf-stage $R/chat.json 128 --bench 3 > gen-final/chat-k5.json 2> gen-final/chat-k5.log; echo "gen chat rc=$?"
$S gen $HOME/hy4flash/hf-stage $R/long.json 64 --prefill-chunk 2048 --bench 3 > gen-final/long-k5.json 2> gen-final/long-k5.log; echo "gen long rc=$?"
for req in chat long; do
  $S gate-route $HOME/hy4flash/hf-stage $R/$req.json $HOME/hy4flash/pr7/tapes/$req-f64 dtapes/$req-f64 gtapes/$req-route-final \
    > $HOME/hy4flash/logs/pr10-gate-route-$req-final.log 2>&1; echo "gate-route $req rc=$?"
done
bash ./verify.sh final
tail -n 9 $HOME/hy4flash/logs/pr10-verify-final.log
G=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts/serve_gate.sh
D=$D bash $G spec5-final gate,turns; echo "spec5-final rc=$?"
echo FINAL-CHAIN-DONE
