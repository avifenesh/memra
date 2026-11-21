#!/usr/bin/env bash
# GB10 box, step 10: the answer comparisons over the pin chain's serve runs (CPU, seconds): drafts on
# against drafts off over the whole gate report and the chat turns, and the two fault-door runs
# against drafts off. pin-chain.sh's own last lines ran these with an earlier argument form of
# the scripts; this is the run the receipts keep.
set -u
D=$HOME/hy4flash/pr10
I=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
cd $D
python3 $I/draft_identity.py serve/spec5-pin/chat-gate.json serve/plain-pin/chat-gate.json serve/draft-identity-pin.json \
  > serve/draft-identity-pin.log 2>&1; echo "draft-identity rc=$?"
python3 $I/turns_identity.py serve/plain-pin/turns.json serve/spec5-pin/turns.json --out serve/turns-identity-pin.json \
  > serve/turns-identity-pin.log 2>&1; echo "turns-identity rc=$?"
for f in draft backup; do
  python3 $I/turns_identity.py serve/plain-pin/turns.json serve/fault-$f-pin/turns.json --out serve/turns-identity-fault-$f-pin.json \
    --rounds 3 --log serve/fault-$f-pin/server.log > serve/turns-identity-fault-$f-pin.log 2>&1; echo "turns-identity fault-$f rc=$?"
done
echo IDENTITY-DONE
