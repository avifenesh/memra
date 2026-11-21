#!/usr/bin/env bash
# GB10 box, step 10: after the CPU oracles: the composite drafter gate on the route, then the serve chain.
set -u
D=$HOME/hy4flash/pr10
cd $D
waitfor --grep "DFLY-CHAIN3B-DONE" $HOME/hy4flash/logs/pr10-dfly-chain3b.log --timeout 2h --every 30s --quiet || exit 1
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh || exit 1
S=$D/target/release/hy4f-spec
for req in chat long; do
  $S gate-route $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/$req.json $HOME/hy4flash/pr7/tapes/$req-f64 \
    dtapes/$req-f64 gtapes/$req-route > $HOME/hy4flash/logs/pr10-gate-route-$req.log 2>&1
  echo "gate-route $req rc=$?"
done
./serve-chain.sh
echo GPU-CHAIN-DONE
