#!/usr/bin/env bash
# GB10 box, step 10: the served gates and readings with DFly2 drafts on, then off, then on again.
set -u
D=$HOME/hy4flash/pr10
cd $D
cp -n $HOME/hy4flash/pr9/tapes.json $HOME/hy4flash/pr9/route-gen.json $D/
waitfor --grep "DFLY-CHAIN2-DONE" $HOME/hy4flash/logs/pr10-dfly-chain2.log --timeout 3h --every 60s --quiet
waitfor --grep "GATE-REDS-DONE" $HOME/hy4flash/logs/pr10-gate-reds.log --timeout 60m --every 30s --quiet
waitfor --grep "BUILD_EXIT=0" $HOME/hy4flash/logs/pr10-build-srv.log --timeout 60m --every 30s --quiet || exit 1
# the build that started before the last source sync: build again (incremental) on the synced tree
SERVER=1 JOBS=8 ./build-gpu.sh > $HOME/hy4flash/logs/pr10-build-srv2.log 2>&1
grep -q BUILD_EXIT=0 $HOME/hy4flash/logs/pr10-build-srv2.log || exit 1
sha256sum target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu target/release/memra > $D/bins.sha256
G=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts/serve_gate.sh
D=$D bash $G spec5 gate,bench,turns; echo "spec5 rc=$?"
D=$D bash $G plain bench,turns MEMRA_HY4F_DRAFT=0; echo "plain rc=$?"
D=$D bash $G spec5-b bench; echo "spec5-b rc=$?"
echo SERVE-CHAIN-DONE
