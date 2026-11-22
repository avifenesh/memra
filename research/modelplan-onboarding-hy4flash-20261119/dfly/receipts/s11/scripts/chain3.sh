#!/usr/bin/env bash
# GB10 box, step 11 chain 3: build h3 (replay records the drafter's probabilities), price the BF16
# arm's step shapes, replay the NVFP4 arm again with the probabilities. Log: logs/s11-chain3.log.
set -u
D=$HOME/hy4flash/s11
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN3 START $(date '+%F %T') $(cat src/HEAD.txt)"
./build-gpu.sh > $L/s11-build-h3.log 2>&1
grep -q BUILD_EXIT=0 $L/s11-build-h3.log || { echo BUILD-FAILED; exit 1; }
mkdir -p bins-h3 && cp target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu bins-h3/
sha256sum bins-h3/* | tee bins-h3.sha256
S=$D/bins-h3/hy4f-spec
R=$HOME/hy4flash/pr7/requests
memok() { waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null; }
memok; $S replay $HOME/hy4flash/hf-stage replay-prompts.json replay/replay-nv-q.jsonl > replay/replay-nv-q.log 2>&1; echo "replay nv-q rc=$?"
memok; $S cost $D/arm-bf16 $R/long.json --reps 8 --temperature 0.9 > replay/cost-bf16-sampled.json 2> replay/cost-bf16-sampled.log; echo "cost bf16 sampled rc=$?"
echo "CHAIN3-DONE $(date '+%F %T')"
