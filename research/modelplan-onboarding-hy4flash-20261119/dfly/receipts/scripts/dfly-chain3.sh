#!/usr/bin/env bash
# GB10 box, step 10: the composite band backend: the target's bf16x2a backend (forced to the f64 picks)
# with its drafter inputs dumped, then the drafter's bf16x2a backend over them, forced along the f64 drafts.
set -u
D=$HOME/hy4flash/pr10
P7=$HOME/hy4flash/pr7
cd $D
LONG_ANCH="stride:16,range:2040-2110,at:163,at:426,at:756,at:1083"
for req in chat long; do
  log=$HOME/hy4flash/logs/pr10-oracle-$req-bf16x2af.log
  echo "START $(date '+%F %T')" > $log
  systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $P7/venv/bin/python -I $D/lane/checkpoint/make_checkpoint_oracle.py $HOME/hy4flash/hf-stage \
    $P7/requests/$req.json $D/tapes/$req-bf16x2af --dtype bf16x2a --force-selections $P7/tapes/$req-f64 \
    --dump-dfly $D/target/$req-bf16x2af >> $log 2>&1
  echo "EXIT rc=$? $(date '+%F %T')" >> $log
  anch=all; [ $req = long ] && anch=$LONG_ANCH
  log=$HOME/hy4flash/logs/pr10-dfly-$req-x2-x2.log
  systemd-run --user --scope -q -p MemoryMax=28G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $P7/venv/bin/python -I $D/lane/dfly/make_dfly_oracle.py $HOME/hy4flash/hf-stage $P7/requests/$req.json \
    $D/target/$req-bf16x2af $D/dtapes/$req-x2-x2 --vllm-src $D/vllm-src --dtype bf16x2a --anchors "$anch" \
    --force-drafts $D/dtapes/$req-f64 > $log 2>&1
  echo "dfly $req rc=$?"
done
echo DFLY-CHAIN3-DONE
