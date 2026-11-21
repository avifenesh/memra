#!/usr/bin/env bash
# GB10 box, step 10: the f64 target oracle again with the DFly2 drafter's inputs dumped (chat, long).
set -u
D=$HOME/hy4flash/pr10
P7=$HOME/hy4flash/pr7
mkdir -p $D/tapes $D/target $HOME/hy4flash/logs
for req in chat long; do
  log=$HOME/hy4flash/logs/pr10-oracle-$req-f64.log
  echo "START $(date '+%F %T') req=$req" > $log
  systemd-run --user --scope -q -p MemoryMax=100G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $P7/venv/bin/python -I $D/lane/checkpoint/make_checkpoint_oracle.py $HOME/hy4flash/hf-stage \
    $P7/requests/$req.json $D/tapes/$req-f64 --dtype f64 --dump-dfly $D/target/$req >> $log 2>&1
  echo "EXIT rc=$? $(date '+%F %T')" >> $log
done
echo CHAIN-DONE >> $HOME/hy4flash/logs/pr10-oracle-long-f64.log
