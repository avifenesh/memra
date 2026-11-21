#!/usr/bin/env bash
# GB10 box: the GPU drafter gate's red arms (drafter-only route, about 3 GB)
set -u
D=$HOME/hy4flash/pr10
cd $D
for req in chat long; do
  for r in layers width no-correction topk; do
    timeout 900 ./target/release/hy4f-spec gate $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/$req.json \
      target/$req dtapes/$req-f64 gtapes/$req-red-$r --red $r > $HOME/hy4flash/logs/pr10-gate-$req-red-$r.log 2>&1
    echo "$req $r rc=$?"
  done
done
echo GATE-REDS-DONE
