#!/usr/bin/env bash
# q-v3r.sh (2x RTX PRO 6000 Server Edition): memra #718 DSpark under TP/EP. Two trees, one gate:
# ctl = main 2ad82c2e9 + the gate's per-round proposal digest (d7e79f5a3, whole drafter on the
# last stage), lane = the expert-id split drafter (acd3986b6). dsv4-gpu-dspark-gate on PP-2 and
# on TP/EP (exact attention TP), each arm on both trees. The drafter is bit-identical when the
# proposal sha of every DS/DB arm matches between ctl and lane on the same topology.
set -u
S=/root/rcpt/q-v3r.summary; R=/root/rcpt/dsparkep; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
cd /root/lane/memra && git fetch -q origin lane/dsv4-dspark-ep-20260924 lane/dsv4-dspark-ep-control-20260924
for a in dctl:d7e79f5a3 dlane:acd3986b6; do
  name=${a%%:*}; c=${a##*:}
  git worktree add -f /root/lane/t-$name $c > /dev/null 2>&1; git -C /root/lane/t-$name checkout -q --detach $c
  [[ -d /root/lane/target-$name ]] || cp -a /root/lane/target-tpa /root/lane/target-$name
  bash /root/box/build.sh /root/lane/t-$name /root/lane/target-$name $name
  echo "build $name $(git -C /root/lane/t-$name rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$name.log | tr '\n' ' ') gate $(sha256sum /root/lane/target-$name/release/dsv4-gpu-dspark-gate | cut -c1-16)" >> $S
done
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
for topo in pp tpep; do
  for name in dctl dlane; do
    X=/root/lane/target-$name/release/dsv4-gpu-dspark-gate; d=$R/$topo-$name; wait_lock
    if [[ $topo == pp ]]; then
      ( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
        /root/box/gate.sh $d $X $M $FX $d/out 2 0,1 --served )
    else
      ( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1
        /root/box/gate.sh $d $X $M $FX $d/out 2 0,1 --served --tpep )
    fi
    echo "$topo $name $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|panicked|REFUSE|out of memory|GATE_DONE' $d/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
    grep -hE '^\[vram' $d/gate.log | tail -2 | sed "s/^/$topo $name /" >> $S
  done
done
echo V3R_DONE >> $S
