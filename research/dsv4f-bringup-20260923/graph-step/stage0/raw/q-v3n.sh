#!/usr/bin/env bash
# q-v3n.sh (2x RTX PRO 6000 Server Edition), after q-v3m.sh: memra #710 stage 0 (commit uploads the
# ring slots once per stage, scatters in place), lane/dsv4-graph-step-20260924 d888630a3 against its base
# 017f2bdae (#699 with the drop guard). DSpark served identity gate on the lane binary, then served
# plain and DSpark cells, one boot per row, order L B B L L B (N=3), cells-spec.txt.
while ! grep -q V3M_DONE /root/rcpt/q-v3m.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3n.summary; R=/root/rcpt/stage0; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-graph-step-20260924
git worktree add -f /root/lane/t-gs0 d888630a3 > /dev/null 2>&1; git -C /root/lane/t-gs0 checkout -q --detach d888630a3
[[ -d /root/lane/target-gs0 ]] || cp -a /root/lane/target-pfix /root/lane/target-gs0
bash /root/box/build.sh /root/lane/t-gs0 /root/lane/target-gs0 gs0
L=/root/lane/target-gs0/release; B=/root/lane/target-pfix/release
echo "build gs0 $(git -C /root/lane/t-gs0 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-gs0.log | tr '\n' ' ') server $(sha256sum $L/memra-server | cut -c1-16) base $(sha256sum $B/memra-server | cut -c1-16)" >> $S
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $L/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "gs0 dspark-served $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $R/dspark-served/gate.log | head -n 6 | tr '\n' ' ')" >> $S
i=0
for arm in L B B L L B; do
  for mode in plain spec; do
    i=$((i+1)); d=$R/$mode-r$i-$arm; bin=$B/memra-server; [[ $arm == L ]] && bin=$L/memra-server
    extra=""; [[ $mode == spec ]] && extra="MEMRA_DSV4_DRAFTER=dspark"
    wait_lock
    /root/box/cell.sh $d $bin /root/box/cells-spec.txt MEMRA_DSV4_SESSIONS=1 $extra > $d.out 2>&1
    echo "stage0 $mode r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
  done
done
echo V3N_DONE >> $S
