#!/usr/bin/env bash
# q-pairP.sh (2x RTX PRO 6000 WS pod), run by q-pair2.sh before its own work: pipelined B-row
# groups (memra #667 levers 1 and 2), lane/dsv4-brow-serve-20260924 e2c106ebe.
# 1. dsv4_rows_gate: the join/leave identity arm, the pipelined two-groups identity arm, and
#    timing (one-row, B=1/2/4 serial, two pipelined groups of 2).
# 2. If it passes, served A/B on one binary, one boot per row, order A D D A A D:
#    A = defaults (2 pipelined lanes), D = MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=2 (two groups
#    of 2 in flight); cells-pipe.txt.
set -u
S=/root/rcpt/q-pairP.summary; R=/root/rcpt/browp; mkdir -p $R/served
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-serve-20260924
git -C /root/lane/t-brow checkout -q --detach e2c106ebe
echo "tree browp $(git -C /root/lane/t-brow rev-parse HEAD)" >> $S
bash /root/box/build.sh /root/lane/t-brow /root/lane/target-brow browp
X=/root/lane/target-brow/release
echo "build browp $(grep -hE 'EXIT' /root/build-browp.log | tr '\n' ' ') gate $(sha256sum $X/dsv4_rows_gate | cut -c1-16) server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
/root/box/gate.sh $R/gate $X/dsv4_rows_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64
echo "rows gate $(grep -hE '^(SESSION|PIPELINED|PASS|FAILED|TIME)|panicked|GATE_DONE' $R/gate/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
if [[ $(grep -c '^PASS' $R/gate/gate.log) -ge 2 ]]; then
  i=0
  for arm in A D D A A D; do
    i=$((i+1)); d=$R/served/r$i-$arm
    env_arm=""; [[ $arm == D ]] && env_arm="MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=2"
    while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
    # shellcheck disable=SC2086
    /root/box/cell.sh $d $X/memra-server /root/box/cells-pipe.txt $env_arm > $d.out 2>&1
    echo "browp served r$i $arm rc=$? $(grep -h 'serving lane\|B-row steps' $d/serve.log 2>/dev/null | cut -c1-120 | tr '\n' ' ') $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
  done
fi
echo BROWP_DONE >> $S
