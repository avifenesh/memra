#!/usr/bin/env bash
# q-pairR.sh (2x RTX PRO 6000 WS pod), between q-pairL.sh and q-pair2.sh: B-row decode across
# requests (memra #667 lever 2). One tree, lane/dsv4-brow-serve-20260924 46c259a2a (the #699
# pipelining lane + the B-row engine lane + lane coalescing), builds the rows gate and the server.
# 1. dsv4_rows_gate: four sessions alone, then as B-row steps with joins, leaves and rotating
#    rows; every row's logits bits must equal its solo step. One-row vs B-row timing, B=1,2,4.
# 2. If it passes, served A/B on one binary (46c259a2a: sampled rows batch too), order A B C C A B B C A:
#    A = defaults (2 pipelined lanes), B = MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=4,
#    C = MEMRA_DSV4_SESSIONS=2 MEMRA_DSV4_ROWS=2; cells-pipe.txt (greedy c1/c2/c4, sampled c2/c4).
# Runs q-pair2.sh when done so the chain keeps one GPU tenant at a time.
while ! grep -q PAIR_DONE /root/rcpt/q-pair.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pairR.summary; R=/root/rcpt/brow; mkdir -p $R/served
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-serve-20260924
git worktree add -f /root/lane/t-brow 46c259a2a > /dev/null 2>&1; git -C /root/lane/t-brow checkout -q --detach 46c259a2a
echo "tree brow $(git -C /root/lane/t-brow rev-parse HEAD)" >> $S
[[ -d /root/lane/target-brow ]] || cp -a /root/lane/target-main /root/lane/target-brow
bash /root/box/build.sh /root/lane/t-brow /root/lane/target-brow brow
X=/root/lane/target-brow/release
echo "build brow $(grep -hE 'EXIT' /root/build-brow.log | tr '\n' ' ') gate $(sha256sum $X/dsv4_rows_gate | cut -c1-16) server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
/root/box/gate.sh $R/gate $X/dsv4_rows_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64
echo "rows gate $(grep -hE '^(WIDTHS|SESSION|PASS|FAILED|TIME)|panicked|GATE_DONE' $R/gate/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
if grep -q '^PASS' $R/gate/gate.log; then
  arm_env() { case $1 in A) ;; B) echo MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=4 ;; C) echo MEMRA_DSV4_SESSIONS=2 MEMRA_DSV4_ROWS=2 ;; esac; }
  i=0
  for arm in A B C C A B B C A; do
    i=$((i+1)); d=$R/served/r$i-$arm
    while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
    # shellcheck disable=SC2046
    /root/box/cell.sh $d $X/memra-server /root/box/cells-pipe.txt $(arm_env $arm) > $d.out 2>&1
    echo "brow served r$i $arm rc=$? $(grep -h 'serving lane\|B-row steps' $d/serve.log 2>/dev/null | cut -c1-120 | tr '\n' ' ') $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
  done
fi
echo ROWS_DONE >> $S
exec bash /root/box/q-pair2.sh
