#!/usr/bin/env bash
# q-v3h.sh (2x RTX PRO 6000 Server Edition), after q-v3g.sh: B-row decode with the two-group width
# policy (memra #667 levers 1 and 2), lane/dsv4-brow-serve-20260924 3d98c0aed. Served A/B on one
# binary, one boot per row, order A D E E D A A D E (N=3):
#   A = defaults (2 pipelined lanes), D = MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=2,
#   E = MEMRA_DSV4_SESSIONS=8 MEMRA_DSV4_ROWS=4; cells-rows8.txt (c1, c2, c4, c8).
while ! grep -q V3G_DONE /root/rcpt/q-v3g.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3h.summary; R=/root/rcpt/rows8; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-serve-20260924
git worktree add -f /root/lane/t-rows 3d98c0aed > /dev/null 2>&1; git -C /root/lane/t-rows checkout -q --detach 3d98c0aed
echo "tree rows $(git -C /root/lane/t-rows rev-parse HEAD)" >> $S
[[ -d /root/lane/target-rows ]] || cp -a /root/lane/target-pipe2 /root/lane/target-rows
bash /root/box/build.sh /root/lane/t-rows /root/lane/target-rows rows
X=/root/lane/target-rows/release
echo "build rows $(grep -hE 'EXIT' /root/build-rows.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
i=0
for arm in A D E E D A A D E; do
  i=$((i+1)); d=$R/r$i-$arm
  case $arm in A) env_arm="" ;; D) env_arm="MEMRA_DSV4_SESSIONS=4 MEMRA_DSV4_ROWS=2" ;; E) env_arm="MEMRA_DSV4_SESSIONS=8 MEMRA_DSV4_ROWS=4" ;; esac
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  # shellcheck disable=SC2086
  /root/box/cell.sh $d $X/memra-server /root/box/cells-rows8.txt $env_arm > $d.out 2>&1
  echo "rows8 served r$i $arm rc=$? $(grep -h 'serving lane\|B-row steps' $d/serve.log 2>/dev/null | cut -c1-120 | tr '\n' ' ') $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
echo V3H_DONE >> $S
