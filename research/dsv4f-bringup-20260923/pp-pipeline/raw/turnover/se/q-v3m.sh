#!/usr/bin/env bash
# q-v3m.sh (2x RTX PRO 6000 Server Edition), after q-v3l.sh: request-turnover stress for the #699 fault
# (one sticky 719 in 18 default-arm rows, q-pairP r5). F = lane/dsv4-pp-pipeline-20260923 017f2bdae
# (VerifyState lands every stage's queued work before it frees), C = the same lane before the fix,
# 289ea34e2. Both on the default route (two pipelined lanes, no env). One boot per row, order
# F C C F F C C F (N=4 each), cells-turnover.txt: c4 x 48 short requests x 4 cells = 768 request
# endings per boot, 3072 per arm. A row's faults = engine-error lines in its serve.log.
while ! grep -q V3L_DONE /root/rcpt/q-v3l.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3m.summary; R=/root/rcpt/turnover; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-pp-pipeline-20260923
for a in pfix:017f2bdae pctl:289ea34e2; do
  name=${a%%:*}; c=${a##*:}
  git worktree add -f /root/lane/t-$name $c > /dev/null 2>&1; git -C /root/lane/t-$name checkout -q --detach $c
  [[ -d /root/lane/target-$name ]] || cp -a /root/lane/target-rows /root/lane/target-$name
  bash /root/box/build.sh /root/lane/t-$name /root/lane/target-$name $name
  echo "build $name $(git -C /root/lane/t-$name rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$name.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$name/release/memra-server | cut -c1-16)" >> $S
done
i=0
for arm in F C C F F C C F; do
  i=$((i+1)); d=$R/r$i-$arm; bin=/root/lane/target-pfix/release/memra-server; [[ $arm == C ]] && bin=/root/lane/target-pctl/release/memra-server
  wait_lock
  /root/box/cell.sh $d $bin /root/box/cells-turnover.txt MEMRA_TIMEOUT_MS_MAX=900000 > $d.out 2>&1
  echo "turnover r$i $arm rc=$? faults=$(grep -c engine-error $d/serve.log 2>/dev/null) $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-120 | tr '\n' ' ') $(grep -h engine-error $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
echo V3M_DONE >> $S
