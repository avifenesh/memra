#!/usr/bin/env bash
# q-mrow.sh: multi-row stream visitor lane. Build the lane tree plus the lane-only A/B read
# (mrow-lane.patch, never merged), GPU component tests, then served DSpark A/B (A = mrow 1 shipped
# dispatch, B = mrow 0 verify on sktail, order A B B A A B B A A B), a plain probe (1 shipped,
# 2 = multi-row visitor also on the one-token step, 0 = batched rows on sktail; order 1 2 0 0 2 1),
# and the DSpark served identity gate on the shipped dispatch.
while ! grep -q FIN_DONE /root/rcpt/q-fin.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-mrow.summary; R=/root/rcpt/mrow; mkdir -p $R/ab $R/plain
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-mrow-20260923 && git worktree add -f /root/lane/memra-mrow FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-mrow && git apply /root/box/mrow-lane.patch && echo "tree $(git rev-parse HEAD) + mrow-lane.patch $(sha256sum /root/box/mrow-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-mrow ]] || cp -a /root/lane/target-fin /root/lane/target-mrow
bash /root/box/build.sh /root/lane/memra-mrow /root/lane/target-mrow mrow
grep -E 'EXIT|BUILD' /root/build-mrow.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-mrow/release
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-mrow && CARGO_TARGET_DIR=/root/lane/target-mrow cargo test --release -j 48 -p memra-engine --lib --no-run ) > $R/test-build.log 2>&1
T=$(grep -o '/root/lane/target-mrow/release/deps/memra_engine-[0-9a-f]*' $R/test-build.log | tail -1)
echo "test-bin $T" >> $S
wait_lock; /root/box/gate.sh $R/component $T stream_matches_sktail_bit_for_bit --ignored --nocapture --test-threads=1
echo "component $(grep -E 'test result|GATE_DONE' $R/component/gate.log | tr '\n' ' ')" >> $S
i=0
for arm in 1 0 0 1 1 0 0 1 1 0; do
  i=$((i+1)); wait_lock
  /root/box/cell.sh $R/ab/r$i-m$arm $X/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_MOE_MROW_STREAM=$arm MEMRA_ENV_AUDIT=warn > $R/ab/r$i-m$arm.out 2>&1
  echo "ab r$i m$arm rc=$? $(grep 'CELL ' $R/ab/r$i-m$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
done
i=0
for arm in 1 2 0 0 2 1; do
  i=$((i+1)); wait_lock
  /root/box/cell.sh $R/plain/p$i-m$arm $X/memra-server /root/box/cells-base.txt MEMRA_DSV4_MOE_MROW_STREAM=$arm MEMRA_ENV_AUDIT=warn > $R/plain/p$i-m$arm.out 2>&1
  echo "plain p$i m$arm rc=$? $(grep 'CELL ' $R/plain/p$i-m$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
done
wait_lock; /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served
echo "dspark-served $(tail -3 $R/dspark-served/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
echo MROW_DONE >> $S
