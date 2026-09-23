#!/usr/bin/env bash
# q-defer.sh: deferred MoE route/mirror checks (memra #670, lane/dsv4-moe-defer-20260923, stacked
# on the multi-row lane). Build the lane tree plus the lane-only A/B read (defer-lane.patch, never
# merged: MEMRA_DSV4_MOE_DEFER=0 allocates no fault word, so every check reads back and
# synchronizes as before), the grouped GPU component tests, served plain A/B (A = deferred, the
# shipped code; B = synchronous; order A B B A A B B A A B), served DSpark A/B in the same order,
# the DSpark served identity gate on the shipped code, and one nsys capture per arm.
while ! grep -q MROWGATE_DONE /root/rcpt/q-mrowgate.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-defer.summary; R=/root/rcpt/defer; mkdir -p $R/ab $R/spec $R/prof
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-defer-20260923 && git worktree add -f /root/lane/memra-defer FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-defer && git apply /root/box/defer-lane.patch && echo "tree $(git rev-parse HEAD) + defer-lane.patch $(sha256sum /root/box/defer-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-defer ]] || cp -a /root/lane/target-mrow /root/lane/target-defer
bash /root/box/build.sh /root/lane/memra-defer /root/lane/target-defer defer
grep -E 'EXIT|BUILD' /root/build-defer.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-defer/release
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-defer && CARGO_TARGET_DIR=/root/lane/target-defer cargo test --release -j 48 -p memra-engine --lib --no-run ) > $R/test-build.log 2>&1
T=$(grep -o '/root/lane/target-defer/release/deps/memra_engine-[0-9a-f]*' $R/test-build.log | tail -1)
echo "test-bin $T" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T dsv4_grouped:: --ignored --nocapture --test-threads=1
echo "component $(grep -E 'test result|GATE_DONE' $R/component/gate.log | tr '\n' ' ')" >> $S
i=0
for arm in on off off on on off off on on off; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_MOE_DEFER=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/ab/r$i-$arm $X/memra-server /root/box/defer-cells.txt "$@" > $R/ab/r$i-$arm.out 2>&1
  echo "ab r$i $arm rc=$? $(grep 'CELL ' $R/ab/r$i-$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
done
i=0
for arm in on off off on on off off on on off; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_MOE_DEFER=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/spec/s$i-$arm $X/memra-server /root/box/cells-spec.txt "$@" > $R/spec/s$i-$arm.out 2>&1
  echo "spec s$i $arm rc=$? $(grep 'CELL ' $R/spec/s$i-$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
done
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  unset MEMRA_DSV4_MOE_DEFER
  /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -h GATE $R/dspark-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
wait_lock; /root/box/prof-served.sh $R/prof/served-plain-on $X/memra-server MEMRA_ENV_AUDIT=warn > $R/prof/on.out 2>&1
echo "prof on $(tail -1 $R/prof/on.out)" >> $S
wait_lock; /root/box/prof-served.sh $R/prof/served-plain-off $X/memra-server MEMRA_DSV4_MOE_DEFER=0 MEMRA_ENV_AUDIT=warn > $R/prof/off.out 2>&1
echo "prof off $(tail -1 $R/prof/off.out)" >> $S
echo DEFER_DONE >> $S
