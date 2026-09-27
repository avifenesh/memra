#!/usr/bin/env bash
# q-dev2.sh (dev pair): multi-row small-kernel diet (lane/dsv4-diet-mrow-20260923). Build the lane
# tree plus the lane-only A/B read (dmrow-lane.patch, never merged: MEMRA_DSV4_DIET_AB=t1 keeps the
# diet on t=1 rows only, unset is the shipped code), the kernel bit gate (single- and multi-row),
# the DSpark served identity gate on the shipped dispatch, served DSpark A/B (order on t1 t1 on on
# t1 t1 on) and served plain A/B (on t1 t1 on) for TTFT.
while ! grep -q DEV1_DONE /root/rcpt/q-dev1.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-dev2.summary; R=/root/rcpt/dmrow; mkdir -p $R/spec $R/plain
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-diet-mrow-20260923 && git worktree add -f /root/lane/memra-dmrow FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-dmrow && git apply /root/box/dmrow-lane.patch && echo "tree $(git rev-parse HEAD) + dmrow-lane.patch $(sha256sum /root/box/dmrow-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-dmrow ]] || cp -a /root/lane/target-diet /root/lane/target-dmrow
bash /root/box/build.sh /root/lane/memra-dmrow /root/lane/target-dmrow dmrow
grep -E 'EXIT|BUILD' /root/build-dmrow.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-dmrow/release
echo "bin $(sha256sum $X/memra-server | cut -c1-16)" >> $S
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-dmrow && CARGO_TARGET_DIR=/root/lane/target-dmrow cargo test -p memra-engine --release --test dsv4_small_diet_gpu --no-run > /root/build-dmrow-test.log 2>&1 )
T=$(ls -t /root/lane/target-dmrow/release/deps/dsv4_small_diet_gpu-* | grep -v '\.d$' | head -1)
echo "test-bin $T $(sha256sum $T | cut -c1-16)" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T --ignored --nocapture --test-threads=1
echo "component $(grep -E 'EXACT|test result|GATE_DONE|panicked' $R/component/gate.log | tr '\n' ' ')" >> $S
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -h GATE $R/dspark-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
j=0
for arm in on t1 t1 on on t1 t1 on; do
  j=$((j+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DIET_AB=t1 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/spec/s$j-$arm $X/memra-server /root/box/cells-spec.txt "$@" > $R/spec/s$j-$arm.out 2>&1
  echo "s$j $arm rc=$? $(grep -hE 'CELL ' $R/spec/s$j-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
i=0
for arm in on t1 t1 on; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DIET_AB=t1 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/plain/r$i-$arm $X/memra-server /root/box/cells-spec.txt "$@" > $R/plain/r$i-$arm.out 2>&1
  echo "r$i $arm rc=$? $(grep -hE 'CELL ' $R/plain/r$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo DEV2_DONE >> $S
