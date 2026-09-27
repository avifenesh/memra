#!/usr/bin/env bash
# q-diet2.sh: small-kernel diet re-measured on the lane tree merged with main (the one-token MoE
# stream visitor, #672, is in the program). Build dccade473 plus the lane-only A/B read
# (diet-lane.patch, never merged), the kernel bit gate, served plain A/B in order
# on off off on on off, and the DSpark served identity gate on the shipped dispatch.
while ! grep -q DEFER_DONE /root/rcpt/q-defer.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-diet2.summary; R=/root/rcpt/diet2; mkdir -p $R/ab
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-small-diet-20260923 && git worktree remove --force /root/lane/memra-diet 2>/dev/null; git worktree add -f /root/lane/memra-diet2 FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-diet2 && git apply /root/box/diet-lane.patch && echo "tree $(git rev-parse HEAD) + diet-lane.patch $(sha256sum /root/box/diet-lane.patch | cut -c1-16)" >> $S
bash /root/box/build.sh /root/lane/memra-diet2 /root/lane/target-diet diet2
grep -E 'EXIT|BUILD' /root/build-diet2.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-diet/release
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-diet2 && CARGO_TARGET_DIR=/root/lane/target-diet cargo test -p memra-engine --release --test dsv4_small_diet_gpu --no-run > /root/build-diet2-test.log 2>&1 )
T=$(ls -t /root/lane/target-diet/release/deps/dsv4_small_diet_gpu-* | grep -v '\.d$' | head -1)
echo "test-bin $T" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T --ignored --nocapture --test-threads=1
echo "component $(grep -E 'EXACT|test result|GATE_DONE' $R/component/gate.log | tr '\n' ' ')" >> $S
i=0
for arm in on off off on on off; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DIET_AB=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/ab/r$i-$arm $X/memra-server /root/box/ab-cells.txt "$@" > $R/ab/r$i-$arm.out 2>&1
  echo "r$i $arm rc=$? $(grep 'CELL greedy-c1' $R/ab/r$i-$arm/controller.log)" >> $S
done
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -h GATE $R/dspark-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
echo DIET2_DONE >> $S
