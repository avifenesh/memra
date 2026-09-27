#!/usr/bin/env bash
# q-diet.sh: small-kernel diet on the served PP-2 program (lane/dsv4-small-diet-20260923).
# Build the lane tree plus the lane-only A/B read (diet-lane.patch, never merged), the kernel bit
# gate, served plain A/B (A = diet, the shipped code; B = MEMRA_DSV4_DIET_AB=0, the unfused
# kernels; order A B B A A B B A A B), the DSpark served identity gate on the shipped dispatch,
# and served DSpark A B B A.
while ! grep -q GATES_DONE /root/rcpt/q-gates.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-diet.summary; R=/root/rcpt/diet; mkdir -p $R/ab $R/spec
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-small-diet-20260923 && git worktree add -f /root/lane/memra-diet FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-diet && git apply /root/box/diet-lane.patch && echo "tree $(git rev-parse HEAD) + diet-lane.patch $(sha256sum /root/box/diet-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-diet ]] || cp -a /root/lane/target-main /root/lane/target-diet
bash /root/box/build.sh /root/lane/memra-diet /root/lane/target-diet diet
grep -E 'EXIT|BUILD' /root/build-diet.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-diet/release
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-diet && CARGO_TARGET_DIR=/root/lane/target-diet cargo test -p memra-engine --release --test dsv4_small_diet_gpu --no-run > /root/build-diet-test.log 2>&1 )
T=$(ls -t /root/lane/target-diet/release/deps/dsv4_small_diet_gpu-* | grep -v '\.d$' | head -1)
echo "test-bin $T" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T --ignored --nocapture --test-threads=1
echo "component $(grep -E 'EXACT|test result|GATE_DONE' $R/component/gate.log | tr '\n' ' ')" >> $S
i=0
for arm in on off off on on off off on on off; do
  i=$((i+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DIET_AB=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/ab/r$i-$arm $X/memra-server /root/box/ab-cells.txt "$@" > $R/ab/r$i-$arm.out 2>&1
  echo "r$i $arm rc=$? $(grep 'CELL greedy-c1' $R/ab/r$i-$arm/controller.log)" >> $S
done
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -h GATE $R/dspark-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
j=0
for arm in on off off on; do
  j=$((j+1)); wait_lock
  if [[ $arm == on ]]; then set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_ENV_AUDIT=warn; else set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DIET_AB=0 MEMRA_ENV_AUDIT=warn; fi
  /root/box/cell.sh $R/spec/s$j-$arm $X/memra-server /root/box/cells-spec.txt "$@" > $R/spec/s$j-$arm.out 2>&1
  echo "s$j $arm rc=$? $(grep -E 'CELL (greedy|sampled)-c1 ' $R/spec/s$j-$arm/controller.log | tr '\n' ' ')" >> $S
done
echo DIET_DONE >> $S
