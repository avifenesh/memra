#!/usr/bin/env bash
# q-dmrow3.sh (dev pair): the S1 multi-row diet A/B on current main. lane = lane/dsv4-diet-mrow-20260923
# a90851bff (the diet change merged with main c3eb41d12, #678), base = main c3eb41d12. Two binaries,
# no lane-only read. Kernel bit gate (single- and multi-row) and the DSpark served identity gate on
# the lane binary, then served DSpark and plain A/B, order l b b l l b b l l b, one boot per row.
while ! grep -q TPEP2_DONE /root/rcpt/q-tpep2.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-dmrow3.summary; R=/root/rcpt/dmrow3; mkdir -p $R/spec $R/plain
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin main lane/dsv4-diet-mrow-20260923
git worktree add -f /root/lane/memra-dmrow3 a90851bff27f3fcc667ff9015838ee20769f1a16 > /dev/null 2>&1
git worktree add -f /root/lane/memra-main3 c3eb41d12 > /dev/null 2>&1
echo "lane-tree $(git -C /root/lane/memra-dmrow3 rev-parse HEAD) base-tree $(git -C /root/lane/memra-main3 rev-parse HEAD)" >> $S
[[ -d /root/lane/target-dmrow3 ]] || cp -a /root/lane/target-tpep2 /root/lane/target-dmrow3
[[ -d /root/lane/target-main3 ]] || cp -a /root/lane/target-latlane /root/lane/target-main3
bash /root/box/build.sh /root/lane/memra-dmrow3 /root/lane/target-dmrow3 dmrow3
bash /root/box/build.sh /root/lane/memra-main3 /root/lane/target-main3 main3
grep -hE 'EXIT|BUILD' /root/build-dmrow3.log /root/build-main3.log | tr '\n' ' ' >> $S; echo >> $S
L=/root/lane/target-dmrow3/release; B=/root/lane/target-main3/release
echo "lane $(sha256sum $L/memra-server | cut -c1-16) base $(sha256sum $B/memra-server | cut -c1-16) lane-dspark-gate $(sha256sum $L/dsv4-gpu-dspark-gate | cut -c1-16)" >> $S
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-dmrow3 && CARGO_TARGET_DIR=/root/lane/target-dmrow3 cargo test --release -j 48 -p memra-engine --test dsv4_small_diet_gpu --test dsv4_latency_kernels_gpu --no-run ) > $R/test-build.log 2>&1
T1=$(grep -o '/root/lane/target-dmrow3/release/deps/dsv4_small_diet_gpu-[0-9a-f]*' $R/test-build.log | tail -1)
T2=$(grep -o '/root/lane/target-dmrow3/release/deps/dsv4_latency_kernels_gpu-[0-9a-f]*' $R/test-build.log | tail -1)
echo "test-bin $T1 $(sha256sum $T1 | cut -c1-16) $T2 $(sha256sum $T2 | cut -c1-16)" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T1 --ignored --nocapture --test-threads=1
echo "component $(grep -E 'EXACT|test result|GATE_DONE' $R/component/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/latency $T2 --ignored --nocapture --test-threads=1
echo "latency $(grep -E 'test result|GATE_DONE' $R/latency/gate.log | tr '\n' ' ')" >> $S
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $L/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $R/dspark-served/gate.log | head -n 6 | tr '\n' ' ')" >> $S
i=0
for arm in lane base base lane lane base base lane lane base; do
  i=$((i+1)); wait_lock; d=$B/memra-server; [[ $arm == lane ]] && d=$L/memra-server
  /root/box/cell.sh $R/spec/s$i-$arm $d /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark > $R/spec/s$i-$arm.out 2>&1
  echo "spec s$i $arm rc=$? $(grep -hE 'CELL ' $R/spec/s$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
i=0
for arm in lane base base lane lane base base lane lane base; do
  i=$((i+1)); wait_lock; d=$B/memra-server; [[ $arm == lane ]] && d=$L/memra-server
  /root/box/cell.sh $R/plain/r$i-$arm $d /root/box/cells-spec.txt > $R/plain/r$i-$arm.out 2>&1
  echo "plain r$i $arm rc=$? $(grep -hE 'CELL ' $R/plain/r$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo DMROW3_DONE >> $S
