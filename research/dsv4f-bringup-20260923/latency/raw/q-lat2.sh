#!/usr/bin/env bash
# q-lat2.sh (dev pair): DSv4 latency lane (task #15) against main 7029cd67c.
# lane = lane/dsv4-latency-20260923 2a41f68bd: register-resident rmsnorm, warp route_m, warp
# expert prefix, grouped wo_a on a grouped dense-fast twin (default ON).
# base = main 7029cd67c plus the lane's latency test file (it only calls FFI present on main).
# 1 builds, 2 correctness on both trees plus the lane's wo_a identity matrix (lib tests),
# 3 interleaved component timing N=5 both orders with 250 ms telemetry, 4 the DSpark served
# identity gate on the lane, 5 served plain then DSpark A/B, order b l l b b l l b (N=4 each).
while ! grep -q DEV2_DONE /root/rcpt/q-dev2.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-lat2.summary; R=/root/rcpt/lat2; mkdir -p $R/plain $R/spec
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
T=crates/memra-engine/tests/dsv4_latency_kernels_gpu.rs
LANE=2a41f68bd6eeeaebb2a525b84ce68389a966970c
BASE=7029cd67cd537d826212d1ab9fb9b4a0acd8851f
cd /root/lane/memra && git fetch -q origin lane/dsv4-latency-20260923 main
git worktree add -f /root/lane/memra-lat $LANE > /dev/null 2>&1
git worktree add -f /root/lane/memra-latbase $BASE > /dev/null 2>&1
git show $LANE:$T > /root/lane/memra-latbase/$T
echo "lane $(git -C /root/lane/memra-lat rev-parse HEAD) base $(git -C /root/lane/memra-latbase rev-parse HEAD) test $(sha256sum /root/lane/memra-latbase/$T | cut -c1-16)" >> $S
for t in base lane; do
  [[ -d /root/lane/target-lat$t ]] || cp -a /root/lane/target-dmrow /root/lane/target-lat$t
done
bash /root/box/build.sh /root/lane/memra-latbase /root/lane/target-latbase latbase
bash /root/box/build.sh /root/lane/memra-lat /root/lane/target-latlane latlane
for t in base lane; do
  grep -E 'EXIT|BUILD' /root/build-lat$t.log | tr '\n' ' ' >> $S; echo >> $S
done
for t in base lane; do
  if [[ $t == base ]]; then W=/root/lane/memra-latbase; else W=/root/lane/memra-lat; fi
  ( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
    cd $W && CARGO_TARGET_DIR=/root/lane/target-lat$t cargo test -p memra-engine --release --test dsv4_latency_kernels_gpu --no-run > /root/build-lat$t-test.log 2>&1
    echo "build-test $t rc=$?" >> $S )
  B=$(ls -t /root/lane/target-lat$t/release/deps/dsv4_latency_kernels_gpu-* | grep -v '\.d$' | head -1)
  cp $B $R/lat-$t.bin; sha256sum $R/lat-$t.bin >> $S
done
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-lat && CARGO_TARGET_DIR=/root/lane/target-latlane cargo test -p memra-engine --release --lib --no-run > /root/build-latlane-lib.log 2>&1
  echo "build-lib lane rc=$?" >> $S )
L=$(ls -t /root/lane/target-latlane/release/deps/memra_engine-* | grep -v '\.d$\|\.rlib$\|\.rmeta$' | head -1)
cp $L $R/lib-lane.bin; sha256sum $R/lib-lane.bin >> $S
export NVIDIA_TF32_OVERRIDE=0
for t in base lane; do
  wait_lock; /root/box/gate.sh $R/correct-$t $R/lat-$t.bin --ignored --test-threads=1 --nocapture --skip latency_kernel_timing
  echo "correct $t $(grep -hE 'golden|test result|GATE_DONE|panicked' $R/correct-$t/gate.log | tr '\n' ' ')" >> $S
done
wait_lock; /root/box/gate.sh $R/woa-lane $R/lib-lane.bin --ignored --test-threads=1 --nocapture dense_wo_a_grouped_fp8_component_tests
echo "woa lane $(grep -hE 'PASS|test result|GATE_DONE|panicked' $R/woa-lane/gate.log | tr '\n' ' ')" >> $S
wait_lock
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo SLOT_BUSY >> $S; exit 75; }
nvidia-smi --query-gpu=index,timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv -lms 250 > $R/telemetry.csv 9>&- &
TEL=$!
for r in 1 2 3 4 5; do
  if [ $((r % 2)) -eq 1 ]; then order="base lane"; else order="lane base"; fi
  for t in $order; do
    DSV4_LATENCY_TREE=$t CUDA_VISIBLE_DEVICES=0 $R/lat-$t.bin --ignored --test-threads=1 --nocapture --exact latency_kernel_timing > $R/timing-r$r-$t.log 2>&1 9>&-
    echo "timing r$r $t rc=$?" >> $S
  done
done
kill $TEL; exec 9>&-
X=/root/lane/target-latlane/release
XB=/root/lane/target-latbase/release
for t in base lane; do d=$X; [[ $t == base ]] && d=$XB; echo "bin $t $(sha256sum $d/memra-server | cut -c1-16)" >> $S; done
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/dspark-served $X/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/dspark-served/out 2 0,1 --served )
echo "dspark-served $(grep -h GATE $R/dspark-served/gate.log | tail -2 | tr '\n' ' ')" >> $S
i=0
for arm in base lane lane base base lane lane base; do
  i=$((i+1)); wait_lock; d=$X; [[ $arm == base ]] && d=$XB
  /root/box/cell.sh $R/plain/r$i-$arm $d/memra-server /root/box/cells-spec.txt MEMRA_ENV_AUDIT=warn > $R/plain/r$i-$arm.out 2>&1
  echo "plain r$i $arm rc=$? $(grep -hE 'CELL ' $R/plain/r$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
i=0
for arm in base lane lane base base lane lane base; do
  i=$((i+1)); wait_lock; d=$X; [[ $arm == base ]] && d=$XB
  /root/box/cell.sh $R/spec/s$i-$arm $d/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark MEMRA_ENV_AUDIT=warn > $R/spec/s$i-$arm.out 2>&1
  echo "spec s$i $arm rc=$? $(grep -hE 'CELL ' $R/spec/s$i-$arm/controller.log | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo LAT2_DONE >> $S
