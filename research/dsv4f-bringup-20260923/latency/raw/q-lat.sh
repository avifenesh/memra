#!/usr/bin/env bash
# q-lat.sh: DSv4 latency-kernel lane component run on the PRO 6000 (task #15).
# base = 31dd11455 plus the lane's test file (captures the route_m golden from the former
# kernel), lane = lane/dsv4-latency-20260923. Correctness on both, then an interleaved
# timing A/B, N=5, both orders, 250 ms telemetry. TF32 forced off, one test thread.
while ! grep -q TPEP_DONE /root/rcpt/q-tpep.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-lat.summary; R=/root/rcpt/lat; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
T=crates/memra-engine/tests/dsv4_latency_kernels_gpu.rs
cd /root/lane/memra && git fetch -q origin lane/dsv4-latency-20260923 && LANE=$(git rev-parse FETCH_HEAD)
git worktree add -f /root/lane/memra-lat $LANE > /dev/null 2>&1
git worktree add -f /root/lane/memra-latbase 31dd1145523ddf6bb5f78605f44b9897bb429b88 > /dev/null 2>&1
git show $LANE:$T > /root/lane/memra-latbase/$T
echo "lane $LANE base 31dd1145523ddf6bb5f78605f44b9897bb429b88 test $(sha256sum /root/lane/memra-latbase/$T | cut -c1-16)" >> $S
for t in base lane; do
  if [[ $t == base ]]; then W=/root/lane/memra-latbase; else W=/root/lane/memra-lat; fi
  ( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
    cd $W && CARGO_TARGET_DIR=/root/lane/target-lat$t cargo test -p memra-engine --release --test dsv4_latency_kernels_gpu --no-run > /root/build-lat$t.log 2>&1
    echo "build $t rc=$?" >> $S )
  B=$(ls -t /root/lane/target-lat$t/release/deps/dsv4_latency_kernels_gpu-* | grep -v '\.d$' | head -1)
  cp $B $R/lat-$t.bin; sha256sum $R/lat-$t.bin >> $S
done
export NVIDIA_TF32_OVERRIDE=0
for t in base lane; do
  wait_lock; /root/box/gate.sh $R/correct-$t $R/lat-$t.bin --ignored --test-threads=1 --nocapture --skip latency_kernel_timing
  echo "correct $t $(grep -hE 'golden|test result|GATE_DONE' $R/correct-$t/gate.log | tr '\n' ' ')" >> $S
done
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
echo LAT_DONE >> $S
