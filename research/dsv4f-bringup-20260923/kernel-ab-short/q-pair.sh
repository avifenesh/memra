#!/usr/bin/env bash
# q-pair.sh (2x RTX PRO 6000 WS pod): main 6978f5fac plus four lanes merged with it.
#   idx    lane/dsv4-indexer-door-451-20260923 798e12674  indexer knee dispatch (#451)
#   sink   lane/dsv4-sink-attn-20260923        764aca9f1  two-launch sink attention (#18)
#   hc2    lane/dsv4-mhc-20260923              d9df672be  fused HC finish (#19)
#   fused2 lane/dsv4-moe-fused-20260923        899aac4d0  fused one-token MoE pair (#17)
# Builds, lane component gates, the DSpark served identity gate per lane, then two served
# campaigns, one boot per row, under /tmp/memra-gpu.lock:
#   short: base/sink/hc2/fused2 share one control, Williams order for 4 arms x5 reps,
#          cells-spec.txt, DSpark then plain.
#   long:  base/idx, l b b l l b b l, cells-long.txt (32k and 64k prompts), plain then DSpark.
set -u
S=/root/rcpt/q-pair.summary; R=/root/rcpt; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
. /root/.cargo/env 2>/dev/null
while ! grep -q SETUP_DONE /root/setup.log 2>/dev/null; do sleep 30; . /root/.cargo/env 2>/dev/null; done
export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm --format=csv,noheader >> $S
declare -A C=([main]=6978f5fac [idx]=798e12674 [sink]=764aca9f1 [hc2]=d9df672be [fused2]=899aac4d0)
cd /root/lane/memra && git fetch -q origin main lane/dsv4-indexer-door-451-20260923 lane/dsv4-sink-attn-20260923 \
  lane/dsv4-mhc-20260923 lane/dsv4-moe-fused-20260923
for a in main idx sink hc2 fused2; do
  git worktree add -f /root/lane/t-$a ${C[$a]} > /dev/null 2>&1
  echo "tree $a $(git -C /root/lane/t-$a rev-parse HEAD)" >> $S
done
bash /root/box/build.sh /root/lane/t-main /root/lane/target-main main
for a in idx sink hc2 fused2; do
  [[ -d /root/lane/target-$a ]] || cp -a /root/lane/target-main /root/lane/target-$a
  bash /root/box/build.sh /root/lane/t-$a /root/lane/target-$a $a
done
for a in main idx sink hc2 fused2; do
  echo "build $a $(grep -hE 'EXIT' /root/build-$a.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$a/release/memra-server | cut -c1-16)" >> $S
done
# Checkpoint: the download runs from setup; every scored row needs the verified bytes.
while ! grep -q DL_EXIT /root/dl2.log 2>/dev/null; do sleep 30; done
grep -q DL_EXIT=0 /root/dl2.log || { echo "DL FAILED $(tail -c 300 /root/dl2.log)" >> $S; exit 1; }
bash /root/box/hashchk.sh; echo "hashchk $(grep HASHCHK /root/hashchk.log)" >> $S
grep -q 'bad=0' /root/hashchk.log || { echo "HASH FAILED" >> $S; exit 1; }
# Lane component gates, lane binaries only.
testbin() {  # testbin <lane> <cargo test selector...>
  local a=$1; shift
  ( cd /root/lane/t-$a && CARGO_TARGET_DIR=/root/lane/target-$a cargo test --release -j 44 -p memra-engine "$@" --no-run ) > $R/$a-test-build.log 2>&1
  grep -o "/root/lane/target-$a/release/deps/[a-z0-9_]*-[0-9a-f]*" $R/$a-test-build.log | tail -1
}
T=$(testbin sink --test dsv4_sink_attn_st_gpu)
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/sink/component $T --ignored --nocapture --test-threads=1
echo "sink component $(sha256sum $T | cut -c1-16) $(grep -E 'cases|red arm|test result|GATE_DONE' $R/sink/component/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
T=$(testbin hc2 --test dsv4_hc_finish_gpu)
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/hc2/component $T --ignored --nocapture --test-threads=1 dsv4_hc_finish_is_bit_identical
echo "hc2 component $(sha256sum $T | cut -c1-16) $(grep -E 'EXACT|test result|panicked|GATE_DONE' $R/hc2/component/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/hc2/timing $T --ignored --nocapture --test-threads=1 dsv4_hc_finish_timing
grep TIMING $R/hc2/timing/gate.log >> $S
T=$(testbin fused2 --lib)
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/fused2/component $T dsv4_grouped:: --ignored --nocapture --test-threads=1
echo "fused2 component $(sha256sum $T | cut -c1-16) $(grep -E 'test result|GATE_DONE' $R/fused2/component/gate.log | tr '\n' ' ')" >> $S
# DSpark served identity gate on every lane binary.
for a in idx sink hc2 fused2; do
  L=/root/lane/target-$a/release; wait_lock
  ( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
    /root/box/gate.sh $R/$a/dspark-served $L/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/$a/dspark-served/out 2 0,1 --served )
  echo "$a dspark-served $(sha256sum $L/dsv4-gpu-dspark-gate | cut -c1-16) $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $R/$a/dspark-served/gate.log | head -n 6 | tr '\n' ' ')" >> $S
done
bin() { [[ $1 == base ]] && echo /root/lane/target-main/release/memra-server || echo /root/lane/target-$1/release/memra-server; }
row() {  # row <dir> <arm> <cells> [env...]
  local dir=$1 arm=$2 cells=$3; shift 3; wait_lock
  /root/box/cell.sh $dir $(bin $arm) $cells "$@" > $dir.out 2>&1
  local rc=$?; echo "${dir#$R/} rc=$rc fused=$(grep -c 'dsv4-moe-fused. ENGAGED' $dir/serve.log 2>/dev/null) $(grep -hE 'CELL ' $dir/controller.log 2>/dev/null | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
}
ORDER="base sink fused2 hc2 sink hc2 base fused2 hc2 fused2 sink base fused2 base hc2 sink base sink fused2 hc2"
mkdir -p $R/short/spec $R/short/plain $R/long/plain $R/long/spec
i=0; for arm in $ORDER; do i=$((i+1)); row $R/short/spec/r$i-$arm $arm /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark; done
i=0; for arm in $ORDER; do i=$((i+1)); row $R/short/plain/r$i-$arm $arm /root/box/cells-spec.txt; done
i=0; for arm in idx base base idx idx base base idx; do i=$((i+1)); row $R/long/plain/r$i-$arm $arm /root/box/cells-long.txt; done
i=0; for arm in idx base base idx idx base base idx; do i=$((i+1)); row $R/long/spec/r$i-$arm $arm /root/box/cells-long.txt MEMRA_DSV4_DRAFTER=dspark; done
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> $S
echo PAIR_DONE >> $S
