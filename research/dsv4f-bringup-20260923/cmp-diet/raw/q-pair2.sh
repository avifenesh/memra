#!/usr/bin/env bash
# q-pair2.sh (2x RTX PRO 6000 WS pod), after q-pair.sh: the compressor diet (fewer copies plus
# BF16 weight storage; lane/dsv4-cmp-snap-20260923 bec39b449, on main 25bbb91f5) against that main, then nsys
# anatomy of main and the lane on the plain and DSpark served routes.
while ! grep -q PAIR_DONE /root/rcpt/q-pair.summary 2>/dev/null; do sleep 30; done
bash /root/box/q-pairP.sh  # pipelined B-row groups first (memra #667)
set -u
S=/root/rcpt/q-pair2.summary; R=/root/rcpt; mkdir -p $R/cmp
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-cmp-snap-20260923
git worktree add -f /root/lane/t-cmp bec39b449 > /dev/null 2>&1
git worktree add -f /root/lane/t-main5 25bbb91f5 > /dev/null 2>&1
echo "tree main5 $(git -C /root/lane/t-main5 rev-parse HEAD)" >> $S
echo "tree cmp $(git -C /root/lane/t-cmp rev-parse HEAD)" >> $S
[[ -d /root/lane/target-cmp ]] || cp -a /root/lane/target-main /root/lane/target-cmp
bash /root/box/build.sh /root/lane/t-cmp /root/lane/target-cmp cmp
[[ -d /root/lane/target-main5 ]] || cp -a /root/lane/target-main /root/lane/target-main5
bash /root/box/build.sh /root/lane/t-main5 /root/lane/target-main5 main5
L=/root/lane/target-cmp/release; B=/root/lane/target-main5/release
echo "build cmp $(grep -hE 'EXIT' /root/build-cmp.log | tr '\n' ' ') server $(sha256sum $L/memra-server | cut -c1-16) base $(sha256sum $B/memra-server | cut -c1-16)" >> $S
( cd /root/lane/t-cmp && CARGO_TARGET_DIR=/root/lane/target-cmp cargo test --release -j 44 -p memra-engine --test dsv4_island_bf16_gpu --no-run ) > $R/cmp/test-build.log 2>&1
T=$(grep -o '/root/lane/target-cmp/release/deps/dsv4_island_bf16_gpu-[0-9a-f]*' $R/cmp/test-build.log | tail -1)
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/cmp/island $T --ignored --nocapture --test-threads=1
echo "cmp island $(sha256sum $T | cut -c1-16) $(grep -hE 'EXACT|test result|panicked|GATE_DONE' $R/cmp/island/gate.log | tr '\n' ' ')" >> $S
wait_lock
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/cmp/dspark-served $L/dsv4-gpu-dspark-gate /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json $R/cmp/dspark-served/out 2 0,1 --served )
echo "cmp dspark-served $(sha256sum $L/dsv4-gpu-dspark-gate | cut -c1-16) $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $R/cmp/dspark-served/gate.log | head -n 6 | tr '\n' ' ')" >> $S
row() {  # row <dir> <arm> [env...]
  local dir=$1 arm=$2; shift 2; wait_lock
  local d=$B/memra-server; [[ $arm == lane ]] && d=$L/memra-server
  /root/box/cell.sh $dir $d /root/box/cells-spec.txt "$@" > $dir.out 2>&1
  local rc=$?; echo "${dir#$R/} rc=$rc $(grep -hE 'CELL ' $dir/controller.log 2>/dev/null | grep -v warmup | cut -c1-200 | tr '\n' ' ')" >> $S
}
mkdir -p $R/cmp/plain $R/cmp/spec
i=0; for arm in lane base base lane lane base base lane lane base; do i=$((i+1)); row $R/cmp/plain/r$i-$arm $arm; done
i=0; for arm in lane base base lane lane base base lane lane base; do i=$((i+1)); row $R/cmp/spec/s$i-$arm $arm MEMRA_DSV4_DRAFTER=dspark; done
if command -v nsys > /dev/null || [[ -x /usr/local/cuda/bin/nsys ]]; then
  # plain-now / dspark-now: current main a8455d29f (sink + HC2) through the q-pairR build, one lane (the serial route)
  N=/root/lane/target-brow/release/memra-server
  for p in "plain-main $B/memra-server" "plain-cmp $L/memra-server" "dspark-main $B/memra-server MEMRA_DSV4_DRAFTER=dspark" "plain-now $N MEMRA_DSV4_SESSIONS=1" "dspark-now $N MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_DRAFTER=dspark"; do
    set -- $p; name=$1; shift; wait_lock
    /root/box/prof-served.sh $R/prof/$name "$@" > $R/prof-$name.out 2>&1
    echo "prof $name $(tail -n 1 $R/prof-$name.out | cut -c1-300)" >> $S
  done
else
  echo "prof skipped: no nsys" >> $S
fi
echo PAIR2_DONE >> $S
