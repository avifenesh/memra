#!/usr/bin/env bash
# q-v4a.sh (2x RTX PRO 6000 Server Edition): memra #710 B-row lane e7a3f32a1 (the multi-row
# dense-fast GEMV, main with the TP/EP flip merged) against main cf6b82db4 (the flip, one lane).
#   Plain concurrency, lane binary: Rw default (two lanes, B-row), R4 (four lanes), Pp (`pp`);
#   main binary: Mn (default = TP/EP one lane). Order Rw Mn Pp Rw Mn Pp Rw Mn Pp R4 R4.
#   DSpark (MEMRA_DSV4_DRAFTER=dspark): lane DsL vs main DsM, order DsL DsM DsM DsL DsL DsM.
set -u
S=/root/rcpt/q-v4a.summary; R=/root/rcpt/tp-rows2; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-20260925 main
for a in rows2:e7a3f32a1 main10:cf6b82db4; do
  n=${a%%:*}; c=${a##*:}
  git worktree add -f /root/lane/t-$n $c > /dev/null 2>&1; git -C /root/lane/t-$n checkout -q --detach $c
  [[ -d /root/lane/target-$n ]] || cp -a /root/lane/target-rows /root/lane/target-$n
  bash /root/box/build.sh /root/lane/t-$n /root/lane/target-$n $n
  echo "build $n $(git -C /root/lane/t-$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$n.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$n/release/memra-server | cut -c1-16)" >> $S
done
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { case $1 in Mn|DsM) echo /root/lane/target-main10/release/memra-server ;; *) echo /root/lane/target-rows2/release/memra-server ;; esac; }
arm_env() {
  case $1 in
    Rw|Mn) echo "MEMRA_ENV_AUDIT=on" ;;
    R4) echo "MEMRA_DSV4_SESSIONS=4" ;;
    Pp) echo "MEMRA_DSV4_TOPOLOGY=pp" ;;
    DsL|DsM) echo "MEMRA_DSV4_DRAFTER=dspark" ;;
  esac
}
row() {
  local i=$1 arm=$2 cells=$3; local d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $arm) $cells $(arm_env $arm) > $d.out 2>&1
  local rc=$?
  echo "serve r$i $arm rc=$rc $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') notarmed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hoE '[0-9]+ serving lane\(s\)' $d/serve.log 2>/dev/null | head -1) $(grep -hE 'FATAL|engine-error' $d/serve.log 2>/dev/null | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
  if [[ $rc == 124 || $rc == 137 ]]; then nvidia-smi > $d.hang-nvsmi.txt 2>&1; pkill -9 -x memra-server; sleep 20; fi
}
i=0
for arm in Rw Mn Pp Rw Mn Pp Rw Mn Pp R4 R4; do i=$((i+1)); row $i $arm /root/box/cells-conc.txt; done
for arm in DsL DsM DsM DsL DsL DsM; do i=$((i+1)); row $i $arm /root/box/cells-spec.txt; done
echo V4A_DONE >> $S
