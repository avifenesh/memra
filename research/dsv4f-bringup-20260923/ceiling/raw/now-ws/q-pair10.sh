#!/usr/bin/env bash
# q-pair10.sh (2x RTX PRO 6000 WS pod): current-main baseline for the ceiling write-up, main 2ad82c2e9
# (fused one-token MoE #704, compressor diet #706, prefill tiles #713, commit cleanup #715, B-row
# #716 off, pipelining #699 on). Served cells-spec, one boot per row: plain (defaults) x3,
# DSpark x3, DSpark with MEMRA_DSV4_VT=slot x3, order P D V V D P P D V, then one nsys anatomy
# of the plain serial route.
set -u
S=/root/rcpt/q-pair10.summary; R=/root/rcpt/now; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin main
git worktree add -f /root/lane/t-now 2ad82c2e9 > /dev/null 2>&1; git -C /root/lane/t-now checkout -q --detach 2ad82c2e9
[[ -d /root/lane/target-now ]] || cp -a /root/lane/target-pfix /root/lane/target-now
bash /root/box/build.sh /root/lane/t-now /root/lane/target-now now
B=/root/lane/target-now/release/memra-server
echo "build now $(git -C /root/lane/t-now rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-now.log | tr '\n' ' ') server $(sha256sum $B | cut -c1-16)" >> $S
arm_env() { case $1 in P) echo "MEMRA_DSV4_SESSIONS=1" ;; D) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_DRAFTER=dspark" ;; V) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_VT=slot" ;; esac; }
i=0
for arm in P D V V D P P D V; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $d $B /root/box/cells-spec.txt $(arm_env $arm) > $d.out 2>&1
  echo "now r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -hE 'FATAL|engine-error' $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
wait_lock
/root/box/prof-served.sh $R/prof-plain $B MEMRA_DSV4_SESSIONS=1 > $R/prof-plain.out 2>&1
echo "prof plain $(tail -n 1 $R/prof-plain.out | cut -c1-300)" >> $S
echo PAIR10_DONE >> $S
