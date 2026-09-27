#!/usr/bin/env bash
# q-pair22.sh (2x RTX PRO 6000 WS pod): memra #710 served A/B of the TP/EP B-row graphs, lane
# lane/dsv4-tp-rows-graph-20260926 ($1) against main cf6b82db4 (the flip: TP/EP one lane).
# Final head of the lane (7150cb5f1: four lanes, B-row graphs, multi-row dense-fast GEMV and
# dots, hoisted compressor projections). Gd = lane default, Pp = lane
# with MEMRA_DSV4_TOPOLOGY=pp, Mn = main. Cells c1/c2/c4/sampled c2/2k c2, one boot per row,
# order Gd Pp Mn x3, 45-minute timeout per row (#722).
set -u
SHA=$1
S=/root/rcpt/q-pair22.summary; R=/root/rcpt/rows-final-served; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-graph-20260926 main
git -C /root/lane/t-rg checkout -q --detach $SHA
bash /root/box/build.sh /root/lane/t-rg /root/lane/target-rg rg
git worktree add -f /root/lane/t-main10 cf6b82db4 > /dev/null 2>&1; git -C /root/lane/t-main10 checkout -q --detach cf6b82db4
[[ -x /root/lane/target-main10/release/memra-server ]] || bash /root/box/build.sh /root/lane/t-main10 /root/lane/target-main10 main10
for n in rg main10; do echo "build $n $(git -C /root/lane/t-$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$n.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$n/release/memra-server | cut -c1-16)" >> $S; done
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { case $1 in Mn) echo /root/lane/target-main10/release/memra-server ;; *) echo /root/lane/target-rg/release/memra-server ;; esac; }
arm_env() {
  case $1 in
    Gd|Mn) echo "MEMRA_ENV_AUDIT=on" ;;
    G4) echo "MEMRA_DSV4_SESSIONS=4" ;;
    Pp) echo "MEMRA_DSV4_TOPOLOGY=pp" ;;
  esac
}
i=0
for arm in Gd Pp Mn Mn Pp Gd Gd Pp Mn; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $arm) /root/box/cells-conc.txt $(arm_env $arm) > $d.out 2>&1
  rc=$?
  echo "serve r$i $arm rc=$rc $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') notarmed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hoE '[0-9]+ serving lane\(s\)' $d/serve.log 2>/dev/null | head -1) $(grep -hE 'FATAL|engine-error|B-row' $d/serve.log 2>/dev/null | grep -v 'steps up to' | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
  if [[ $rc == 124 || $rc == 137 ]]; then nvidia-smi > $d.hang-nvsmi.txt 2>&1; pkill -9 -x memra-server; sleep 20; fi
done
echo PAIR22_DONE >> $S
