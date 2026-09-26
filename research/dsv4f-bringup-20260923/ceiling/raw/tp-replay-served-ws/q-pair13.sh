#!/usr/bin/env bash
# q-pair13.sh (2x RTX PRO 6000 WS pod), after q-pair12.sh: memra #710 served TP/EP plain on the
# full-token replay graphs. Measurement branch lane/dsv4-tpep-serve-measure-20260924 758b05089 (served
# TP/EP selector, replay capacity to 16384, greedy replay, checks on in replay, expert-split
# drafter). Served cells-spec, serial route, one boot per row, order Pp Tr Tr Pp Pp Tr (N=3):
# Pp = PP-2 plain (host sampler, the default); Tr = TP/EP plain on replay (device sampler, so
# the sampled cells replay too).
while ! grep -q PAIR12_DONE /root/rcpt/q-pair12.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair13.summary; R=/root/rcpt/tpreplay-serve; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-serve-measure-20260924
git worktree add -f /root/lane/t-tpr 758b05089 > /dev/null 2>&1; git -C /root/lane/t-tpr checkout -q --detach 758b05089
[[ -d /root/lane/target-tpr ]] || cp -a /root/lane/target-tpm /root/lane/target-tpr
bash /root/box/build.sh /root/lane/t-tpr /root/lane/target-tpr tpr
X=/root/lane/target-tpr/release
echo "build tpr $(git -C /root/lane/t-tpr rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-tpr.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
arm_env() {
  case $1 in
    Pp) echo "MEMRA_DSV4_SESSIONS=1" ;;
    Tr) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn MEMRA_DSV4_SAMPLER=device" ;;
  esac
}
i=0
for arm in Pp Tr Tr Pp Pp Tr; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $d $X/memra-server /root/box/cells-spec.txt $(arm_env $arm) > $d.out 2>&1
  echo "serve r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') armed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hE 'FATAL|engine-error|not armed' $d/serve.log 2>/dev/null | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo PAIR13_DONE >> $S
