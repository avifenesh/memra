#!/usr/bin/env bash
# q-v3x.sh (2x RTX PRO 6000 Server Edition): memra #710 TP/EP default flip, lane
# lane/dsv4-tpep-default-20260925 9d19b5014.
#   1. dsv4_tp_replay_long_gate: the replay-to-eager handoff (limit 640, greedy and vendor
#      sampling), the real 16384 limit crossed (greedy), and the no-handoff 304-step row.
#   2. Served A/B, one boot per row, order Df Pp Pp Df Df Pp (N=3): Df = the naked default
#      (TP/EP attention TP2, device sampler, replay), Pp = MEMRA_DSV4_TOPOLOGY=pp.
#   3. Hc: the default with the parked host cache on, so repeated prompts restore then arm.
#   4. DSpark: MEMRA_DSV4_DRAFTER=dspark on both placements, order Df Pp Pp Df (N=2).
set -u
S=/root/rcpt/q-v3x.summary; R=/root/rcpt/tpep-default; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-default-20260925
git worktree add -f /root/lane/t-flip 9d19b5014 > /dev/null 2>&1; git -C /root/lane/t-flip checkout -q --detach 9d19b5014
[[ -d /root/lane/target-flip ]] || cp -a /root/lane/target-tpa2 /root/lane/target-flip
bash /root/box/build.sh /root/lane/t-flip /root/lane/target-flip flip
X=/root/lane/target-flip/release
echo "build flip $(git -C /root/lane/t-flip rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-flip.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16) gate $(sha256sum $X/dsv4_tp_replay_long_gate | cut -c1-16)" >> $S
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
gate() { # name steps env...
  local name=$1 steps=$2; shift 2; local d=$R/gate-$name; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
    env NVIDIA_TF32_OVERRIDE=0 "$@" $X/dsv4_tp_replay_long_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt $steps 9>&- > $d/gate.log 2>&1 )
  echo "gate $name rc=$? $(grep -hE 'PASS:|FAILED|HANDOFF|FIRST|panicked|TIME rep=2' $d/gate.log | cut -c1-230 | tr '\n' ' ')" >> $S
}
gate handoff-greedy 500 DSV4_REPLAY_GATE_CAPACITY=2048 DSV4_REPLAY_GATE_LIMIT=640 DSV4_REPLAY_GATE_SAMPLING=greedy
gate handoff-default 500 DSV4_REPLAY_GATE_CAPACITY=2048 DSV4_REPLAY_GATE_LIMIT=640
gate plain-304 304
gate limit16384-greedy 16100 DSV4_REPLAY_GATE_CAPACITY=20000 DSV4_REPLAY_GATE_SAMPLING=greedy
arm_env() {
  case $1 in
    Df) echo "MEMRA_ENV_AUDIT=on" ;;
    Pp) echo "MEMRA_DSV4_TOPOLOGY=pp" ;;
    Hc) echo "MEMRA_DSV4_KV_HOST_MB=16384" ;;
    DsDf) echo "MEMRA_DSV4_DRAFTER=dspark" ;;
    DsPp) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_TOPOLOGY=pp" ;;
  esac
}
row() { # i arm cells
  local i=$1 arm=$2 cells=$3; local d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  timeout -k 30 2400 /root/box/cell.sh $d $X/memra-server $cells $(arm_env $arm) > $d.out 2>&1
  local rc=$?
  echo "serve r$i $arm rc=$rc $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') notarmed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hE 'placement TP|placement PP|plain sampler' $d/serve.log 2>/dev/null | head -1 | cut -c1-140) $(grep -hE 'FATAL|engine-error|not armed' $d/serve.log 2>/dev/null | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
  if [[ $rc == 124 || $rc == 137 ]]; then nvidia-smi > $d.hang-nvsmi.txt 2>&1; pkill -9 -x memra-server; sleep 20; fi
}
i=0
for arm in Df Pp Pp Df Df Pp; do i=$((i+1)); row $i $arm /root/box/cells-flip.txt; done
i=$((i+1)); row $i Hc /root/box/cells-spec.txt
for arm in DsDf DsPp DsPp DsDf; do i=$((i+1)); row $i $arm /root/box/cells-spec.txt; done
echo V3X_DONE >> $S
