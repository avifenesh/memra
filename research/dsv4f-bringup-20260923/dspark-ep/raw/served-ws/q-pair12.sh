#!/usr/bin/env bash
# q-pair12.sh (2x RTX PRO 6000 WS pod), after q-pair11.sh: memra #718 and #710, the served TP/EP
# program with the expert-split drafter. Measurement branch lane/dsv4-tpep-serve-measure-20260924
# d1be33841 (the #718 lane plus a never-merged load-time MEMRA_DSV4_TOPOLOGY selector). First the
# DSpark gate on TP/EP with the fixed fused expectation, then served cells-spec, one boot per
# row, serial route: PP plain (Pp), TP/EP plain (Tp), PP DSpark (Pd), TP/EP DSpark (Td), order
# Pp Tp Pd Td Td Pd Tp Pp Pp Tp Pd Td (N=3 each).

set -u
S=/root/rcpt/q-pair12.summary; R=/root/rcpt/tpserve; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-serve-measure-20260924
git worktree add -f /root/lane/t-tpm d1be33841 > /dev/null 2>&1; git -C /root/lane/t-tpm checkout -q --detach d1be33841
[[ -d /root/lane/target-tpm ]] || cp -a /root/lane/target-now /root/lane/target-tpm
bash /root/box/build.sh /root/lane/t-tpm /root/lane/target-tpm tpm
X=/root/lane/target-tpm/release
echo "build tpm $(git -C /root/lane/t-tpm rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-tpm.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16) gate $(sha256sum $X/dsv4-gpu-dspark-gate | cut -c1-16)" >> $S
wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1
  /root/box/gate.sh $R/gate-tpep $X/dsv4-gpu-dspark-gate $M $FX $R/gate-tpep/out 2 0,1 --served --tpep )
echo "gate tpep $(grep -hoE 'proposal sha [0-9a-f]+' $R/gate-tpep/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|panicked|out of memory|GATE_DONE' $R/gate-tpep/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
arm_env() {
  case $1 in
    Pp) echo "MEMRA_DSV4_SESSIONS=1" ;;
    Tp) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
    Pd) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_DRAFTER=dspark" ;;
    Td) echo "MEMRA_DSV4_SESSIONS=1 MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn MEMRA_DSV4_DRAFTER=dspark" ;;
  esac
}
i=0
for arm in Pp Tp Pd Td Td Pd Tp Pp Pp Tp Pd Td; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $d $X/memra-server /root/box/cells-spec.txt $(arm_env $arm) > $d.out 2>&1
  echo "serve r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') $(grep -hE 'FATAL|engine-error|admit-mem\] route=dsv4-thread model=\"dsv4f\" calibrated' $d/serve.log 2>/dev/null | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
done
echo PAIR12_DONE >> $S
