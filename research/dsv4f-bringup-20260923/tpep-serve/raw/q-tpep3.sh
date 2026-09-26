#!/usr/bin/env bash
# q-tpep3.sh: served TP/EP A/B with the #679 deferred partition checks (lane/dsv4-tpep-serve-20260923
# 874d3667e, merged with main c3eb41d12 = #678 latency kernels). topology-lane.patch re-adds the
# lane-only MEMRA_DSV4_TOPOLOGY selector (never merged). Grouped GPU component tests, verify gate
# on three arms, DSpark TP/EP gate, served plain pp vs tp_ep_attn N=5 each plus tp_ep N=3,
# DSpark probe, one nsys capture per arm.
while ! grep -q FUSED_DONE /root/rcpt/q-fused.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-tpep3.summary; R=/root/rcpt/tpep3; mkdir -p $R/verify $R/plain $R/spec $R/prof /root/nsystmp
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-tpep-serve-20260923 && git worktree add -f /root/lane/memra-tpep3 FETCH_HEAD > /dev/null 2>&1
cd /root/lane/memra-tpep3 && git apply /root/box/topology-lane.patch && echo "tree $(git rev-parse HEAD) + topology-lane.patch $(sha256sum /root/box/topology-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-tpep3 ]] || cp -a /root/lane/target-fused /root/lane/target-tpep3
bash /root/box/build.sh /root/lane/memra-tpep3 /root/lane/target-tpep3 tpep3
grep -E 'EXIT|BUILD' /root/build-tpep3.log | tr '\n' ' ' >> $S; echo >> $S
X=/root/lane/target-tpep3/release; M=/data/dsv4f/nvfp4; TAPE=/root/box/tape-rebuild.txt; FX=/root/box/dspark-fx-tape416.json
echo "bin server $(sha256sum $X/memra-server | cut -c1-16) verify $(sha256sum $X/dsv4_tp_ep_verify_gate | cut -c1-16) dspark $(sha256sum $X/dsv4-gpu-dspark-gate | cut -c1-16)" >> $S
( . /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
  cd /root/lane/memra-tpep3 && CARGO_TARGET_DIR=/root/lane/target-tpep3 cargo test --release -j 48 -p memra-engine --lib --no-run ) > $R/test-build.log 2>&1
T=$(grep -o '/root/lane/target-tpep3/release/deps/memra_engine-[0-9a-f]*' $R/test-build.log | tail -1)
echo "test-bin $T $(sha256sum $T | cut -c1-16)" >> $S
wait_lock; NVIDIA_TF32_OVERRIDE=0 /root/box/gate.sh $R/component $T dsv4_grouped:: --ignored --nocapture --test-threads=1
echo "component $(grep -E 'test result|GATE_DONE' $R/component/gate.log | tr '\n' ' ')" >> $S
wait_lock
( export MEMRA_DSV4_DECODE_PATH=device
  /root/box/gate.sh $R/verify/pp $X/dsv4_tp_ep_verify_gate $M $TAPE pp )
echo "verify pp $(grep -hE '^(PASS|FAILED|SEQ|CHUNK|VERIFY)|GATE_DONE' $R/verify/pp/gate.log | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
for at in 0 1; do
  wait_lock
  ( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=$at
    /root/box/gate.sh $R/verify/tpep-at$at $X/dsv4_tp_ep_verify_gate $M $TAPE tpep )
  echo "verify tpep-at$at $(grep -hE '^(PASS|FAILED|SEQ|CHUNK|VERIFY)|GATE_DONE' $R/verify/tpep-at$at/gate.log | tr '\n' ' ')" >> $S
done
wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark
  /root/box/gate.sh $R/dspark-tpep $X/dsv4-gpu-dspark-gate $M $FX $R/dspark-tpep/out 2 0,1 --served --tpep )
echo "dspark-tpep $(grep -hE 'GATE \[|FAIL|panicked|REFUSE|GATE_DONE' $R/dspark-tpep/gate.log | head -n 8 | tr '\n' ' ')" >> $S
if ! grep -q 'PASS topology=tpep' $R/verify/tpep-at1/gate.log 2>/dev/null; then
  echo "SKIP: tpep-at1 verify gate did not pass" >> $S; echo TPEP3_DONE >> $S; exit 0
fi
arm_env() {
  case $1 in
    pp) echo "MEMRA_ENV_AUDIT=warn" ;;
    tpa) echo "MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
    tp) echo "MEMRA_DSV4_TOPOLOGY=tp_ep MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
  esac
}
i=0
for arm in pp tpa tpa pp pp tpa tpa pp pp tpa tp tp tp; do
  i=$((i+1)); wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $R/plain/r$i-$arm $X/memra-server /root/box/cells-spec.txt $(arm_env $arm) > $R/plain/r$i-$arm.out 2>&1
  echo "plain r$i $arm rc=$? $(grep -hE 'CELL ' $R/plain/r$i-$arm/controller.log | grep -v warmup | tr '\n' ' ') $(grep -h 'FATAL' $R/plain/r$i-$arm/serve.log 2>/dev/null | head -1 | cut -c1-200)" >> $S
done
i=0; tpa_ok=1
for arm in tpa pp tp pp tpa tpa pp pp tpa tpa pp; do
  i=$((i+1))
  [[ $arm == tpa && $tpa_ok == 0 ]] && continue
  [[ $arm == pp && $tpa_ok == 0 && $i -gt 2 ]] && continue
  wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $R/spec/s$i-$arm $X/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark $(arm_env $arm) > $R/spec/s$i-$arm.out 2>&1
  rc=$?
  echo "spec s$i $arm rc=$rc $(grep -hE 'CELL ' $R/spec/s$i-$arm/controller.log 2>/dev/null | grep -v warmup | tr '\n' ' ') $(grep -h 'FATAL' $R/spec/s$i-$arm/serve.log 2>/dev/null | head -1 | cut -c1-200)" >> $S
  [[ $arm == tpa && $rc != 0 ]] && tpa_ok=0
done
for arm in tpa pp; do
  wait_lock
  # shellcheck disable=SC2046
  TMPDIR=/root/nsystmp bash /root/box/prof-served.sh $R/prof/$arm $X/memra-server $(arm_env $arm) > $R/prof/$arm.out 2>&1
  echo "prof $arm rc=$? $(grep -h PROF_DONE $R/prof/$arm/controller.log 2>/dev/null)" >> $S
  rm -f $R/prof/$arm/nsys.sqlite
done
echo TPEP3_DONE >> $S
