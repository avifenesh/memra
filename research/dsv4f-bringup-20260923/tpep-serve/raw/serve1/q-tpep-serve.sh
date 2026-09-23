#!/usr/bin/env bash
# q-tpep-serve.sh: served TP/EP against PP-2 (memra #454) on the lane tree q-tpep built.
# Runs only if the verify gate passed on tpep-at1 and the DSpark gate passed on the TP/EP walk.
# Plain: pp vs tp_ep_attn, order A B B A A B B A A B, plus tp_ep (replicated attention) N=2.
# DSpark: pp vs tp_ep_attn, same order. One boot per row, 250 ms telemetry from cell.sh.
while ! grep -q DSPARK_TPEP_DONE /root/rcpt/q-dspark-tpep.summary 2>/dev/null; do sleep 30; done
S=/root/rcpt/q-tpep-serve.summary; R=/root/rcpt/tpep-serve; mkdir -p $R/plain $R/spec
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
X=/root/lane/target-mrow/release
echo "tree $(git -C /root/lane/memra-tpep rev-parse HEAD) bin $(sha256sum $X/memra-server | cut -c1-16)" >> $S
if ! grep -q 'PASS topology=tpep' /root/rcpt/tpep-verify/tpep-at1/gate.log 2>/dev/null; then
  echo "SKIP: tpep-at1 verify gate did not pass" >> $S; echo TPEP_SERVE_DONE >> $S; exit 0
fi
if ! grep -q 'GPU DSPARK GATE \[PASS\]' /root/rcpt/dspark-tpep/tpep/gate.log 2>/dev/null; then
  echo "NOTE: DSpark TP/EP gate did not pass; DSpark rows skipped" >> $S; SPEC=0
else
  SPEC=1
fi
arm_env() {
  case $1 in
    pp) echo "MEMRA_ENV_AUDIT=warn" ;;
    tpa) echo "MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
    tp) echo "MEMRA_DSV4_TOPOLOGY=tp_ep MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
  esac
}
i=0
for arm in pp tpa tpa pp pp tpa tpa pp pp tpa tp tp; do
  i=$((i+1)); wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $R/plain/r$i-$arm $X/memra-server /root/box/cells-spec.txt $(arm_env $arm) > $R/plain/r$i-$arm.out 2>&1
  echo "plain r$i $arm rc=$? $(grep -hE 'topology|CELL ' $R/plain/r$i-$arm/serve.log $R/plain/r$i-$arm/controller.log | grep -v warmup | cut -c1-240 | tr '\n' ' ')" >> $S
done
if [[ $SPEC == 1 ]]; then
  i=0
  for arm in pp tpa tpa pp pp tpa tpa pp pp tpa; do
    i=$((i+1)); wait_lock
    # shellcheck disable=SC2046
    /root/box/cell.sh $R/spec/s$i-$arm $X/memra-server /root/box/cells-spec.txt MEMRA_DSV4_DRAFTER=dspark $(arm_env $arm) > $R/spec/s$i-$arm.out 2>&1
    echo "spec s$i $arm rc=$? $(grep -hE 'CELL ' $R/spec/s$i-$arm/controller.log | grep -v warmup | tr '\n' ' ')" >> $S
  done
fi
echo TPEP_SERVE_DONE >> $S
