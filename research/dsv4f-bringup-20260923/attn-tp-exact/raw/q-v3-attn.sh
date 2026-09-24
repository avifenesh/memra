cd /root/lane/memra && git fetch -q origin lane/dsv4-attn-tp-exact-20260923
git worktree add -f /root/lane/t-attn 75c55dfd4 > /dev/null 2>&1
cd /root/lane/t-attn && git apply /root/box/topology-lane.patch
echo "tree attn $(git rev-parse HEAD) + topology-lane.patch $(sha256sum /root/box/topology-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-attn ]] || cp -a /root/lane/target-pipe /root/lane/target-attn
bash /root/box/build.sh /root/lane/t-attn /root/lane/target-attn attn
X=/root/lane/target-attn/release; M=/data/dsv4f/nvfp4; TAPE=/root/box/tape-rebuild.txt; FX=/root/box/dspark-fx-tape416.json
echo "build attn $(grep -hE 'EXIT' /root/build-attn.log | tr '\n' ' ') server $(sha256sum $X/memra-server | cut -c1-16) tpep-gate $(sha256sum $X/dsv4_tp_ep_gate | cut -c1-16)" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
for at in 0 1; do
  wait_lock
  ( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=$at; /root/box/gate.sh $R/tpep-gate-at$at $X/dsv4_tp_ep_gate $M $TAPE )
  echo "tpep-gate at$at $(grep -hE '^(DIGEST|PASS)|panicked|GATE_DONE' $R/tpep-gate-at$at/gate.log | cut -c1-220 | tr '\n' ' ')" >> $S
done
d0=$(grep -h '^DIGEST' $R/tpep-gate-at0/gate.log); d1=$(grep -h '^DIGEST' $R/tpep-gate-at1/gate.log)
[[ -n "$d0" && "$d0" == "$d1" ]] && echo "DIGEST_EQUAL yes" >> $S || echo "DIGEST_EQUAL no" >> $S
wait_lock
( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $R/verify-at1 $X/dsv4_tp_ep_verify_gate $M $TAPE tpep )
echo "verify at1 $(grep -hE '^(PASS|FAILED)|panicked|GATE_DONE' $R/verify-at1/gate.log | tr '\n' ' ')" >> $S
wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1
  /root/box/gate.sh $R/dspark-tpa $X/dsv4-gpu-dspark-gate $M $FX $R/dspark-tpa/out 2 0,1 --served --tpep )
echo "dspark-tpa $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $R/dspark-tpa/gate.log | head -n 6 | tr '\n' ' ')" >> $S
# Eager vs graph on the exact program: the anchor replay protocol (REBASELINE), one load.
wait_lock
( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=1 MEMRA_DSV4_SAMPLER=device
  /root/box/gate.sh $R/replay-at1 $X/dsv4_tp_ep_sampled_perf_gate $M $TAPE --full-token-replay )
echo "replay at1 $(grep -hE '^PASS|panicked|GATE_DONE' $R/replay-at1/gate.log | cut -c1-200 | tr '\n' ' ') $(grep -c '^MEASURE' $R/replay-at1/gate.log) measures" >> $S
arm_env() {
  case $1 in
    pp) echo "MEMRA_ENV_AUDIT=warn" ;;
    tpa) echo "MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
    tp) echo "MEMRA_DSV4_TOPOLOGY=tp_ep MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn" ;;
  esac
}
row() {  # row <dir> <arm> [env...]
  local dir=$1 arm=$2; shift 2; wait_lock
  # shellcheck disable=SC2046
  /root/box/cell.sh $dir $X/memra-server /root/box/cells-spec.txt "$@" $(arm_env $arm) > $dir.out 2>&1
  local rc=$?
  echo "${dir#$R/} rc=$rc $(grep -hE 'CELL ' $dir/controller.log 2>/dev/null | grep -v warmup | cut -c1-200 | tr '\n' ' ') $(grep -hE 'FATAL|attention-TP2\]|MEMORY' $dir/serve.log 2>/dev/null | head -3 | cut -c1-200 | tr '\n' ' ')" >> $S
  return $rc
}
# Served fit probe first: DSpark on the exact attention TP2 program at the served context.
row $R/spec/fit-tpa tpa MEMRA_DSV4_DRAFTER=dspark; fit=$?
echo "FIT dspark tpa rc=$fit" >> $S
i=0; for arm in pp tpa tpa pp pp tpa tpa pp pp tpa tp tp; do i=$((i+1)); row $R/plain/r$i-$arm $arm; done
if [[ $fit == 0 ]]; then
  i=0; for arm in tpa pp pp tpa tpa pp pp tpa tpa pp; do i=$((i+1)); row $R/spec/s$i-$arm $arm MEMRA_DSV4_DRAFTER=dspark; done
fi
wait_lock
# shellcheck disable=SC2046
TMPDIR=/root/nsystmp bash /root/box/prof-served.sh $R/prof/tpa $X/memra-server $(arm_env tpa) > $R/prof/tpa.out 2>&1
echo "prof tpa $(tail -n 1 $R/prof/tpa.out | cut -c1-300)" >> $S
rm -f $R/prof/tpa/nsys.sqlite
echo V3_DONE >> $S
