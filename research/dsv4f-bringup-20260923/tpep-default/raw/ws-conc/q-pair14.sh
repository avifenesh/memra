#!/usr/bin/env bash
# q-pair14.sh (2x RTX PRO 6000 WS pod): memra #710 concurrency cost of the TP/EP flip. Same
# measurement binary as q-pair13 (lane/dsv4-tpep-serve-measure-20260924 758b05089). Pp = PP-2 at
# its load default (two pipelined lanes, host sampler); Tr = TP/EP attention TP on the full-token
# replay (one lane, device sampler). Cells c1/c2/c4 plus a 2k-context c2, one boot per row, order
# Pp Tr Tr Pp Pp Tr (N=3). Each row runs under a 40-minute timeout so a #722 hang ends the row.
set -u
S=/root/rcpt/q-pair14.summary; R=/root/rcpt/tpep-conc; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
X=/root/lane/target-tpr/release
echo "binary tpr 758b05089 server $(sha256sum $X/memra-server | cut -c1-16)" >> $S
arm_env() {
  case $1 in
    Pp) echo "MEMRA_ENV_AUDIT=warn" ;;
    Tr) echo "MEMRA_DSV4_TOPOLOGY=tp_ep_attn MEMRA_DSV4_EP=pair MEMRA_ENV_AUDIT=warn MEMRA_DSV4_SAMPLER=device" ;;
  esac
}
i=0
for arm in Pp Tr Tr Pp Pp Tr; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  # shellcheck disable=SC2046
  timeout -k 30 2400 /root/box/cell.sh $d $X/memra-server /root/box/cells-conc.txt $(arm_env $arm) > $d.out 2>&1
  rc=$?
  echo "serve r$i $arm rc=$rc $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ') lanes=$(grep -hoE '[0-9]+ serving lane\(s\)' $d/serve.log 2>/dev/null | head -1) notarmed=$(grep -c 'TP/EP replay not armed' $d/serve.log 2>/dev/null) $(grep -hE 'FATAL|engine-error' $d/serve.log 2>/dev/null | head -2 | cut -c1-200 | tr '\n' ' ')" >> $S
  if [[ $rc == 124 || $rc == 137 ]]; then
    nvidia-smi > $d.hang-nvsmi.txt 2>&1; pkill -9 -x memra-server; sleep 20
  fi
done
echo PAIR14_DONE >> $S
