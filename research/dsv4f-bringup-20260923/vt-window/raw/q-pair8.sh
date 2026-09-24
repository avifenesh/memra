#!/usr/bin/env bash
# q-pair8.sh (2x RTX PRO 6000 WS pod), after q-pair4.sh: the confidence-window rerun. q-pair4's two
# vt rows passed MEMRA_DSV4_VT=slot@0.5, which the engine refuses per request ("unknown (off | slot)"),
# so both rows are void. Same binary (target-main5, main 25bbb91f5), same cells, one boot per row,
# order vt d6 d6 vt vt d6 (N=3): MEMRA_DSV4_VT=slot with tau at its 0.5 default against depth 6.
# bench.py now counts a stream error or a stream without finish_reason as an error.
while ! grep -q PAIR4_DONE /root/rcpt/q-pair4.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair8.summary; R=/root/rcpt/vt; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
B=/root/lane/target-main5/release/memra-server
echo "base $(sha256sum $B | cut -c1-16) bench $(md5sum /root/box/bench.py | cut -c1-12)" >> $S
i=0
for arm in vt d6 d6 vt vt d6; do
  i=$((i+1)); wait_lock; d=$R/r$i-$arm
  if [[ $arm == vt ]]; then set -- MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_VT=slot; else set -- MEMRA_DSV4_DRAFTER=dspark; fi
  /root/box/cell.sh $d $B /root/box/cells-spec.txt "$@" > $d.out 2>&1
  echo "vt r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -hE 'FATAL|engine-error' $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
echo PAIR8_DONE >> $S
