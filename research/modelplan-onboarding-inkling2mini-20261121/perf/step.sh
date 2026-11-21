#!/bin/bash
# usage: step.sh NAME NEWBIN [BASEBIN]: gate, bit identity vs base, interleaved timed decode A/B
N=$1; NEW=$2; BASE=${3:-/home/avifenesh/.cache/inkling2mini/perf/base-dig}
O=/home/avifenesh/.cache/inkling2mini/perf/$N
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh
mkdir -p $O
cat $BASE/SHA256SUMS $NEW/SHA256SUMS > $O/binaries.sha256
cat /proc/loadavg
bash $C; rc=$?; echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"; fi
$NEW/ink2-gpu-gate > $O/gate.txt 2>&1; echo "GATE_RC=$?"; grep -c ' OK$' $O/gate.txt; grep FAIL $O/gate.txt
for cls in declared float; do
  for arm in base new; do
    B=$BASE; [ $arm = new ] && B=$NEW
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk 1 --out $O/id-$cls-p1c1-$arm.json > $O/id-$cls-p1c1-$arm.log 2>&1; echo "RC id $cls p1c1 $arm=$?"
    $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --out $O/id-$cls-p2-$arm.json > $O/id-$cls-p2-$arm.log 2>&1; echo "RC id $cls p2 $arm=$?"
  done
done
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for i in 1 2 3; do
  if [ $((i % 2)) -eq 1 ]; then order="base new"; else order="new base"; fi
  for arm in $order; do
    B=$BASE; [ $arm = new ] && B=$NEW
    echo "MARK $arm-$i start $(date +%T.%N)"
    $B/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --gen 32 --out $O/gen-$arm-$i.json > $O/gen-$arm-$i.log 2>&1; echo "RC gen $arm $i=$?"
    echo "MARK $arm-$i end $(date +%T.%N)"
  done
done
kill $SMI
ls $T | head -30
