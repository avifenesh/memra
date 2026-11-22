#!/bin/bash
# usage: ctx.sh NAME PROMPTLEN BASEBIN NEWBIN: decode 32 tokens after a PROMPTLEN prompt, interleaved, 3 takes each
N=$1; PL=$2; BASE=$3; NEW=$4
O=/home/avifenesh/.cache/inkling2mini/perf/$N; mkdir -p $O
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
CTX=$(( (PL + 64 + 1023) / 1024 * 1024 ))
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for i in 1 2 3; do
  order="base new"; [ $((i % 2)) -eq 0 ] && order="new base"
  for arm in $order; do
    B=$BASE; [ $arm = new ] && B=$NEW
    echo "MARK $arm-$i start $(date +%T.%N)"
    $B/ink2-gpu $M --class declared --tape $T/p3_logscale.exact_f32 --prompt-len $PL --context $CTX --gen 32 --out $O/gen-$arm-$i.json > $O/gen-$arm-$i.log 2>&1; echo "RC gen $arm $i=$?"
    echo "MARK $arm-$i end $(date +%T.%N)"
  done
done
kill $SMI
