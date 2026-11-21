#!/bin/bash
# usage: cls2.sh NAME CLASS BASEBIN "BASEFLAGS" NEWBIN "NEWFLAGS" [PROMPTLEN] [notime]
N=$1; CLS=$2; BASE=$3; BF=$4; NEW=$5; NF=$6; PL=${7:-0}; NOTIME=$8
O=/home/avifenesh/.cache/inkling2mini/perf/$N; mkdir -p $O
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
{ echo "base $BASE $BF"; cat $BASE/SHA256SUMS; echo "new $NEW $NF"; cat $NEW/SHA256SUMS; } > $O/binaries.sha256
if [ "$PL" = 0 ]; then
for arm in base new; do
  B=$BASE; F=$BF; [ $arm = new ] && { B=$NEW; F=$NF; }
  $B/ink2-gpu $M --class $CLS --tape $T/p1_chat.exact_f32 --score --chunk 1 $F --out $O/id-p1c1-$arm.json > $O/id-p1c1-$arm.log 2>&1; echo "RC id p1c1 $arm=$?"
  $B/ink2-gpu $M --class $CLS --tape $T/p2_window.exact_f32 --score $F --out $O/id-p2-$arm.json > $O/id-p2-$arm.log 2>&1; echo "RC id p2 $arm=$?"
done
fi
[ -n "$NOTIME" ] && exit 0
GEN="--tape $T/p1_chat.exact_f32 --gen 32"
[ "$PL" != 0 ] && GEN="--tape $T/p3_logscale.exact_f32 --prompt-len $PL --context $(( (PL + 64 + 1023) / 1024 * 1024 )) --gen 32"
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for i in 1 2 3; do
  order="base new"; [ $((i % 2)) -eq 0 ] && order="new base"
  for arm in $order; do
    B=$BASE; F=$BF; [ $arm = new ] && { B=$NEW; F=$NF; }
    echo "MARK $arm-$i start $(date +%T.%N)"
    $B/ink2-gpu $M --class $CLS $GEN $F --out $O/gen-$arm-$i.json > $O/gen-$arm-$i.log 2>&1; echo "RC gen $arm $i=$?"
    echo "MARK $arm-$i end $(date +%T.%N)"
  done
done
kill $SMI
