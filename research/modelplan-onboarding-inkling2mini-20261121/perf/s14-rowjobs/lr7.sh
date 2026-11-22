#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a28-bin
$B/ink2-gpu-gate > $P/a28-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a28-gate.txt; grep FAIL $P/a28-gate.txt | cut -c1-300
[ $g -ne 0 ] && exit 3
O=$P/gr7-id; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for ch in 1 2 3 4 5 8; do
    $P/a26-bin/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --prec mix=f32,epilogue=f32 --out $O/id-$cls-c$ch-base.json > /dev/null 2>&1; echo "RC $cls c$ch base=$?"
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --out $O/id-$cls-c$ch-new.json > /dev/null 2>&1; echo "RC $cls c$ch new=$?"
  done
done
O=$P/prof-rows7; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for R in 1 2 4 8; do
    nsys profile -f true -o $O/$cls-r$R --trace=cuda --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $R --eager --out $O/$cls-r$R.json > $O/$cls-r$R.log 2>&1; echo "NSYS $cls $R=$?"
    nsys export --type sqlite -f true -o $O/$cls-r$R.sqlite $O/$cls-r$R.nsys-rep > /dev/null 2>&1
  done
done
