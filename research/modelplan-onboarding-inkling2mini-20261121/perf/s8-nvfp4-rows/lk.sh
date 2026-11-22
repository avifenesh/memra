#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
$P/a4-bin/ink2-gpu-gate > $P/a4-gate.txt 2>&1; echo GATE_RC=$?; tail -1 $P/a4-gate.txt; grep -E 'FAIL|prec-adopted|gemv-rows' $P/a4-gate.txt
# multi-row path vs the tile path: calls of 4 rows, all-f64 on a4 against m1 (both classes)
O=$P/gk-rows; mkdir -p $O
for cls in shipped float nvfp4-rig; do
  for arm in base new; do
    if [ $arm = base ]; then B=$P/m1-bin/ink2-gpu; F=""; else B=$P/a4-bin/ink2-gpu; F="--prec all=f64"; fi
    $B $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk 4 $F --out $O/id-$cls-$arm.json > $O/id-$cls-$arm.log 2>&1; echo "RC $cls $arm=$?"
  done
done
O=$P/prof-dec; mkdir -p $O
for cls in nvfp4-rig shipped; do
  nsys profile -f true -o $O/$cls --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $P/a4-bin/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --gen 24 --out $O/$cls.json > $O/$cls.log 2>&1; echo NSYS_$cls=$?
  nsys stats -f csv -r cuda_gpu_kern_sum -o $O/$cls $O/$cls.nsys-rep > /dev/null 2>&1
done
