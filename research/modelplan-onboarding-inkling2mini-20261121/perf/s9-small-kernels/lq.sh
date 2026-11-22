#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
$P/a8-bin/ink2-gpu-gate > $P/a8-gate.txt 2>&1; echo GATE_RC=$?; tail -1 $P/a8-gate.txt; grep FAIL $P/a8-gate.txt
O=$P/gq-id; mkdir -p $O
for cls in shipped float nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      B=$P/a6-bin; [ $arm = new ] && B=$P/a8-bin
      $B/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > $O/id-$cls-c$ch-$arm.log 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
done
bash $P/cls2.sh gq-rig nvfp4-rig $P/a6-bin "" $P/a8-bin "" > $P/gq-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gq-sh shipped $P/a6-bin "" $P/a8-bin "" > $P/gq-sh.log 2>&1; echo RC_sh=$?
O=$P/prof-a8; mkdir -p $O
nsys profile -f true -o $O/nvfp4-rig --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $P/a8-bin/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.exact_f32 --gen 24 --out $O/nvfp4-rig.json > $O/nvfp4-rig.log 2>&1; echo NSYS=$?
nsys stats -f csv -r cuda_gpu_kern_sum -o $O/nvfp4-rig $O/nvfp4-rig.nsys-rep > /dev/null 2>&1
