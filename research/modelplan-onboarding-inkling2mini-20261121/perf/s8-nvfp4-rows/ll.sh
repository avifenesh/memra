#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; K=/home/avifenesh/.cache/inkling2mini/klsens/corpus
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
$P/a6-bin/ink2-gpu-gate > $P/a6-gate.txt 2>&1; echo GATE_RC=$?; tail -1 $P/a6-gate.txt; grep FAIL $P/a6-gate.txt
O=$P/gl-id; mkdir -p $O
for cls in shipped float nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      B=$P/a4-bin; [ $arm = new ] && B=$P/a6-bin
      $B/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > $O/id-$cls-c$ch-$arm.log 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
done
B=$P/m1-bin/ink2-gpu; A='controller=f32,norm=f32,attn=f64g'
for c in rawdoc chatwrap; do
  $B $M --class float --tape $T/p1_chat.exact_f32 --context 2048 --corpus $K/$c.json --arms "$A" --out $P/cum/$c-set2.json > $P/cum/$c-set2.log 2>&1; echo RC_$c=$?
done
$B $M --class float --tape $T/p1_chat.exact_f32 --context 2048 --corpus $P/corpus-p2.json --arms "$A" --out $P/cum/p2-set2.json > $P/cum/p2-set2.log 2>&1; echo RC_p2=$?
bash $P/cls2.sh gl-rig nvfp4-rig $P/a4-bin "" $P/a6-bin "" 0 > $P/gl-rig.log 2>&1; echo RC_rig=$?
