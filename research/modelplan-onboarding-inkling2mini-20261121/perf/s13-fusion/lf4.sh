#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a26-bin
$B/ink2-gpu-gate > $P/a26-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a26-gate.txt; grep FAIL $P/a26-gate.txt | cut -c1-300
[ $g -ne 0 ] && exit 3
O=$P/gf4-id; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      X=$P/a23-bin; [ $arm = new ] && X=$B
      $X/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > /dev/null 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
  for arm in base new; do
    X=$P/a23-bin; [ $arm = new ] && X=$B
    $X/ink2-gpu $M --class $cls --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --out $O/id-$cls-p3-$arm.json > /dev/null 2>&1; echo "RC $cls p3 $arm=$?"
  done
done
bash $P/cls2.sh gf4-rig nvfp4-rig $P/a23-bin "" $B "" > $P/gf4-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gf4-rigm nvfp4-rig $B "" $B "--prec mix=f32,epilogue=f32" > $P/gf4-rigm.log 2>&1; echo RC_rigm=$?
bash $P/cls2.sh gf4-rigm4k nvfp4-rig $B "" $B "--prec mix=f32,epilogue=f32" 4096 > $P/gf4-rigm4k.log 2>&1; echo RC_rigm4k=$?
bash $P/cls2.sh gf4-shm shipped $B "" $B "--prec mix=f32,epilogue=f32" > $P/gf4-shm.log 2>&1; echo RC_shm=$?
# pins of the program with mix and epilogue in f32 (the candidate default), and its chunk invariance
Q=$P/pin-mixepi; mkdir -p $Q
for cls in shipped float nvfp4-rig; do
  for ch in 1 4 8 512; do
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --prec mix=f32,epilogue=f32 --out $Q/$cls-p1-c$ch.json > /dev/null 2>&1; echo "PIN $cls p1 c$ch=$?"
  done
  $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --chunk 512 --prec mix=f32,epilogue=f32 --out $Q/$cls-p2-c512.json > /dev/null 2>&1; echo "PIN $cls p2=$?"
done
$B/ink2-gpu $M --class nvfp4-rig --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --prec mix=f32,epilogue=f32 --out $Q/nvfp4-rig-p3-c1024.json > /dev/null 2>&1; echo PIN p3=$?
