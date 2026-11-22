#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a11-bin
$B/ink2-gpu-gate > $P/a11-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a11-gate.txt; grep FAIL $P/a11-gate.txt | cut -c1-300
[ $g -ne 0 ] && exit 3
O=$P/gt-id; mkdir -p $O
for cls in shipped float nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      X=$P/a9-bin; [ $arm = new ] && X=$B
      $X/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > $O/id-$cls-c$ch-$arm.log 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
done
for arm in base new; do
  X=$P/a9-bin; [ $arm = new ] && X=$B
  $X/ink2-gpu $M --class nvfp4-rig --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --out $O/id-rig-p3-$arm.json > $O/id-rig-p3-$arm.log 2>&1; echo "RC p3 $arm=$?"
done
bash $P/cls2.sh gt-4k-fold nvfp4-rig $P/a9-bin "" $B "" 4096 > $P/gt-4k-fold.log 2>&1; echo RC_4k_fold=$?
bash $P/cls2.sh gt-4k-f32 nvfp4-rig $B "" $B "--prec attn=f32" 4096 > $P/gt-4k-f32.log 2>&1; echo RC_4k_f32=$?
bash $P/cls2.sh gt-4k-f32t nvfp4-rig $B "" $B "--prec attn=f32t,conv=f32c,router=f32c" 4096 > $P/gt-4k-f32t.log 2>&1; echo RC_4k_f32t=$?
