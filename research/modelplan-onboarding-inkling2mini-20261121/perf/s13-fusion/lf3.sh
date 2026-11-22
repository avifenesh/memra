#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a25-bin
$B/ink2-gpu-gate > $P/a25-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a25-gate.txt; grep -E 'FAIL|attn' $P/a25-gate.txt | cut -c1-260
[ $g -ne 0 ] && exit 3
O=$P/gf3-id; mkdir -p $O
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
bash $P/cls2.sh gf3-rig nvfp4-rig $P/a23-bin "" $B "" > $P/gf3-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gf3-rig4k nvfp4-rig $P/a23-bin "" $B "" 4096 > $P/gf3-rig4k.log 2>&1; echo RC_rig4k=$?
bash $P/cls2.sh gf3-sh4k shipped $P/a23-bin "" $B "" 4096 > $P/gf3-sh4k.log 2>&1; echo RC_sh4k=$?
O=$P/prof-bk4; mkdir -p $O
for cls in nvfp4-rig shipped; do
  nsys profile -f true -o $O/$cls-c1 --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --gen 32 --out $O/$cls-c1.json > $O/$cls-c1.log 2>&1; echo "NSYS $cls c1=$?"
  nsys export --type sqlite -f true -o $O/$cls-c1.sqlite $O/$cls-c1.nsys-rep > /dev/null 2>&1
done
