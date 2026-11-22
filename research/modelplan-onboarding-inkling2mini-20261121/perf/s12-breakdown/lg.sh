#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a23-bin
$B/ink2-gpu-gate > $P/a23-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a23-gate.txt; grep -E 'FAIL|attn' $P/a23-gate.txt | cut -c1-260
[ $g -ne 0 ] && exit 3
O=$P/gg-id; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      X=$P/a22-bin; [ $arm = new ] && X=$B
      $X/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > /dev/null 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
  for arm in base new; do
    X=$P/a22-bin; [ $arm = new ] && X=$B
    $X/ink2-gpu $M --class $cls --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --out $O/id-$cls-p3-$arm.json > /dev/null 2>&1; echo "RC $cls p3 $arm=$?"
  done
done
bash $P/cls2.sh gg-rig nvfp4-rig $P/a22-bin "" $B "" > $P/gg-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gg-rig4k nvfp4-rig $P/a22-bin "" $B "" 4096 > $P/gg-rig4k.log 2>&1; echo RC_rig4k=$?
bash $P/cls2.sh gg-sh4k shipped $P/a22-bin "" $B "" 4096 > $P/gg-sh4k.log 2>&1; echo RC_sh4k=$?
O=$P/prof-bk2; mkdir -p $O
for cls in nvfp4-rig shipped; do
  nsys profile -f true -o $O/$cls-4k --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p3_logscale.exact_f32 --prompt-len 4096 --context 5120 --gen 16 --out $O/$cls-4k.json > $O/$cls-4k.log 2>&1; echo "NSYS $cls 4k=$?"
  nsys export --type sqlite -f true -o $O/$cls-4k.sqlite $O/$cls-4k.nsys-rep > /dev/null 2>&1
done
