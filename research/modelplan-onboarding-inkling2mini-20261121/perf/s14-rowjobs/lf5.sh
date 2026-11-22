#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a27-bin
$B/ink2-gpu-gate > $P/a27-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a27-gate.txt; grep FAIL $P/a27-gate.txt | cut -c1-300
[ $g -ne 0 ] && exit 3
O=$P/gf5-id; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for ch in 1 4 512; do
    tape=p1_chat.exact_f32; [ $ch = 512 ] && tape=p2_window.exact_f32
    for arm in base new; do
      X=$P/a26-bin; [ $arm = new ] && X=$B
      $X/ink2-gpu $M --class $cls --tape $T/$tape --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > /dev/null 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
done
bash $P/cls2.sh gf5-rig nvfp4-rig $P/a26-bin "" $B "" > $P/gf5-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gf5-sh shipped $P/a26-bin "" $B "" > $P/gf5-sh.log 2>&1; echo RC_sh=$?
O=$P/prof-bk5; mkdir -p $O
nsys profile -f true -o $O/nvfp4-rig-c1 --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.exact_f32 --gen 32 --out $O/nvfp4-rig-c1.json > $O/nvfp4-rig-c1.log 2>&1; echo NSYS=$?
nsys export --type sqlite -f true -o $O/nvfp4-rig-c1.sqlite $O/nvfp4-rig-c1.nsys-rep > /dev/null 2>&1
