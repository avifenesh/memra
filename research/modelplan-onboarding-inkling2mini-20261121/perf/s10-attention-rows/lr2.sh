#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a17-bin
$B/ink2-gpu-gate > $P/a17-gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $P/a17-gate.txt; grep -E 'FAIL|gemv-rows' $P/a17-gate.txt | cut -c1-200
[ $g -ne 0 ] && exit 3
O=$P/gr2-id; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for ch in 1 2 3 4 8; do
    for arm in base new; do
      X=$P/a16-bin; [ $arm = new ] && X=$B
      $X/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --out $O/id-$cls-c$ch-$arm.json > /dev/null 2>&1; echo "RC $cls c$ch $arm=$?"
    done
  done
done
O=$P/prof-rows2; mkdir -p $O
for cls in shipped nvfp4-rig; do
  for R in 1 2 4 8; do
    nsys profile -f true -o $O/$cls-r$R --trace=cuda --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $R --eager --out $O/$cls-r$R.json > $O/$cls-r$R.log 2>&1; echo "NSYS $cls $R=$?"
    nsys export --type sqlite -f true -o $O/$cls-r$R.sqlite $O/$cls-r$R.nsys-rep > /dev/null 2>&1
  done
done

# chunk invariance of each arm of the adopted program (p1, chunk 1 vs 4 vs 8), and the old program on a17 vs a13
O=$P/gr2-inv; mkdir -p $O
for spec in 'controller=f32,norm=f32,conv=f64,router=f64,attn=f64g' 'controller=f32,norm=f32,conv=f64,router=f64,attn=f32t' 'controller=f32,norm=f32,conv=f32c,router=f64,attn=f64g' 'controller=f32,norm=f32,conv=f64,router=f32c,attn=f64g' 'controller=f32,norm=f32,conv=f32c,router=f32c,attn=f32t'; do
  n=$(echo $spec | tr ',=' '__')
  for ch in 1 4 8; do
    $B/ink2-gpu $M --class shipped --tape $T/p1_chat.exact_f32 --score --chunk $ch --prec "$spec" --out $O/$n-c$ch.json > /dev/null 2>&1; echo "INV $n c$ch=$?"
  done
done
for cls in shipped nvfp4-rig; do
  $P/a13-bin/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --chunk 512 --out $O/old-$cls-a13.json > /dev/null 2>&1
  $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --chunk 512 --prec 'controller=f32,norm=f32,conv=f64,router=f64,attn=f64g' --out $O/old-$cls-a17.json > /dev/null 2>&1; echo "OLD $cls=$?"
done
