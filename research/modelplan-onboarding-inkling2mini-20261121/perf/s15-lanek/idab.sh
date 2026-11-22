#!/bin/bash
# usage: idab.sh NAME NEWBIN BASEBIN [ab]   (lane K: gate, identity vs pins + base, optional A/B)
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
PIN=/home/avifenesh/.cache/inkling2mini/wt-perf/research/modelplan-onboarding-inkling2mini-20261121/perf/pin-mixepi
N=$1; B=$2; BASE=$3; AB=$4; O=$P/$N; mkdir -p $O
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
$B/ink2-gpu-gate > $O/gate.txt 2>&1; g=$?; echo GATE_RC=$g; tail -1 $O/gate.txt; grep FAIL $O/gate.txt | cut -c1-300
[ $g -ne 0 ] && exit 3
for cls in shipped nvfp4-rig; do
  for ch in 1 4 8 512; do
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --out $O/$cls-p1-c$ch.json > /dev/null 2>&1; echo "RC $cls p1 c$ch=$? $(python3 $P/dcmp.py $PIN/$cls-p1-c$ch.json $O/$cls-p1-c$ch.json)"
  done
  $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --chunk 512 --out $O/$cls-p2-c512.json > /dev/null 2>&1; echo "RC $cls p2 c512=$? $(python3 $P/dcmp.py $PIN/$cls-p2-c512.json $O/$cls-p2-c512.json)"
  for ch in 2 3; do
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --out $O/$cls-p1-c$ch.json > /dev/null 2>&1; r1=$?
    $BASE/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk $ch --out $O/$cls-p1-c$ch-base.json > /dev/null 2>&1; r2=$?
    echo "RC $cls p1 c$ch=$r1/$r2 $(python3 $P/dcmp.py $O/$cls-p1-c$ch-base.json $O/$cls-p1-c$ch.json)"
  done
done
$B/ink2-gpu $M --class nvfp4-rig --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --out $O/nvfp4-rig-p3-c1024.json > /dev/null 2>&1; echo "RC p3=$? $(python3 $P/dcmp.py $PIN/nvfp4-rig-p3-c1024.json $O/nvfp4-rig-p3-c1024.json)"
[ -z "$AB" ] && exit 0
bash $P/cls2.sh $N-rig nvfp4-rig $BASE "" $B "" > $P/$N-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh $N-sh shipped $BASE "" $B "" > $P/$N-sh.log 2>&1; echo RC_sh=$?
bash $P/cls2.sh $N-rig4k nvfp4-rig $BASE "" $B "" 4096 > $P/$N-rig4k.log 2>&1; echo RC_rig4k=$?
