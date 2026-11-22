#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a18-bin
grep -q 'ALL OK' $P/a18-gate.txt || { echo "a18 gate not OK"; exit 3; }
bash $P/cls2.sh gf-rig nvfp4-rig $B "--prec all=f64" $B "" > $P/gf-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gf-sh shipped $B "--prec all=f64" $B "" > $P/gf-sh.log 2>&1; echo RC_sh=$?
bash $P/cls2.sh gf-fl float $B "--prec all=f64" $B "" > $P/gf-fl.log 2>&1; echo RC_fl=$?
bash $P/cls2.sh gf-rig4k nvfp4-rig $B "--prec all=f64" $B "" 4096 > $P/gf-rig4k.log 2>&1; echo RC_rig4k=$?
bash $P/cls2.sh gf-sh4k shipped $B "--prec all=f64" $B "" 4096 > $P/gf-sh4k.log 2>&1; echo RC_sh4k=$?
O=$P/gf-long; mkdir -p $O
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for cls in nvfp4-rig shipped; do
  for PL in 512 2048 8192 32768; do
    echo "MARK $cls-$PL start $(date +%T.%N)"
    $B/ink2-gpu $M --class $cls --tape $T/p3_logscale.exact_f32 --prompt-len $PL --context $(( (PL + 64 + 1023) / 1024 * 1024 )) --gen 16 --out $O/$cls-$PL.json > $O/$cls-$PL.log 2>&1; echo "RC $cls $PL=$?"
    echo "MARK $cls-$PL end $(date +%T.%N)"
  done
done
kill $SMI
