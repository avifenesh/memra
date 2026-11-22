#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a11-bin
grep -q 'ALL OK' $P/a11-gate.txt || { echo "a10 gate not OK"; exit 3; }
O=$P/gt-long; mkdir -p $O
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for arm in f32t f32 f64g; do
  case $arm in f32t) SPEC="attn=f32t,conv=f32c,router=f32c";; *) SPEC="attn=$arm";; esac
  for PL in 8192 32768; do
    echo "MARK $arm-$PL start $(date +%T.%N)"
    $B/ink2-gpu $M --class nvfp4-rig --tape $T/p3_logscale.exact_f32 --prompt-len $PL --context $(( (PL + 64 + 1023) / 1024 * 1024 )) --gen 16 --prec $SPEC --out $O/$arm-$PL.json > $O/$arm-$PL.log 2>&1; echo "RC $arm $PL=$?"
    echo "MARK $arm-$PL end $(date +%T.%N)"
  done
done
kill $SMI

O=$P/ncu-gv; mkdir -p $O
sudo -n /usr/local/cuda/bin/ncu --kernel-name regex:gemv_nvfp4_jobs_kernel --launch-skip 40 --launch-count 4 --set full -f -o $O/gv $B/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.exact_f32 --gen 4 --eager --out $O/gv.json > $O/ncu.log 2>&1; echo NCU=$?
sudo -n /usr/local/cuda/bin/ncu --kernel-name regex:mix_dot_kernel --launch-skip 40 --launch-count 2 --set full -f -o $O/mix $B/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.exact_f32 --gen 4 --eager --out $O/mix.json > $O/ncu-mix.log 2>&1; echo NCU_MIX=$?
sudo -n /usr/local/cuda/bin/ncu --kernel-name regex:ink2_expert_gemm_kernel --launch-skip 40 --launch-count 4 --set full -f -o $O/ex $B/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.exact_f32 --gen 4 --eager --out $O/ex.json > $O/ncu-ex.log 2>&1; echo NCU_EX=$?
sudo -n chown -R avifenesh $O
