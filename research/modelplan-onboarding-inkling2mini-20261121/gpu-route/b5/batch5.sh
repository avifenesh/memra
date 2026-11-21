#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen5
M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh
O=$L/b5
mkdir -p $O
cat $BIN/SHA256SUMS
cat /proc/loadavg
bash $C; rc=$?
echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then
  sudo -n systemctl restart nvidia-powerd
  waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"
fi
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
for i in 1 2 3; do
  echo "MARK float-$i $(date +%T.%N)"
  $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --gen 32 --out $O/gen-float-$i.json; echo "RC=$?"
  echo "MARK declared-$i $(date +%T.%N)"
  $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --gen 32 --out $O/gen-declared-$i.json; echo "RC=$?"
done
echo "MARK end-timed $(date +%T.%N)"
$BIN/ink2-gpu $M --class declared --tape $T/p3_logscale.actq_f32 --context 8448 --score --force-router --force-moe-in --out $O/p3-forced-refusal.json; echo "RC_EXPECT_REFUSAL=$?"
$BIN/ink2-gpu $M --class float --tape $T/p3_logscale.exact_f32 --context 8448 --score --out $O/p3-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p3_logscale.exact_f32 --context 8448 --score --out $O/p3-declared-vs-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p2_floor512.exact_f32 --score --out $O/p2floor512-float.json 2>&1 | tail -3; echo "RC=$?"
kill $SMI
