#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/serve-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/serve-frozen2
M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh
CHECK=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121/serve/serve_check.py
O=$L/s2
mkdir -p $O; cd $O
cat $BIN/SHA256SUMS
echo "loadavg $(cat /proc/loadavg)"
bash $C; rc=$?; echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then
  sudo -n systemctl restart nvidia-powerd
  waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"
fi
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > smi.csv 2>&1 &
SMI=$!
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep -v '^ink2-gpu: class\|^ink2-gpu: memory' $tag.log | tail -n 3; }
run batch-red $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --red batch-rows-rotated --out batch-red.json
run batch-long-declared $BIN/ink2-gpu $M --class declared --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-declared.json
run batch-long-float $BIN/ink2-gpu $M --class float --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-float.json
# the server
echo "MARK server-boot $(date +%T.%N)"
MEMRA_REWRITE_BUNDLE=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121/serve/verify-serve MEMRA_MODELS=ink2=$M MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $BIN/memra-server > server.log 2>&1 &
SRV=$!
if waitfor --http http://127.0.0.1:18341/readyz --timeout 5m; then
  echo "MARK serve-check $(date +%T.%N)"
  python3 $CHECK http://127.0.0.1:18341 lane-h-local ink2 $O --levels 1,2,4 --n 8 --max-tokens 128 > serve-check.log 2>&1; echo "RC serve-check=$?"
  tail -n 40 serve-check.log
  echo "MARK serve-check-end $(date +%T.%N)"
else
  echo "SERVER NOT READY"; tail -n 40 server.log
fi
kill $SRV
waitfor --gone $SRV --timeout 2m
kill $SMI
