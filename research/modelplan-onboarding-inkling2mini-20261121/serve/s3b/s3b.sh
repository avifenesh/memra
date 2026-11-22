#!/bin/bash
# s3 with the split attention falling back to the tiled form when a sequence outgrows the
# load context (the server failed to boot in s3: memra_ink2g_attn 45025). After the merge of lane P's decode (memra#1399): the batching gates, the route's numerics against
# the pre-merge receipts (probe tape, head rows on every corpus document, MoE reference) and the
# served c=1/2/4 timing under clampcheck, on the frozen merged binaries.
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
BIN=$L/serve-frozen4
M=/home/avifenesh/models/inkling2-mini
C=$L/ceiling/clampcheck.sh
R=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121
CHECK=$R/serve/serve_check.py
CP=$R/checkpoint-parity
O=$R/serve/s3b
HD=$L/cp-hidden-merged-b
mkdir -p $O $HD; cd $O
cat $BIN/SHA256SUMS $BIN/COMMIT
echo "loadavg $(cat /proc/loadavg)"
bash $C; rc=$?; echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then
  sudo -n systemctl restart nvidia-powerd
  waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"
fi
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > smi.csv 2>&1 &
SMI=$!
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep -v '^ink2-gpu: class\|^ink2-gpu: memory' $tag.log | tail -n 3; }
run batch-declared $BIN/ink2-gpu $M --class shipped --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-declared.json
run batch-long-declared $BIN/ink2-gpu $M --class shipped --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-declared.json
PROBE=$(sed -n 's/^tokens\t//p' $CP/probe/lane-e-cpu-actq.tsv | tr ',' ' ')
run probe-native env MEMRA_ORACLE_OUT=$O/native-route.tsv MEMRA_FULL_PREC=1 $BIN/ink2-gpu $M $PROBE
cmp $O/native-route.tsv $CP/probe/native-route.tsv && echo "PROBE_TAPE_BIT_EQUAL_TO_PRE_MERGE" || echo "PROBE_TAPE_DIFFERS"
for s in oracle rawdoc chatwrap; do
  case $s in oracle) J=$CP/oracle-prompts.json;; *) J=$L/klsens/corpus/$s.json;; esac
  run dump-$s $BIN/ink2-gpu $M --class shipped --dump-hidden $J $HD/$s
  n=0; d=0; for f in $L/cp-hidden/$s/*.f32; do n=$((n+1)); cmp -s $f $HD/$s/$(basename $f) || d=$((d+1)); done
  echo "HEAD_ROWS $s: $((n-d)) of $n documents bit-equal to the pre-merge dump"
done
echo "MARK server-boot $(date +%T.%N)"
MEMRA_REWRITE_BUNDLE=$R/serve/verify-serve MEMRA_MODELS=ink2=$M MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $BIN/memra-server > server.log 2>&1 &
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
echo S3_DONE
