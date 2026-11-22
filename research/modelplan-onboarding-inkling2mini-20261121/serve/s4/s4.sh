#!/bin/bash
# The serve route on lane P's #1399 at 790355f639 (with #1401): batching gates for both served classes,
# the MoE reference, probe tapes, head-row dumps, served c=1/2/4 for both classes under clampcheck, and
# verify checkpoint + verify serve for the shipped class. Frozen binaries in serve-frozen5.
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
BIN=$L/serve-frozen5
M=/home/avifenesh/models/inkling2-mini
C=$L/ceiling/clampcheck.sh
R=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121
CHECK=$R/serve/serve_check.py
CP=$R/checkpoint-parity
O=$R/serve/s4
HD=$L/cp-hidden-s4
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
for cls in shipped nvfp4-rig; do
  run batch-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-$cls.json
  run batch-$cls-p2 $BIN/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --batch-check 6,24,5 --out batch-$cls-p2.json
  run batch-long-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-$cls.json
  run moe-ref-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --out moe-ref-$cls.json
done
run batch-red $BIN/ink2-gpu $M --class shipped --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --red batch-rows-rotated --out batch-red.json
run moe-ref-red-rig $BIN/ink2-gpu $M --class nvfp4-rig --tape $T/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --red moe-ref-index-order --out moe-ref-red-rig.json
PROBE=$(sed -n 's/^tokens\t//p' $CP/probe/lane-e-cpu-actq.tsv | tr ',' ' ')
run probe-shipped env MEMRA_ORACLE_OUT=$O/native-shipped.tsv MEMRA_FULL_PREC=1 $BIN/ink2-gpu $M $PROBE
run probe-rig env MEMRA_ORACLE_OUT=$O/native-nvfp4-rig.tsv MEMRA_FULL_PREC=1 $BIN/ink2-gpu --class nvfp4-rig $M $PROBE
for cls in shipped nvfp4-rig; do
  for s in oracle rawdoc chatwrap; do
    case $s in oracle) J=$CP/oracle-prompts.json;; *) J=$L/klsens/corpus/$s.json;; esac
    run dump-$cls-$s $BIN/ink2-gpu $M --class $cls --dump-hidden $J $HD/$cls/$s
  done
done
for cls in shipped nvfp4-rig; do
  echo "MARK server-boot-$cls $(date +%T.%N)"
  MEMRA_INK2_CLASS=$cls MEMRA_REWRITE_BUNDLE=$R/serve/verify-serve MEMRA_MODELS=ink2=$M MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $BIN/memra-server > server-$cls.log 2>&1 &
  SRV=$!
  if waitfor --http http://127.0.0.1:18341/readyz --timeout 5m; then
    echo "MARK serve-check-$cls $(date +%T.%N)"
    mkdir -p $O/$cls
    python3 $CHECK http://127.0.0.1:18341 lane-h-local ink2 $O/$cls --levels 1,2,4 --n 8 --max-tokens 128 > serve-check-$cls.log 2>&1; echo "RC serve-check-$cls=$?"
    grep -E '^c=|ALL PASS|FAIL' serve-check-$cls.log | cut -c1-400
    echo "MARK serve-check-end-$cls $(date +%T.%N)"
  else
    echo "SERVER NOT READY ($cls)"; tail -n 20 server-$cls.log
  fi
  kill $SRV
  waitfor --gone $SRV --timeout 2m
done
V=$CP/verify-s4
mkdir -p $V
cp $BIN/SHA256SUMS $BIN/COMMIT $V/
run verify-checkpoint $BIN/memra model verify checkpoint $M --against inkling2 --out $V --oracle $CP/probe/lane-e-cpu-actq.tsv --native-runner $BIN/ink2-gpu
run verify-serve env MEMRA_CTX=8192 MEMRA_INK2_CLASS=shipped $BIN/memra model verify serve $M --against inkling2 --out $V --native-runner $BIN/memra-server
kill $SMI
echo S4_DONE
