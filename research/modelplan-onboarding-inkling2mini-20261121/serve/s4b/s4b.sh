#!/bin/bash
# s4's second part: the MoE reference check's red arms on both classes, and the served c=1/2/4 of both
# classes on the bundle s4's verify wrote (s4's servers refused the bundle from before #1401: its plan
# hash is the old plan's). Binaries: serve-frozen6.
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
BIN=$L/serve-frozen6
M=/home/avifenesh/models/inkling2-mini
C=$L/ceiling/clampcheck.sh
R=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121
CHECK=$R/serve/serve_check.py
O=$R/serve/s4b
mkdir -p $O; cd $O
cat $BIN/SHA256SUMS $BIN/COMMIT
bash $C; rc=$?; echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"; fi
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > smi.csv 2>&1 &
SMI=$!
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep 'moe reference\|rows differing' $tag.log | tail -n 2; }
run red-index-shipped $BIN/ink2-gpu $M --class shipped --tape $T/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --red moe-ref-index-order --out red-index-shipped.json
for cls in shipped nvfp4-rig; do
  run red-wad-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --red moe-ref-weight-after-down --out red-wad-$cls.json
done
# the probe under the route's precision program and under lane P's all-f64 reference arm, both classes
PROBE=$(sed -n 's/^tokens\t//p' $R/checkpoint-parity/probe/lane-e-cpu-actq.tsv | tr ',' ' ')
for cls in shipped nvfp4-rig; do
  run probe-$cls-default env MEMRA_ORACLE_OUT=$O/native-$cls-default.tsv $BIN/ink2-gpu --class $cls $M $PROBE
  run probe-$cls-f64 env MEMRA_ORACLE_OUT=$O/native-$cls-f64.tsv $BIN/ink2-gpu --class $cls --prec all=f64 $M $PROBE
done
for cls in shipped nvfp4-rig; do
  echo "MARK server-boot-$cls $(date +%T.%N)"
  MEMRA_INK2_CLASS=$cls MEMRA_REWRITE_BUNDLE=$R/checkpoint-parity/verify-s4 MEMRA_MODELS=ink2=$M MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $BIN/memra-server > server-$cls.log 2>&1 &
  SRV=$!
  if waitfor --http http://127.0.0.1:18341/readyz --timeout 5m; then
    mkdir -p $O/$cls
    python3 $CHECK http://127.0.0.1:18341 lane-h-local ink2 $O/$cls --levels 1,2,4 --n 8 --max-tokens 128 > serve-check-$cls.log 2>&1; echo "RC serve-check-$cls=$?"
    grep -E '^c=|ALL PASS|FAIL' serve-check-$cls.log | cut -c1-500
    echo "MARK serve-check-end-$cls $(date +%T.%N)"
  else
    echo "SERVER NOT READY ($cls)"; tail -n 5 server-$cls.log
  fi
  kill $SRV; waitfor --gone $SRV --timeout 2m
done
kill $SMI
echo S4B_DONE
