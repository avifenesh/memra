#!/bin/bash
# usage: stepn.sh NAME BASEBIN ARM1BIN [ARM2BIN]: gate on the last arm, digests of each arm vs base,
# timed takes interleaved base/arms in rotating orders (3 takes each), 250 ms telemetry
N=$1; BASE=$2; shift 2; ARMS=("$@")
O=/home/avifenesh/.cache/inkling2mini/perf/$N
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh
mkdir -p $O
{ echo "base $BASE"; cat $BASE/SHA256SUMS; i=0; for a in "${ARMS[@]}"; do i=$((i+1)); echo "arm$i $a"; cat $a/SHA256SUMS; done; } > $O/binaries.sha256
cat /proc/loadavg
bash $C; rc=$?; echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"; fi
LAST=${ARMS[-1]}
$LAST/ink2-gpu-gate > $O/gate.txt 2>&1; echo "GATE_RC=$?"; tail -1 $O/gate.txt
names=(base); bins=($BASE); i=0; for a in "${ARMS[@]}"; do i=$((i+1)); names+=(arm$i); bins+=($a); done
for cls in declared float; do
  for k in "${!names[@]}"; do
    B=${bins[$k]}; arm=${names[$k]}
    $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --score --chunk 1 --out $O/id-$cls-p1c1-$arm.json > $O/id-$cls-p1c1-$arm.log 2>&1; echo "RC id $cls p1c1 $arm=$?"
    $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --score --out $O/id-$cls-p2-$arm.json > $O/id-$cls-p2-$arm.log 2>&1; echo "RC id $cls p2 $arm=$?"
  done
done
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
n=${#names[@]}
for i in 1 2 3; do
  order=$(seq 0 $((n-1))); [ $((i % 2)) -eq 0 ] && order=$(seq $((n-1)) -1 0)
  for k in $order; do
    B=${bins[$k]}; arm=${names[$k]}
    echo "MARK $arm-$i start $(date +%T.%N)"
    $B/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --gen 32 --out $O/gen-$arm-$i.json > $O/gen-$arm-$i.log 2>&1; echo "RC gen $arm $i=$?"
    echo "MARK $arm-$i end $(date +%T.%N)"
  done
done
kill $SMI
