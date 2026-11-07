#!/bin/bash
# Quiet-slot cells, 2026-11-07, after the box's training lane's "quiet": the timed half on GPU 1, the gates on GPU 2:
#   1. kv-vmm TIMED half (run-timed.sh into the 2026-11-05 untimed root), through power-cell.sh at 300 W
#      with REASSERT=1 (the box's cap timer returns cards to 280 W every 10 min).
#   2. spec-cache-mixed-gate.sh and dspark-serve-smoke.sh as fixed on fix4/gates 585f1f2d77 (tools7/),
#      on the bin6 server (fix stack at 99c65ab0b2, the binary their failing run used), each through
#      power-cell.sh at 300 W. Untimed.
# Starts only on a non-empty quiet-go flag. The timed half (GPU 1) must end by 14:15; the box's next training
# arm starts on GPUs 0, 1, 3 when it reports. The untimed gates then run on GPU 2 (the eval card, free for
# about 3 h after that start) under its gpu2.lock, within 2 h. Deletes the staged drafter at the end.
set -u
W=<pair-kit>; M=<maxq-kit>
O=$W/outq; mkdir -p $O; S=$O/status.txt
K=$W/tools7; B=$W/bin6; DF=$W/dflash2-q38
Q38=$M/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
PC=$M/power/power-cell.sh
WF=<box-home>/.local/bin/waitfor
export PATH=<box-home>/.local/bin:$PATH
STOP=$(date -d '2026-11-07 14:15' +%s)      # the box is quiet until 14:15 (the box's training lane)
cleanup() { rm -rf <pair-kit>/dflash2-q38; echo "quiet cleaned $(date +%H:%M:%S)" >> $S; }
trap cleanup EXIT
echo "quiet armed $(date '+%m-%d %H:%M:%S')" >> $S
$WF --until "test -s $W/quiet-go" --timeout 4h --every 30s > $O/wait.log 2>&1 || { echo "quiet: no go flag" >> $S; exit 3; }
[ "$(date +%s)" -lt "$STOP" ] || { echo "quiet: too late" >> $S; exit 3; }
( cd $K && sha256sum -c --quiet tools7.sha256 ) > $O/tools-check.log 2>&1 || { echo "quiet: tools7 fails its manifest" >> $S; exit 3; }
( cd $B && sha256sum -c --quiet bin6.sha256 ) > $O/bin-check.log 2>&1 || { echo "quiet: bin6 fails its manifest" >> $S; exit 3; }
echo "quiet start $(date +%H:%M:%S); cards $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $S

# 1. kv-vmm timed half. run-timed.sh refuses unless its caller holds the card's window lock on FD 9.
exec 9><box-work>/evals/gpu1.lock; flock -w 600 9 || { echo "quiet: gpu1.lock busy" >> $S; exit 3; }
REMAIN=$(( (STOP - $(date +%s)) / 60 )); [ $REMAIN -gt 120 ] && REMAIN=120
REASSERT=1 bash $PC 1 300 kv-vmm-timed $O -- env OUT=$M/kv-vmm/out-20261105T071533Z CARD=1 WINDOW_MIN=$REMAIN \
    bash $M/kv-vmm/run-timed.sh > $O/kv-vmm-timed.log 2>&1
rc=$?
exec 9>&-
if [ $rc = 2 ] && grep -q REFUSED $O/kv-vmm-timed.log; then echo "quiet: kv-vmm-timed REFUSED: $(grep -m1 REFUSED $O/kv-vmm-timed.log | cut -c1-200)" >> $S; exit 3; fi
echo "kv-vmm-timed rc=$rc window_min=$REMAIN (TIMED HALF REPORTED $(date +%H:%M:%S)) :: $(cat $O/power-kv-vmm-timed.txt 2>/dev/null | cut -c1-300) :: $(head -n 1 $M/kv-vmm/out-20261105T071533Z/VERDICT.txt 2>/dev/null | cut -c1-300)" >> $S

# 2. the two fixed single-card gates on GPU 2: its eval lock on FD 8, the canonical lock on FD 9 (reused by both)
GEND=$(( $(date +%s) + 7200 ))
exec 8><box-work>/evals/gpu2.lock; flock -w 600 8 || { echo "quiet: gpu2.lock busy" >> $S; exit 3; }
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "quiet: memra-gpu.lock busy" >> $S; exit 3; }
[ "$(date +%s)" -lt "$((GEND - 2700))" ] || { echo "quiet: no time left for the gates" >> $S; exit 0; }
REASSERT=1 bash $PC 2 300 spec-cache-mixed $O -- env CUDA_VISIBLE_DEVICES=2 MEMRA_GPU_LOCK=/tmp/memra-gpu.lock \
    timeout $(( GEND - 1500 - $(date +%s) > 2400 ? 2400 : GEND - 1500 - $(date +%s) )) bash $K/spec-cache-mixed-gate.sh $Q38 $B/memra-server $O/spec-cache-mixed > $O/spec-cache-mixed.log 2>&1
echo "SPEC-CACHE-MIXED-Q rc=$? :: $(grep -a -E 'ok:|FAIL|UNAVAILABLE|GATE' $O/spec-cache-mixed.log | tr '\n' '|' | cut -c1-700)" >> $S
if [ -s "$DF/READY" ] && [ "$(date +%s)" -lt "$((GEND - 1500))" ]; then
    REASSERT=1 bash $PC 2 300 dspark-smoke $O -- env CUDA_VISIBLE_DEVICES=2 MEMRA_GPU_LOCK=/tmp/memra-gpu.lock \
        timeout $(( GEND - 120 - $(date +%s) > 1800 ? 1800 : GEND - 120 - $(date +%s) )) bash $K/dspark-serve-smoke.sh $Q38 $DF $B/memra-server $O/dspark-smoke > $O/dspark-smoke.log 2>&1
    echo "DSPARK-SMOKE-Q rc=$? :: $(grep -a -E 'LOW=1 tooth|FAIL|IDENTICAL|ALL GREEN|FAILED' $O/dspark-smoke.log | tr '\n' '|' | cut -c1-600)" >> $S
else
    echo "DSPARK-SMOKE-Q SKIP (no drafter or no time)" >> $S
fi
echo "quiet done $(date +%H:%M:%S); power $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader -i 1,2 | tr '\n' ' ')" >> $S
