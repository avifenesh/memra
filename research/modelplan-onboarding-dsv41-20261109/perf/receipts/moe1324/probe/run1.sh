#!/bin/bash
# The four-card box: the memra#1324 probe on one card. Holds that card's box lock (flock -n: refuses when held), 15-minute cap,
# 250 ms telemetry, then releases. Order: main base base main (each binary in its own process).
# usage: run1.sh GPU OUTDIR ROUNDS ROWS [VARIANTS] [ORDER]
set -u
G=$1 O=$2 ROUNDS=${3:-6} ROWS=${4:-1,2,4,8,9,10,12,16} VARS=${5:-} ORDER=${6:-"main base base main"}
P=/home/access/dsv41/moe1324/probe
EV=/mnt/Data/work/evals
mkdir -p $O
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G
say() { echo "$(date +%T) $*" | tee -a $O/runs.txt; }
[ -e /mnt/Data/work/memra-wrx-ds/QUIET ] && { say "QUIET set: not run"; exit 3; }
exec {fd}>>"$EV/gpu$G.lock"
flock -n "$fd" || { say "ABORT: gpu$G.lock held by someone else; nothing run"; exit 2; }
say "START gpu$G locked; load $(cut -d' ' -f1-3 /proc/loadavg); apps [$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')]"
nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
DL=$(( $(date +%s) + 870 ))
i=0
for b in $ORDER; do
  i=$((i+1))
  left=$(( DL - $(date +%s) ))
  [ $left -gt 30 ] || { say "SKIP run$i-$b (budget)"; continue; }
  if [ $b = base ]; then
    taskset -c 28-31,60-63 timeout $left $P/bin/probe-base 0 $ROUNDS $ROWS > $O/run$i-$b.txt 2>&1
  else
    taskset -c 28-31,60-63 timeout $left $P/bin/probe-$b 0 $ROUNDS $ROWS $VARS > $O/run$i-$b.txt 2>&1
  fi
  say "run$i $b rc $?"
done
kill $SMI
flock -u "$fd"
say "DONE gpu$G released"
