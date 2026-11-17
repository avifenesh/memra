#!/bin/bash
# The four-card box, memra#1324: one four-card step of the lane through bin/perf_cfix_four.sh, every box lock held with flock -n
# (a held lock aborts: nothing run), a hard budget under 15 minutes, all four released at the end.
#   gates: arm B's dsv41-gate kernels (first card) and tp gates (default, peer-pull), no timed runs
#   ab:    A B B A B A A B, device-push, one-row decode + rows 4 8 9 10 12 16, 40 layers, no gates
# usage: moe_arm.sh PHASE TAG ARM_A ARM_B
# Start: (cd /home/access/dsv41 && nohup setsid bash moe1324/moe_arm.sh ab moe1324-ab1 main25 moe1324a) > LOG 2>&1 < /dev/null &
set -u
PHASE=$1 TAG=$2 A=$3 B=$4
D=/home/access/dsv41
EV=/mnt/Data/work/evals
cd $D || exit 1
say() { echo "$(date +%T) $*"; }
for t in $A $B; do [ -x perf-bins/$t/dsv41-tpbench ] || { say "missing perf-bins/$t"; exit 2; }; done
[ -e /mnt/Data/work/memra-wrx-ds/QUIET ] && { say "QUIET set: not run"; exit 3; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$apps" ] || { say "ABORT: compute apps on the box: $apps"; exit 3; }
for i in 0 1 2 3; do
  exec {fd}>>"$EV/gpu$i.lock"
  if ! flock -n "$fd"; then say "ABORT: gpu$i.lock is held by someone else; nothing run"; exit 2; fi
  eval "FD$i=$fd"
done
export TP4_HELD=1
say "MOE-START $PHASE $TAG A=$A B=$B: all four box locks held; load $(cut -d' ' -f1-3 /proc/loadavg)"
nvidia-smi --query-gpu=index,power.limit,clocks.max.sm,temperature.gpu --format=csv,noheader
if [ $PHASE = gates ]; then
  env K3_ARM_A=$A K3_ARM_B=$B K3_TAG=$TAG K3_KGATE=1 K3_ORDER=" " K3_ROWS="4" bash bin/perf_cfix_four.sh 600
else
  env K3_ARM_A=$A K3_ARM_B=$B K3_TAG=$TAG K3_GATES=0 K3_ORDER="A B B A B A A B" K3_ROWS="4 8 9 10 12 16" \
    bash bin/perf_cfix_four.sh 840
fi
rc=$?
for i in 0 1 2 3; do eval "fd=\$FD$i"; flock -u "$fd"; eval "exec $fd>&-"; done
touch $EV/dsv41.moe1324-$PHASE-done
say "MOE-DONE $PHASE $TAG rc $rc: all four box locks released"
